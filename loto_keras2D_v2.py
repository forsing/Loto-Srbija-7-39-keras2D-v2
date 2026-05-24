from __future__ import annotations

"""
loto_keras2D_v2.py — Loto 7/39 predikcija sa Keras MLP (potpuno determinističko)

  • Multi-label cilj: 39 sigmoid izlaza + BCE (umesto 7 linearnih → prosek pozicija).
  • Feature engineering (sve iz prošlosti):
      - lag(5) flatten,
      - rolling frekvencije (W=20/50/100),
      - gap po broju,
      - statistike prošlog kola (suma, parnost, low/high, raspon).
  • Vremenski tačan split: poslednjih 100 kola = back-test.
  • Pos-weight u BCE (kompenzacija ~18% pozitivnih po sample-u).
  • Mini-ansambl: dve različite arhitekture (široka + uska/duboka) sa istim SEED,
    finalni skor = prosek sigmoida. Dropout + BatchNorm + L2 regularizacija.
  • Predikcija sledećeg kola iz POSLEDNJEG reda CSV-a (a ne testX[0]).
  • Post-processing: top-7 jedinstvenih, sortirano, 1..39.
  • Back-test metrike: hits/7, AUC (macro), LRAP.
  • Snimanje u loto_keras2D_v2_predikcija.txt (append, sa timestampom).
  • Determinizam: SEED=39, PYTHONHASHSEED, TF_DETERMINISTIC_OPS=1,
    tf.keras.utils.set_random_seed, BLAS na 1 nit, tf single-thread.

Pokretanje:
    python loto_keras2D_v2.py




ako je final_val_auc ≈ best_val_auc → svejedno je, uzmi bilo koje
ako je final_val_auc << best_val_auc → model je počeo da preobučava, BEST je objektivno pametniji izbor
ako je final_val_auc > best_val_auc → BEST nije ni potreban (FINAL je sam najbolji)
Praksa: uvek koristi BEST. FINAL je samo informativno, da vidiš da li si trenirao predugo.




Kako radi:
custom callback _KeepBestWeights pamti težine iz epohe sa najvećim val_auc (bez prekida treninga)
posle treninga čuvamo i BEST i FINAL težine za svaki model
predikcije i back-test se rade za sve 6 varijanti
Ispis u terminalu:

🧪 Back-test:
   model            hits/7   hit%   AUC   LRAP
   WIDE_best         X.XXX   X.X% X.XXX X.XXX
   WIDE_final        ...
   DEEP_best         ...
   DEEP_final        ...
   ENSEMBLE_best     ...
   ENSEMBLE_final    ...

🎯 Predikcija SLEDEĆEG kola:
      WIDE_best        -> [...]
      WIDE_final       -> [...]
      DEEP_best        -> [...]
      DEEP_final       -> [...]
   🏁 ENSEMBLE_best    -> [...]
   🏁 ENSEMBLE_final   -> [...]
"""

import os

SEED = 39
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # manje šuma
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import warnings
warnings.filterwarnings("ignore")

import random
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pytz

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Na M1/M2 Macu novi Adam je spor — koristimo legacy Adam ako postoji.
try:
    from tensorflow.keras.optimizers.legacy import Adam  # type: ignore
except Exception:
    from tensorflow.keras.optimizers import Adam  # fallback

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, label_ranking_average_precision_score

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
try:
    tf.keras.utils.set_random_seed(SEED)
except Exception:
    pass

# Single-thread TF da bude bit-egzaktan između pokretanja
try:
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
except Exception:
    pass


# ============================================================
# Konfiguracija
# ============================================================
CSV_PATH      = "/Users/4c/Desktop/GHQ/data/loto7_4620_k41.csv"
OUT_TXT       = "/Users/4c/Desktop/GHQ/KvantniRegresor/loto_keras2D_v2_predikcija.txt"
N_MIN, N_MAX  = 1, 39
K             = 7
LAG           = 5
WINDOWS       = (20, 50, 100)
BACKTEST_N    = 100
VAL_N         = 200       # vremenska validacija (poslednjih 200 trening uzoraka)
EPOCHS        = 4620
BATCH         = 64
LR            = 1e-3


def stamp() -> str:
    return datetime.now(pytz.timezone("Europe/Belgrade")).strftime("%d.%m.%Y_%H.%M.%S")


T0 = time.time()
print()
print("🔁 loto_keras2D_v2 — start ", stamp())
print()


# ============================================================
# 1) Učitavanje CSV-a (bez headera, 7 kolona)
# ============================================================
df = pd.read_csv(CSV_PATH, header=None)
df = df.iloc[:, :K].astype(int)
draws = df.values
N = draws.shape[0]
print(f"✅ CSV učitan: {CSV_PATH}")
print(f"   broj izvlačenja: {N}, brojeva po kolu: {K}")
print()


# ============================================================
# 2) Multi-hot (N, 39)
# ============================================================
def draws_to_multihot(rows: np.ndarray) -> np.ndarray:
    M = rows.shape[0]
    out = np.zeros((M, N_MAX), dtype=np.int8)
    for i in range(M):
        for v in rows[i]:
            if N_MIN <= v <= N_MAX:
                out[i, v - 1] = 1
    return out


Y_full = draws_to_multihot(draws)


# ============================================================
# 3) Feature engineering (samo iz prošlosti)
# ============================================================
def build_features(draws_arr, y_multi, lag=LAG, windows=WINDOWS):
    n, _ = draws_arr.shape
    feats = []
    for L in range(1, lag + 1):
        shifted = np.zeros_like(draws_arr)
        shifted[L:] = draws_arr[:-L]
        feats.append(shifted)
    lag_block = np.concatenate(feats, axis=1)

    cum = np.cumsum(y_multi, axis=0)
    rolling_blocks = []
    for W in windows:
        rolled = np.zeros_like(cum, dtype=float)
        rolled[1:W + 1] = cum[:W]
        rolled[W + 1:] = cum[W:-1] - cum[:-W - 1]
        rolling_blocks.append(rolled / float(W))
    roll_block = np.concatenate(rolling_blocks, axis=1)

    gap = np.zeros((n, N_MAX), dtype=float)
    last_seen = np.full(N_MAX, -1, dtype=int)
    for i in range(n):
        for k in range(N_MAX):
            gap[i, k] = (i - last_seen[k]) if last_seen[k] >= 0 else i + 1
        for v in draws_arr[i]:
            last_seen[v - 1] = i

    prev = np.zeros_like(draws_arr)
    prev[1:] = draws_arr[:-1]
    s_sum = prev.sum(axis=1, keepdims=True).astype(float)
    s_odd = (prev % 2 == 1).sum(axis=1, keepdims=True).astype(float)
    s_low = (prev <= 19).sum(axis=1, keepdims=True).astype(float)
    s_rng = (prev.max(axis=1, keepdims=True) - prev.min(axis=1, keepdims=True)).astype(float)
    stat_block = np.concatenate([s_sum, s_odd, s_low, s_rng], axis=1)

    return np.concatenate([lag_block, roll_block, gap, stat_block], axis=1)


X_full = build_features(draws, Y_full)
print(f"✅ Features: X_full.shape = {X_full.shape}, Y_full.shape = {Y_full.shape}")
print()

START = max(LAG, max(WINDOWS))
X_all = X_full[START:N].astype(float)
Y_all = Y_full[START:N].astype(float)

n_total = X_all.shape[0]
n_train = n_total - BACKTEST_N
assert n_train > VAL_N + 200, "Premalo podataka za train/val/back-test."

# vremenski split: trening — val — back-test (sve hronološki)
X_train_full, Y_train_full = X_all[:n_train],            Y_all[:n_train]
X_tr,         Y_tr         = X_train_full[:-VAL_N],      Y_train_full[:-VAL_N]
X_val,        Y_val        = X_train_full[-VAL_N:],      Y_train_full[-VAL_N:]
X_back,       Y_back       = X_all[n_train:],            Y_all[n_train:]

scaler = StandardScaler()
X_tr_s   = scaler.fit_transform(X_tr)
X_val_s  = scaler.transform(X_val)
X_back_s = scaler.transform(X_back)
X_next_s = scaler.transform(X_full[N - 1:N].astype(float))

print(f"   train={X_tr_s.shape[0]}  val={X_val_s.shape[0]}  back={X_back_s.shape[0]}")
print()


# ============================================================
# 4) Modeli — dve arhitekture (mini-ansambl)
# ============================================================
def build_wide(input_dim, seed=SEED):
    init = tf.keras.initializers.GlorotUniform(seed=seed)
    inp = Input(shape=(input_dim,))
    x = Dense(256, activation="relu", kernel_initializer=init, kernel_regularizer=l2(1e-5))(inp)
    x = BatchNormalization()(x)
    x = Dropout(0.3, seed=seed)(x)
    x = Dense(128, activation="relu", kernel_initializer=init, kernel_regularizer=l2(1e-5))(x)
    x = Dropout(0.3, seed=seed + 1)(x)
    out = Dense(N_MAX, activation="sigmoid", kernel_initializer=init)(x)
    m = Model(inp, out, name="wide")
    return m


def build_deep(input_dim, seed=SEED + 7):
    init = tf.keras.initializers.GlorotUniform(seed=seed)
    inp = Input(shape=(input_dim,))
    x = Dense(128, activation="relu", kernel_initializer=init, kernel_regularizer=l2(1e-5))(inp)
    x = BatchNormalization()(x)
    x = Dropout(0.25, seed=seed)(x)
    x = Dense(96, activation="relu", kernel_initializer=init, kernel_regularizer=l2(1e-5))(x)
    x = Dropout(0.25, seed=seed + 1)(x)
    x = Dense(64, activation="relu", kernel_initializer=init, kernel_regularizer=l2(1e-5))(x)
    x = Dropout(0.25, seed=seed + 2)(x)
    out = Dense(N_MAX, activation="sigmoid", kernel_initializer=init)(x)
    m = Model(inp, out, name="deep")
    return m


# Pos-weight u BCE: ~7/39 = 0.179 pozitivnih → daj veći weight pozitivnima
def make_loss():
    pos = float(Y_tr.sum() / Y_tr.size)
    w_pos = (1.0 - pos) / pos
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    def weighted_bce(y_true, y_pred):
        # sample/element weighting po klasi (1 → w_pos, 0 → 1)
        weights = y_true * w_pos + (1.0 - y_true) * 1.0
        eps = 1e-7
        y_pred_c = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        loss = -(y_true * tf.math.log(y_pred_c) + (1 - y_true) * tf.math.log(1 - y_pred_c))
        return tf.reduce_mean(loss * weights)

    return weighted_bce


class _KeepBestWeights(tf.keras.callbacks.Callback):
    """Pamti težine epohe sa najvećim val_auc — bez prekidanja treninga."""
    def __init__(self):
        super().__init__()
        self.best_auc = -np.inf
        self.best_epoch = 0
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        v = logs.get("val_auc")
        if v is not None and v > self.best_auc:
            self.best_auc = float(v)
            self.best_epoch = int(epoch) + 1
            self.best_weights = [w.copy() for w in self.model.get_weights()]


def train_one(builder, name):
    model = builder(X_tr_s.shape[1])
    model.compile(optimizer=Adam(LR), loss=make_loss(),
                  metrics=[tf.keras.metrics.AUC(name="auc")])
    # EarlyStopping izbačen — treniramo sve epohe; ali pamtimo BEST težine.
    keeper = _KeepBestWeights()
    cbs = [
        ReduceLROnPlateau(monitor="val_loss", patience=6, factor=0.5,
                          min_lr=1e-5, verbose=0),
        keeper,
    ]
    print(f"⚛️ Treniram {name} ({EPOCHS} epoha) ...")
    hist = model.fit(
        X_tr_s, Y_tr,
        validation_data=(X_val_s, Y_val),
        epochs=EPOCHS, batch_size=BATCH, verbose=0,
        shuffle=False,         # vremenski red — bez shuffle-a
        callbacks=cbs,
    )
    final_auc = float(hist.history["val_auc"][-1])
    final_weights = [w.copy() for w in model.get_weights()]
    best_weights  = keeper.best_weights if keeper.best_weights is not None else final_weights
    print(f"   ✅ {name}: epohe={EPOCHS}, best_epoch={keeper.best_epoch}, "
          f"best_val_auc={keeper.best_auc:.4f}, final_val_auc={final_auc:.4f}")
    return model, best_weights, final_weights


# ============================================================
# 5) Trening dva modela (BEST + FINAL težine za svaki)
# ============================================================
m_wide, w_wide_best, w_wide_final = train_one(build_wide, "WIDE")
m_deep, w_deep_best, w_deep_final = train_one(build_deep, "DEEP")
print()


def predict_with(model, weights, X):
    model.set_weights(weights)
    return model.predict(X, verbose=0)


# ============================================================
# 6) Top-K iz skorova
# ============================================================
def topk_from_scores(scores_1d, k=K):
    s = np.asarray(scores_1d, dtype=float).copy()
    order = np.lexsort((np.arange(N_MAX), -s))
    chosen = order[:k] + 1
    return np.sort(chosen)


# ============================================================
# 7) Back-test (BEST i FINAL za WIDE, DEEP, ENSEMBLE)
# ============================================================
print(f"🧪 Back-test (poslednjih {BACKTEST_N} izvlačenja):")

S = {
    "WIDE_best":  predict_with(m_wide, w_wide_best,  X_back_s),
    "WIDE_final": predict_with(m_wide, w_wide_final, X_back_s),
    "DEEP_best":  predict_with(m_deep, w_deep_best,  X_back_s),
    "DEEP_final": predict_with(m_deep, w_deep_final, X_back_s),
}
S["ENSEMBLE_best"]  = (S["WIDE_best"]  + S["DEEP_best"])  / 2.0
S["ENSEMBLE_final"] = (S["WIDE_final"] + S["DEEP_final"]) / 2.0

def avg_hits(scores_2d, Y):
    h = 0
    for i in range(scores_2d.shape[0]):
        true_set = set(np.where(Y[i] == 1)[0] + 1)
        pred_set = set(topk_from_scores(scores_2d[i]).tolist())
        h += len(true_set & pred_set)
    return h / scores_2d.shape[0]

def safe_auc(Y, scores):
    try:
        return roc_auc_score(Y, scores, average="macro")
    except Exception:
        return float("nan")

def safe_lrap(Y, scores):
    try:
        return label_ranking_average_precision_score(Y.astype(int), scores)
    except Exception:
        return float("nan")

order = ["WIDE_best", "WIDE_final", "DEEP_best", "DEEP_final",
         "ENSEMBLE_best", "ENSEMBLE_final"]
print(f"   {'model':<16} {'hits/7':>8} {'hit%':>7} {'AUC':>7} {'LRAP':>7}")
for name in order:
    s = S[name]
    h = avg_hits(s, Y_back); a = safe_auc(Y_back, s); l = safe_lrap(Y_back, s)
    print(f"   {name:<16} {h:>8.3f} {100*h/K:>6.1f}% {a:>7.3f} {l:>7.3f}")
print(f"   (slučajan baseline ≈ {7*7/39:.3f} hits/7)")
print()


# ============================================================
# 8) Prava predikcija SLEDEĆEG kola (6 kombinacija)
# ============================================================
s_next = {
    "WIDE_best":  predict_with(m_wide, w_wide_best,  X_next_s)[0],
    "WIDE_final": predict_with(m_wide, w_wide_final, X_next_s)[0],
    "DEEP_best":  predict_with(m_deep, w_deep_best,  X_next_s)[0],
    "DEEP_final": predict_with(m_deep, w_deep_final, X_next_s)[0],
}
s_next["ENSEMBLE_best"]  = (s_next["WIDE_best"]  + s_next["DEEP_best"])  / 2.0
s_next["ENSEMBLE_final"] = (s_next["WIDE_final"] + s_next["DEEP_final"]) / 2.0

preds = {name: topk_from_scores(s_next[name]) for name in order}

print("🎯 Predikcija SLEDEĆEG kola:")
for name in order:
    tag = "🏁" if name.startswith("ENSEMBLE") else "  "
    print(f"   {tag} {name:<16} -> {preds[name].tolist()}")
print()


# ============================================================
# 9) Validacija + snimanje
# ============================================================
def describe(pick):
    s = int(pick.sum())
    odd = int((pick % 2 == 1).sum())
    low = int((pick <= 19).sum())
    rng = int(pick.max() - pick.min())
    return f"suma={s}, neparnih={odd}/{K}, niskih(≤19)={low}/{K}, raspon={rng}"

for name in order:
    p = preds[name]
    assert len(set(p.tolist())) == K
    assert p.min() >= N_MIN and p.max() <= N_MAX
    assert list(p) == sorted(p.tolist())
    print(f"✅ {name:<16} validan ({describe(p)}).")

with open(OUT_TXT, "a", encoding="utf-8") as f:
    f.write(f"\n--- {stamp()} (seed={SEED}, N={N}) ---\n")
    for name in order:
        p = preds[name]
        f.write(f"{name:<16} -> {p.tolist()}  ({describe(p)})\n")
print(f"📝 Snimljeno u: {OUT_TXT}")

print()
print("🔁 loto_keras2D_v2 — stop ", stamp())
elapsed = time.time() - T0
print(f"⏱  Ukupno vreme: {str(timedelta(seconds=int(elapsed)))}  ({elapsed:.1f} s)")
print()




"""

To je upravo veliki problem koji vredi pomenuti:

Best epoche su 11 (WIDE) i 3 (DEEP). Znači posle 3-11 epoha modeli prestaju da uče i sledeće epohe samo gube vreme. 
Razlog: lutrija je dominantno šum, val_auc ~0.51 (slučajno bi bio 0.50).


WIDE best_val_auc = 0.5086
DEEP best_val_auc = 0.5167
razlika svega 0.008 = praktično ista — to je očekivano za loto
I rezultat back-testa to potvrđuje:

WIDE_best       1.28
DEEP_best       1.38
ENSEMBLE_best   1.27
baseline ≈      1.26
Sve je u oblasti slučajnog baseline-a — to je normalno za loto.


BEST se nalazi rano (epoha 3 i 11)
posle toga val_auc ne raste više

determinizam je deterministički → BEST je identičan i pri 200 i pri 1000 epoha ...
FINAL će biti drugačiji (zato što treniras dalje), ali BEST se neće mrdnuti.


Ako je BEST već našao plato u epohi 3 ili 11, FINAL posle 1000 epoha je samo preobučen model. To se vidi i u back-testu:

WIDE_best   1.28        WIDE_final   1.21   ← gore za 0.07
DEEP_best   1.38        DEEP_final   1.17   ← gore za 0.21
Što duže treniras, FINAL je sve dalje od BEST i sve gori. 
Znači povećanjem epoha samo pogoršavaš FINAL kombinaciju, jer je već gore od BEST.

Praktično pravilo:
BEST = pametna predikcija (uvek koristi)
FINAL = samo dokaz da je trening krenuo da preobučava (ako je gori od BEST)




BEST — bolje statistički.

Iz back-testa:

model	hits/7
WIDE_best
1.28
WIDE_final
1.21
DEEP_best
1.38 ← najbolji
DEEP_final
1.17
ENSEMBLE_best
1.27
ENSEMBLE_final
1.24
(slučajno)
1.26
DEEP_best je najbolji u back-testu. ENSEMBLE_best je tu negde, FINAL su svi ispod baseline-a (preobučeni).

Realno: razlika 1.38 vs 1.26 je vrlo mala — svega 0.12 broja prosečno. 
To je u okviru šuma. Loto je dominantno slučajan, model ne može da pobedi prirodu, jer je ona takođe dominantno slučajna.

Ako moraš jednu — odigraj DEEP_best.

Ako igras dve: DEEP_best + ENSEMBLE_best.

(Bez očekivanja čuda — ovo je samo statistički malo bolje od slučajnog.)






4620 epoha

🔁 loto_keras2D_v2 — start  24.05.2026_12.25.19

✅ CSV učitan: /data/loto7_4620_k41.csv
   broj izvlačenja: 4620, brojeva po kolu: 7

✅ Features: X_full.shape = (4620, 195), Y_full.shape = (4620, 39)

   train=4220  val=200  back=100

⚛️ Treniram WIDE (4620 epoha) ...
   ✅ WIDE: epohe=4620, best_epoch=11, best_val_auc=0.5086, final_val_auc=0.5000
⚛️ Treniram DEEP (4620 epoha) ...
   ✅ DEEP: epohe=4620, best_epoch=3, best_val_auc=0.5167, final_val_auc=0.5072

🧪 Back-test (poslednjih 100 izvlačenja):
   model              hits/7    hit%     AUC    LRAP
   WIDE_best           1.280   18.3%   0.527   0.247
   WIDE_final          1.310   18.7%   0.545   0.246
   DEEP_best           1.380   19.7%   0.508   0.263
   DEEP_final          1.220   17.4%   0.513   0.241
   ENSEMBLE_best       1.270   18.1%   0.528   0.247
   ENSEMBLE_final      1.220   17.4%   0.542   0.245
   (slučajan baseline ≈ 1.256 hits/7)

🎯 Predikcija SLEDEĆEG kola:
      WIDE_best        -> [6, 7, 23, 24, 26, 35, 36]
      WIDE_final       -> [6, 23, 25, 26, 32, 36, 38]
      DEEP_best        -> [7, 10, 13, 18, 23, 32, 33]
      DEEP_final       -> [7, 23, 26, 32, 33, 35, 36]
   🏁 ENSEMBLE_best    -> [6, 7, 8, 23, 24, 26, 35]
   🏁 ENSEMBLE_final   -> [7, 21, 23, 25, 26, 32, 36]

✅ WIDE_best        validan (suma=157, neparnih=3/7, niskih(≤19)=2/7, raspon=30).
✅ WIDE_final       validan (suma=186, neparnih=2/7, niskih(≤19)=1/7, raspon=32).
✅ DEEP_best        validan (suma=136, neparnih=4/7, niskih(≤19)=4/7, raspon=26).
✅ DEEP_final       validan (suma=192, neparnih=4/7, niskih(≤19)=1/7, raspon=29).
✅ ENSEMBLE_best    validan (suma=129, neparnih=3/7, niskih(≤19)=3/7, raspon=29).
✅ ENSEMBLE_final   validan (suma=170, neparnih=4/7, niskih(≤19)=1/7, raspon=29).
📝 Snimljeno u: /Users/4c/Desktop/GHQ/KvantniRegresor/loto_keras2D_v2_predikcija.txt

🔁 loto_keras2D_v2 — stop  24.05.2026_12.37.36
⏱  Ukupno vreme: 0:12:17  (737.1 s)

"""
