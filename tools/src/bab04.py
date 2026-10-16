from nbkit import md, code
from common import header, footer

NB = {}

# ------------------------------------------------------------------ 4.1 ALS
P = "notebooks/bab04_hybrid_cf/01_als_dengan_bias.ipynb"
NB[P] = [
header(P, "ALS dengan bias: pembaruan tertutup bergantian", 4, "Kode 4.2.1",
 ["Memahami bahwa ALS menyelesaikan tiap faktor sebagai regresi ridge (solusi tertutup).",
  "Menguraikan sumbangan rata-rata global, bias, dan faktor laten terhadap akurasi.",
  "Menyetel dimensi d dan regularisasi λ pada data validasi."],
 catatan="> Implementasi ALS di buku ditulis dengan perulangan Python agar mudah dibaca. Itu **lambat dan tidak production-ready** untuk data besar. Untuk skala nyata, gunakan pustaka ALS terdistribusi atau vektorisasi."),
md("## 1. Kode dari buku"),
code('''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def als_bias(R, n_users, n_items, d=50, lam=0.02, iters=10, seed=0):
    """R: daftar (u, i, r) dengan indeks bilangan bulat dari data latih."""
    rng = np.random.default_rng(seed)
    mu = np.mean([r for _, _, r in R])
    bu, bi = np.zeros(n_users), np.zeros(n_items)
    P = rng.normal(0, 0.1, (n_users, d))
    Q = rng.normal(0, 0.1, (n_items, d))

    by_user, by_item = {}, {}
    for u, i, r in R:
        by_user.setdefault(u, []).append((i, r))
        by_item.setdefault(i, []).append((u, r))

    for _ in range(iters):
        # Persamaan (4-3): bias pengguna
        for u, items in by_user.items():
            bu[u] = sum(r - mu - bi[i] - P[u] @ Q[i] for i, r in items) / (len(items) + lam)
        # Bias item, dengan peran pengguna dan item dipertukarkan
        for i, users in by_item.items():
            bi[i] = sum(r - mu - bu[u] - P[u] @ Q[i] for u, r in users) / (len(users) + lam)
        # Persamaan (4-4): faktor pengguna sebagai solusi ridge
        for u, items in by_user.items():
            Qu = Q[[i for i, _ in items]]
            y = np.array([r - mu - bu[u] - bi[i] for i, r in items])
            P[u] = np.linalg.solve(Qu.T @ Qu + lam * np.eye(d), Qu.T @ y)
        # Faktor item, simetris dengan faktor pengguna
        for i, users in by_item.items():
            Pi = P[[u for u, _ in users]]
            y = np.array([r - mu - bu[u] - bi[i] for u, r in users])
            Q[i] = np.linalg.solve(Pi.T @ Pi + lam * np.eye(d), Pi.T @ y)
    return mu, bu, bi, P, Q
'''),
md("""
## 2. Data simulasi

500 pengguna × 200 item dengan struktur laten berpangkat 4, bias pengguna dan item, derau, dan kepadatan 20%. Kita sembunyikan 20% rating untuk uji.
"""),
code('''
def simulasi_laten(m=500, n=200, rank=4, density=0.20, noise=0.5, seed=11):
    rng = np.random.default_rng(seed)
    P = rng.normal(0, 1, (m, rank)); Q = rng.normal(0, 1, (n, rank))
    bu = rng.normal(0, 0.5, m)[:, None]; bi = rng.normal(0, 0.5, n)[None, :]
    nilai = 3.4 + bu + bi + 0.6 * P @ Q.T + rng.normal(0, noise, (m, n))
    nilai = np.clip(np.round(nilai), 1, 5)
    u, i = np.nonzero(rng.random((m, n)) < density)
    return u, i, nilai[u, i], m, n

u, i, r, m, n = simulasi_laten()
perm = np.random.default_rng(0).permutation(len(r))
cut = int(0.2 * len(r))
te, tr = perm[:cut], perm[cut:]
R_latih = list(zip(u[tr], i[tr], r[tr]))
print("rating latih:", len(tr), "| uji:", len(te), "| kepadatan: %.1f%%" % (100 * len(r) / (m * n)))

def pred(model, uu, ii, faktor=True):
    mu, bu, bi, P, Q = model
    out = mu + bu[uu] + bi[ii]
    return out + (P[uu] * Q[ii]).sum(1) if faktor else out

def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))
'''),
md("""
## 3. Eksperimen A: sumbangan tiap komponen

Latih ALS (*d* = 8, λ = 5) lalu bandingkan RMSE uji: hanya rata-rata global, rata-rata global + bias, dan model penuh. λ di sini lebih besar daripada nilai awal buku karena data kecil dan berderau.
"""),
code('''
model = als_bias(R_latih, m, n, d=8, lam=5.0, iters=10)
mu = model[0]
rows = {
    "rata-rata global": rmse(r[te], np.full(len(te), mu)),
    "+ bias pengguna dan item": rmse(r[te], pred(model, u[te], i[te], faktor=False)),
    "+ faktor laten (model penuh)": rmse(r[te], pred(model, u[te], i[te])),
}
print(pd.Series(rows, name="RMSE uji").round(3))
assert rows["+ bias pengguna dan item"] < rows["rata-rata global"]
assert rows["+ faktor laten (model penuh)"] < rows["+ bias pengguna dan item"]
'''),
md("""
**Interpretasi (simulasi).** Pada simulasi ini faktor laten menyumbang penurunan galat terbesar, karena struktur laten sengaja dibuat kuat (bobot 0,6 pada perkalian faktor). Di data nyata biasanya sebaliknya: baseline bias sering sudah mengambil porsi besar, dan faktor laten memberi tambahan yang lebih kecil. Karena itu baseline bias tetap wajib dilaporkan, agar kontribusi faktor laten terukur dengan jujur.
"""),
md("""
## 4. Eksperimen B: jumlah iterasi

Latih ulang dengan `iters` berbeda dan lihat galat latih dan uji.
"""),
code('''
rows = []
for it in (1, 2, 3, 5, 10):
    mdl = als_bias(R_latih, m, n, d=8, lam=5.0, iters=it)
    rows.append({"iters": it, "RMSE_latih": rmse(r[tr], pred(mdl, u[tr], i[tr])),
                 "RMSE_uji": rmse(r[te], pred(mdl, u[te], i[te]))})
hasil_it = pd.DataFrame(rows).set_index("iters")
print(hasil_it.round(3))
assert hasil_it["RMSE_latih"].iloc[-1] < hasil_it["RMSE_latih"].iloc[0]
'''),
md("""
**Interpretasi.** ALS menurunkan galat latih dengan cepat: setiap langkah adalah solusi tertutup per pengguna atau item, bukan langkah kecil gradien. Pada data ini galat berhenti turun setelah sekitar lima iterasi. Perhatikan galat uji, bukan hanya galat latih, saat memutuskan berhenti.
"""),
md("""
## 5. Eksperimen C: dimensi d dan regularisasi λ

Grid kecil dengan `iters=6` agar cepat. Pilih kombinasi terbaik pada **validasi**, lalu laporkan pada uji. Di sini validasi diambil dari sebagian data latih.
"""),
code('''
perm_tr = np.random.default_rng(1).permutation(len(tr))
va_idx, fit_idx = tr[perm_tr[:len(tr) // 5]], tr[perm_tr[len(tr) // 5:]]
R_fit = list(zip(u[fit_idx], i[fit_idx], r[fit_idx]))

rows = []
for d in (2, 4, 8, 16):
    for lam in (0.5, 2.0, 5.0, 15.0):
        mdl = als_bias(R_fit, m, n, d=d, lam=lam, iters=6)
        rows.append({"d": d, "lam": lam, "RMSE_validasi": rmse(r[va_idx], pred(mdl, u[va_idx], i[va_idx]))})
grid = pd.DataFrame(rows).pivot(index="d", columns="lam", values="RMSE_validasi")
print(grid.round(3))
d_best, lam_best = [(d, l) for d in grid.index for l in grid.columns if grid.loc[d, l] == grid.values.min()][0]
final = als_bias(R_latih, m, n, d=d_best, lam=lam_best, iters=6)
print(f"terbaik pada validasi: d={d_best}, lam={lam_best} | RMSE uji = {rmse(r[te], pred(final, u[te], i[te])):.3f}")
'''),
md("""
**Interpretasi.** λ yang terlalu kecil membuat solusi ridge menghafal (terutama pada pengguna dengan sedikit rating), dan galat validasi naik pada *d* besar. λ yang terlalu besar membuat model *underfit*. Nilai menengah menstabilkan faktor. Dimensi *d* lebih besar hanya membantu bila λ ikut memadai. Menyetel *d* dan λ bersama pada validasi, lalu melaporkan satu kali pada uji, adalah protokol yang benar.
"""),
footer([
 "Mengapa solusi tertutup `np.linalg.solve(Qu.T @ Qu + lam * I, Qu.T @ y)` stabil untuk pengguna dengan satu rating, padahal `Qu` hanya satu baris?",
 "Dengan `lam=0`, apa yang terjadi pada pengguna yang jumlah ratingnya lebih sedikit daripada d?",
 "Bagaimana ALS diparalelkan, dan mengapa langkah per pengguna (dan per item) saling bebas?",
]),
]

# ------------------------------------------------------------------ 4.2 OOF fitur
P = "notebooks/bab04_hybrid_cf/02_fitur_statistik_out_of_fold.ipynb"
NB[P] = [
header(P, "Fitur statistik out-of-fold: mencegah kebocoran target", 4, "Kode 4.4.1",
 ["Memahami mengapa rata-rata rating per pengguna atau item harus dihitung out-of-fold untuk data latih.",
  "Mengukur kerusakan akibat kebocoran pada pengguna dengan sedikit rating.",
  "Melihat peran penghalusan (m) pada rata-rata yang bertumpu pada sedikit data."],
 catatan="> Model hilir di notebook ini adalah `HistGradientBoostingRegressor` dari scikit-learn (bukan LightGBM seperti pada eksperimen di buku), agar tidak memerlukan pustaka tambahan."),
md("## 1. Kode dari buku"),
code('''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.ensemble import HistGradientBoostingRegressor


def oof_mean(train, key, k=5, m=5, seed=0):
    """Rata-rata rating per `key` (user atau item) dengan penghalusan m, out-of-fold."""
    gmean = train['rating'].mean()
    out = np.empty(len(train))
    for fit_idx, enc_idx in KFold(n_splits=k, shuffle=True, random_state=seed).split(train):
        stats = train.iloc[fit_idx].groupby(key)['rating'].agg(['sum', 'count'])
        keys = train.iloc[enc_idx][key]
        s = keys.map(stats['sum']).fillna(0).to_numpy()
        c = keys.map(stats['count']).fillna(0).to_numpy()
        out[enc_idx] = (s + m * gmean) / (c + m)
    return out


def oof_mean_test(train, test, key, m=5):
    """Untuk data uji: statistik dari seluruh data latih, tanpa skema lipatan."""
    gmean = train['rating'].mean()
    stats = train.groupby(key)['rating'].agg(['sum', 'count'])
    s = test[key].map(stats['sum']).fillna(0).to_numpy()
    c = test[key].map(stats['count']).fillna(0).to_numpy()
    return (s + m * gmean) / (c + m)
'''),
md("""
## 2. Data simulasi dengan pengguna berjejak tipis

Jumlah rating per pengguna sangat timpang (lognormal): sepertiga lebih pengguna hanya punya satu rating. Itulah situasi saat kebocoran paling merusak.
"""),
code('''
rng = np.random.default_rng(0)
m_, n_, N = 3000, 300, 12000
pu = rng.lognormal(0, 1.8, m_); pu /= pu.sum()
pi = rng.lognormal(0, 1.0, n_); pi /= pi.sum()
u = rng.choice(m_, N, p=pu); i = rng.choice(n_, N, p=pi)
bu = rng.normal(0, .6, m_); bi = rng.normal(0, .5, n_)
r = np.clip(np.round(3.4 + bu[u] + bi[i] + rng.normal(0, 1.0, N)), 1, 5)
df = pd.DataFrame({"user": u, "item": i, "rating": r}).drop_duplicates(["user", "item"]).reset_index(drop=True)
perm = rng.permutation(len(df))
test = df.iloc[perm[:len(df) // 5]].reset_index(drop=True)
train = df.iloc[perm[len(df) // 5:]].reset_index(drop=True)
n_per_user = train.groupby("user").size()
print("latih:", len(train), "| uji:", len(test))
print("pengguna dengan tepat 1 rating latih: %.0f%% | median rating per pengguna: %d"
      % (100 * (n_per_user == 1).mean(), n_per_user.median()))
'''),
md("""
## 3. Eksperimen A: fitur "bocor" versus out-of-fold

Fitur *bocor* menghitung rata-rata per pengguna atau item dari **seluruh data latih termasuk baris itu sendiri**. Fitur *out-of-fold* memakai `oof_mean`. Data uji selalu memakai `oof_mean_test`.
"""),
code('''
def rata2_bocor(train, key, m=5):
    """Salah: rata-rata memuat rating baris itu sendiri."""
    g = train["rating"].mean()
    st = train.groupby(key)["rating"].agg(["sum", "count"])
    s = train[key].map(st["sum"]).to_numpy(); c = train[key].map(st["count"]).to_numpy()
    return (s + m * g) / (c + m)

def rmse(a, b): return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))

def evaluasi(m=5):
    Xt = np.column_stack([oof_mean_test(train, test, "user", m), oof_mean_test(train, test, "item", m)])
    X_bocor = np.column_stack([rata2_bocor(train, "user", m), rata2_bocor(train, "item", m)])
    X_oof = np.column_stack([oof_mean(train, "user", m=m), oof_mean(train, "item", m=m)])
    hasil = {}
    for nama, X in (("bocor", X_bocor), ("out-of-fold", X_oof)):
        g = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, random_state=0)
        g.fit(X, train["rating"])
        hasil[nama] = g.predict(Xt)
    return hasil

pred_uji = evaluasi(m=5)
y = test["rating"].to_numpy()
print("baseline rata-rata global:", round(rmse(y, train["rating"].mean()), 3))
for nama, p in pred_uji.items():
    print(f"RMSE uji, fitur {nama:12s}: {rmse(y, p):.3f}")
assert rmse(y, pred_uji["out-of-fold"]) < rmse(y, pred_uji["bocor"])
'''),
md("""
## 4. Eksperimen B: siapa yang paling dirugikan?

Pecah galat uji menurut banyaknya rating latih milik pengguna.
"""),
code('''
n_tr = test["user"].map(n_per_user).fillna(0).to_numpy()
kelompok = pd.cut(n_tr, [-1, 0, 1, 3, 10, 10**6], labels=["0 (baru)", "1", "2-3", "4-10", ">10"])
rows = []
for kel in kelompok.categories:
    mk = (kelompok == kel)
    if mk.sum() < 20: continue
    rows.append({"rating_latih_pengguna": kel, "n_uji": int(mk.sum()),
                 "bocor": rmse(y[mk], pred_uji["bocor"][mk]),
                 "out-of-fold": rmse(y[mk], pred_uji["out-of-fold"][mk])})
per_kel = pd.DataFrame(rows).set_index("rating_latih_pengguna")
print(per_kel.round(3))

fig, ax = plt.subplots(figsize=(5.4, 3.2))
per_kel[["bocor", "out-of-fold"]].plot.bar(ax=ax, rot=0)
ax.set(xlabel="rating latih milik pengguna", ylabel="RMSE uji")
fig.tight_layout()
'''),
md("""
**Interpretasi (simulasi).** Kerugian akibat kebocoran terkonsentrasi pada pengguna dengan sedikit rating. Untuk pengguna dengan satu rating latih, fitur bocor adalah fungsi persis dari rating itu sendiri, sehingga model belajar "percaya fitur ini hampir mutlak". Di data uji, fitur itu tidak lagi memuat rating target, dan model yang terlalu percaya menjadi salah. Pada pengguna dengan banyak rating, rata-rata hampir tidak terpengaruh satu baris, jadi selisihnya kecil.

Karena banyak sistem nyata didominasi pengguna berjejak tipis, kebocoran seperti ini dapat tampak sebagai "model bagus di validasi, buruk di produksi".
"""),
md("""
## 5. Eksperimen C: penghalusan m

Rata-rata dengan sedikit data berisik. Penghalusan menarik rata-rata ke rata-rata global sebesar bobot `m`. Bandingkan beberapa nilai *m* pada pipeline out-of-fold.
"""),
code('''
rows = []
for mm in (0.1, 1, 5, 20, 100):
    p = evaluasi(m=mm)["out-of-fold"]
    rows.append({"m": mm, "RMSE_uji (out-of-fold)": rmse(y, p)})
print(pd.DataFrame(rows).set_index("m").round(3))
'''),
md("""
**Interpretasi.** Pengaruh *m* jauh lebih kecil daripada pengaruh kebocoran. Pada simulasi ini, *m* kecil sampai sedang mirip hasilnya. *m* yang sangat besar membuang sinyal per pengguna. Prioritas pertama: pastikan fitur tidak membocorkan target. Penyetelan *m* menyusul.
"""),
footer([
 "Mengapa data uji memakai statistik dari *seluruh* data latih, bukan skema lipatan?",
 "Apa yang terjadi bila lipatan OOF dibuat per pengguna (bukan per baris)? Kapan itu perlu?",
 "Fitur lain apa yang rawan bocor dengan cara serupa (misalnya fitur hasil model lain yang dilatih pada data yang sama)?",
]),
]
