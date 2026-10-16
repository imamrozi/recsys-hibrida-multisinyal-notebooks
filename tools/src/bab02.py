from nbkit import md, code
from common import header, footer

NB = {}

# ------------------------------------------------------------------ 2.1 CBF
P = "notebooks/bab02_pendekatan_dasar/01_content_based_tfidf.ipynb"
NB[P] = [
header(P, "Content-based filtering: profil pengguna dan representasi item", 2, "Kode 2.1.1",
 ["Membangun profil pengguna sebagai rata-rata fitur item berbobot rating.",
  "Membandingkan representasi genre (satu-hot) dengan TF-IDF dari teks.",
  "Melihat langsung mengapa CBF cenderung menyempit (over-specialization)."]),
md("## 1. Kode dari buku\n\nToko buku mini dengan enam buku: tiga fantasi (i1–i3) dan tiga sains (i4–i6). Pengguna u2 menilai i1=4, i3=5, i5=1."),
code('''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Deskripsi singkat enam buku pada toko buku mini (Gambar 1.1)
deskripsi = [
    "penyihir naga kerajaan petualangan",      # i1 (fantasi)
    "naga ksatria kerajaan sihir",             # i2 (fantasi)
    "penyihir sihir misteri hutan",            # i3 (fantasi)
    "robot galaksi eksperimen teknologi",      # i4 (sains)
    "galaksi astronot teknologi riset",        # i5 (sains)
    "laboratorium kimia eksperimen reaksi"    # i6 (sains)
]
genre = np.array([[1, 0]] * 3 + [[0, 1]] * 3, dtype=float)  # [fantasi, sains]
X_teks = TfidfVectorizer().fit_transform(deskripsi).toarray()
rating_u2 = {0: 4, 2: 5, 4: 1}                    # indeks 0-based: i1, i3, i5


def profil(rating_u, X):
    """Rata-rata fitur item, diberi bobot rating (Persamaan 2-2)."""
    idx = list(rating_u)
    w = np.array([rating_u[i] for i in idx], dtype=float)
    return (w[:, None] * X[idx]).sum(axis=0) / w.sum()


def skor(rating_u, X):
    """Kemiripan kosinus profil dengan tiap item (Persamaan 2-3)."""
    return cosine_similarity(profil(rating_u, X)[None, :], X)[0]
'''),
code('''
for nama, X in (("genre", genre), ("tf-idf", X_teks)):
    s = skor(rating_u2, X)
    print(nama, {f"i{j + 1}": round(float(s[j]), 3) for j in (1, 3, 5)})
'''),
md("""
Yang belum dinilai u2 adalah i2, i4, i6. Dengan genre, i2 (fantasi) unggul. Dengan TF-IDF skornya lebih halus karena kata yang dipakai ikut menentukan.
"""),
md("## 2. Eksperimen A: peta kemiripan antaritem\n\nKemiripan kosinus antar-deskripsi menunjukkan apa yang sebenarnya dilihat CBF."),
code('''
S = cosine_similarity(X_teks)
fig, ax = plt.subplots(figsize=(4.2, 3.6))
im = ax.imshow(S, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(6), [f"i{j+1}" for j in range(6)])
ax.set_yticks(range(6), [f"i{j+1}" for j in range(6)])
for a in range(6):
    for b in range(6):
        ax.text(b, a, f"{S[a, b]:.2f}", ha="center", va="center", fontsize=7)
fig.colorbar(im, fraction=0.046)
fig.tight_layout()

dalam = S[:3, :3][np.triu_indices(3, 1)].mean()
antar = S[:3, 3:].mean()
print(f"rata-rata kemiripan dalam genre fantasi = {dalam:.3f} | fantasi-sains = {antar:.3f}")
assert dalam > antar
'''),
md("""
**Interpretasi.** Item sejenis hanya "dekat" jika berbagi kata. i2 dan i3 hampir tidak berbagi kata dengan i1 selain satu dua istilah, padahal sama-sama fantasi: **representasi menentukan batas kemampuan CBF**. Dengan satu-hot genre, ketiganya identik.
"""),
md("""
## 3. Eksperimen B: bobot rating mengubah profil

Profil adalah rata-rata berbobot. Ubah rating u2 pada i5 (sains) dari 1 menjadi 5 dan lihat skor tiga buku yang belum dinilai.
"""),
code('''
rows = []
for r5 in (1, 3, 5):
    s = skor({0: 4, 2: 5, 4: r5}, genre)
    rows.append({"rating_i5": r5, "i2_fantasi": s[1], "i4_sains": s[3], "i6_sains": s[5]})
tabel = pd.DataFrame(rows).set_index("rating_i5").round(3)
print(tabel)
assert tabel.loc[5, "i4_sains"] > tabel.loc[1, "i4_sains"]
'''),
md("""
**Interpretasi.** Pada pembobotan ini, rating rendah hanya *mengecilkan* kontribusi item sains, bukan menjadikannya negatif. Profil CBF dengan rata-rata berbobot tidak punya konsep "tidak suka". Ada varian yang memusatkan rating (dikurangi rata-rata pengguna) agar rating rendah menjadi bobot negatif.
"""),
md("## 4. Eksperimen C: over-specialization\n\nPengguna dengan riwayat fantasi saja: apa yang direkomendasikan 3 teratas?"),
code('''
rating_fantasi = {0: 5, 1: 4}                    # hanya menilai i1 dan i2
s = skor(rating_fantasi, X_teks)
belum = [j for j in range(6) if j not in rating_fantasi]
urut = sorted(belum, key=lambda j: -s[j])
print("urutan rekomendasi:", [f"i{j+1}" for j in urut])
print("skor:", {f"i{j+1}": round(float(s[j]), 3) for j in urut})
assert urut[0] == 2          # i3 (fantasi) di puncak; seluruh item sains tertinggal
'''),
md("""
**Interpretasi.** Profil yang dibentuk dari masa lalu hanya mencari hal yang mirip dengan masa lalu. Tidak ada mekanisme bawaan untuk menyodorkan sesuatu yang berbeda. Solusi di bab-bab lanjutan: re-ranking untuk keragaman, atau menggabungkan sinyal kolaboratif.
"""),
footer([
 "Mengapa i2 dan i3 (sama-sama fantasi) tidak selalu dianggap mirip oleh TF-IDF?",
 "Modifikasi apa pada fungsi `profil` yang membuat rating rendah menjadi sinyal negatif?",
 "Pada pengguna baru tanpa riwayat, apa yang dikembalikan `profil`? Mengapa itu masalah cold-start sisi pengguna?",
]),
]

# ------------------------------------------------------------------ 2.2a kNN
P = "notebooks/bab02_pendekatan_dasar/02_collaborative_filtering_knn.ipynb"
NB[P] = [
header(P, "Collaborative filtering berbasis tetangga (user-based k-NN)", 2, "Kode 2.2.1",
 ["Menghitung kemiripan pengguna dari matriks rating yang jarang.",
  "Memprediksi rating dari k tetangga terdekat.",
  "Memahami trade-off nilai k: akurasi, cakupan, dan cold-start."]),
md("## 1. Kode dari buku\n\nLima pengguna dan enam item (Gambar 1.1)."),
code('''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

nan = np.nan
R = np.array([[5, 4, nan, 1, nan, nan],      # u1
              [4, nan, 5, nan, 1, nan],      # u2
              [nan, 5, 4, nan, nan, 2],      # u3
              [nan, nan, nan, 5, 4, nan],    # u4
              [nan, nan, nan, 4, 5, nan]])   # u5 (Gambar 1.1)


def kemiripan_pengguna(R, u):
    """Kosinus antara baris u dan baris lain, sel kosong dianggap 0."""
    Z = np.nan_to_num(R)
    sim = Z @ Z[u] / (np.linalg.norm(Z, axis=1) * np.linalg.norm(Z[u]))
    sim[u] = 0.0                                   # abaikan diri sendiri
    return sim


def prediksi_knn(R, u, i, k=2):
    """Rata-rata rating item i dari k tetangga terdekat, berbobot kemiripan."""
    sim = kemiripan_pengguna(R, u)
    penilai = np.where(~np.isnan(R[:, i]) & (sim > 0))[0]
    tetangga = penilai[np.argsort(-sim[penilai])][:k]
    if len(tetangga) == 0:
        return np.nan                              # tidak ada bukti: cold-start
    return float(sim[tetangga] @ R[tetangga, i] / sim[tetangga].sum())
'''),
code('''
print("kemiripan u2 ke u1..u5:", np.round(kemiripan_pengguna(R, 1), 3))
for i in (1, 3, 5):
    print(f"r_hat(u2, i{i + 1}) =", round(prediksi_knn(R, 1, i), 2))
'''),
md("""
Perhatikan i4 dan i6: u2 hanya punya satu atau dua tetangga yang menilainya, sehingga prediksi bertumpu pada bukti yang tipis.
"""),
md("""
## 2. Eksperimen A: skala lebih besar dengan data simulasi

Kita bangkitkan 300 pengguna × 120 item dengan struktur selera laten (4 kelompok), kepadatan 25%, lalu sembunyikan 20% rating untuk uji. Prediksi k-NN dibandingkan dengan baseline rata-rata global dan rata-rata item.
"""),
code('''
def simulasi_rating(m=300, n=120, g=4, density=0.25, noise=0.6, seed=1):
    rng = np.random.default_rng(seed)
    grup_u = rng.integers(0, g, m)                 # kelompok selera pengguna
    grup_i = rng.integers(0, g, n)                 # kelompok item
    cocok = np.where(grup_u[:, None] == grup_i[None, :], 4.2, 2.4)
    bias_u = rng.normal(0, 0.3, m)[:, None]
    bias_i = rng.normal(0, 0.3, n)[None, :]
    nilai = cocok + bias_u + bias_i + rng.normal(0, noise, (m, n))
    nilai = np.clip(np.round(nilai), 1, 5)
    mask = rng.random((m, n)) < density
    return np.where(mask, nilai, np.nan)

R_big = simulasi_rating()
obs = np.argwhere(~np.isnan(R_big))
rng = np.random.default_rng(0)
uji = obs[rng.choice(len(obs), size=int(0.2 * len(obs)), replace=False)]
R_latih = R_big.copy()
R_latih[uji[:, 0], uji[:, 1]] = np.nan
y_uji = R_big[uji[:, 0], uji[:, 1]]
print("pengguna x item:", R_big.shape, "| rating teramati:", len(obs),
      "| uji:", len(uji), "| kepadatan latih: %.1f%%" % (100 * (~np.isnan(R_latih)).mean()))
'''),
code('''
def rmse(y, yh):
    ok = ~np.isnan(yh)
    return float(np.sqrt(np.mean((y[ok] - yh[ok]) ** 2))), float(ok.mean())

def knn_eval(R_latih, uji, y, ks, center=False):
    """RMSE uji k-NN. center=True: rata-rata tiap pengguna dikurangkan lebih dulu."""
    X = R_latih - np.nanmean(R_latih, axis=1, keepdims=True) if center else R_latih
    Z = np.nan_to_num(X)
    nrm = np.linalg.norm(Z, axis=1) + 1e-12
    S = (Z @ Z.T) / np.outer(nrm, nrm)
    np.fill_diagonal(S, 0.0)
    rows = []
    for k in ks:
        yh = np.full(len(uji), np.nan)
        for j, (u, i) in enumerate(uji):
            penilai = np.where(~np.isnan(R_latih[:, i]) & (S[u] > 0))[0]
            tet = penilai[np.argsort(-S[u, penilai])][:k]
            if len(tet):
                yh[j] = S[u, tet] @ R_latih[tet, i] / S[u, tet].sum()
        r, cakupan = rmse(y, yh)
        rows.append({"k": k, "RMSE": r, "cakupan_%": 100 * cakupan})
    return pd.DataFrame(rows).set_index("k")

mu = np.nanmean(R_latih)
rata_item = np.nanmean(R_latih, axis=0)
base_global = rmse(y_uji, np.full(len(uji), mu))[0]
base_item = rmse(y_uji, np.array([rata_item[i] if not np.isnan(rata_item[i]) else mu for _, i in uji]))[0]
ks = (1, 2, 5, 10, 20, 50)
hasil_mentah = knn_eval(R_latih, uji, y_uji, ks)
print("RMSE rata-rata global:", round(base_global, 3), "| rata-rata item:", round(base_item, 3))
print(hasil_mentah.round(3))
'''),
md("""
Dengan kosinus pada rating mentah (seperti di buku), keunggulan atas baseline rata-rata kecil. Semua rating bernilai positif, jadi kosinus banyak ditentukan oleh *seberapa banyak* item yang dinilai pengguna, bukan oleh *selera*-nya.
"""),
md("""
## 3. Eksperimen B: memusatkan rating per pengguna

Kurangi rata-rata tiap pengguna sebelum menghitung kosinus (ini setara dengan kosinus terpusat). Rating di atas kebiasaan pengguna jadi positif, di bawahnya negatif, sehingga selera ikut terbaca.
"""),
code('''
hasil_pusat = knn_eval(R_latih, uji, y_uji, ks, center=True)
print(hasil_pusat.round(3))

fig, ax = plt.subplots(figsize=(5.2, 3.2))
ax.plot(hasil_mentah.index, hasil_mentah["RMSE"], "o-", label="kosinus mentah")
ax.plot(hasil_pusat.index, hasil_pusat["RMSE"], "s-", label="kosinus terpusat")
ax.axhline(base_item, ls="--", c="gray", label="rata-rata item")
ax.set(xscale="log", xlabel="k (tetangga)", ylabel="RMSE uji")
ax.legend()
fig.tight_layout()

# klaim yang diuji: pemusatan memperbaiki hasil, dan k-NN terpusat mengalahkan baseline
assert hasil_pusat["RMSE"].min() < hasil_mentah["RMSE"].min() < base_global
assert hasil_pusat["RMSE"].min() < base_item
'''),
md("""
**Interpretasi (data simulasi).** *k* kecil berisik karena bertumpu pada satu atau dua tetangga. *k* sangat besar mencampur kelompok selera yang berbeda sehingga prediksi ditarik ke rata-rata. Titik terbaik ada di antaranya. Pemusatan rating memberi perbaikan yang jauh lebih besar daripada sekadar memilih *k* terbaik pada kosinus mentah. Pada matriks jarang, laporkan juga cakupan, bukan hanya RMSE.
"""),
footer([
 "Mengapa sel kosong dianggap 0 pada `kemiripan_pengguna`, dan masalah apa yang ditimbulkan? (Petunjuk: bandingkan dengan kosinus yang dihitung pada item yang sama-sama dinilai.)",
 "Apa yang terjadi pada cakupan jika kepadatan matriks diturunkan menjadi 3%?",
 "Mengapa `simulasi_rating` membuat k-NN tampak bagus? Ciri data nyata apa yang tidak tertangkap?",
]),
]

# ------------------------------------------------------------------ 2.2b MF
P = "notebooks/bab02_pendekatan_dasar/03_matrix_factorization_sgd.ipynb"
NB[P] = [
header(P, "Faktorisasi matriks berbias dengan SGD", 2, "Kode 2.2.2",
 ["Memahami prediksi r̂ = μ + b_u + b_i + p_u·q_i dan pembaruan SGD-nya.",
  "Melihat overfitting: galat latih turun, galat uji naik, saat model terlalu ekspresif.",
  "Melihat peran regularisasi."]),
md("## 1. Kode dari buku"),
code('''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

nan = np.nan
R = np.array([[5, 4, nan, 1, nan, nan],      # u1
              [4, nan, 5, nan, 1, nan],      # u2
              [nan, 5, 4, nan, nan, 2],      # u3
              [nan, nan, nan, 5, 4, nan],    # u4
              [nan, nan, nan, 4, 5, nan]])   # u5 (Gambar 1.1)


class BiasedMF:
    """r_hat = mu + b_u + b_i + p_u . q_i, dilatih dengan SGD per rating."""

    def __init__(self, d=2, lr=0.05, reg=0.3, epochs=300, seed=0):
        self.d, self.lr, self.reg = d, lr, reg
        self.epochs, self.seed = epochs, seed

    def fit(self, R):
        rng = np.random.default_rng(self.seed)
        m, n = R.shape
        obs = np.argwhere(~np.isnan(R))              # hanya sel yang teramati
        self.mu = np.nanmean(R)
        self.bu, self.bi = np.zeros(m), np.zeros(n)
        self.P = 0.1 * rng.standard_normal((m, self.d))
        self.Q = 0.1 * rng.standard_normal((n, self.d))
        for _ in range(self.epochs):
            for u, i in rng.permutation(obs):
                e = R[u, i] - self.predict(u, i)     # galat e_ui
                self.bu[u] += self.lr * (e - self.reg * self.bu[u])
                self.bi[i] += self.lr * (e - self.reg * self.bi[i])
                pu = self.P[u].copy()
                self.P[u] += self.lr * (e * self.Q[i] - self.reg * pu)
                self.Q[i] += self.lr * (e * pu - self.reg * self.Q[i])
        return self

    def predict(self, u, i):
        return self.mu + self.bu[u] + self.bi[i] + self.P[u] @ self.Q[i]
'''),
code('''
mf = BiasedMF().fit(R)
obs = np.argwhere(~np.isnan(R))
err = [R[u, i] - mf.predict(u, i) for u, i in obs]
print("RMSE latih:", round(float(np.sqrt(np.mean(np.square(err)))), 3))
for i in (1, 3, 5):
    print(f"r_hat(u2, i{i + 1}) =", round(float(mf.predict(1, i)), 2))
'''),
md("""
## 2. Eksperimen A: overfitting dan dimensi laten

Data simulasi 80 × 50 dengan struktur laten berpangkat 3 dan derau. Kita latih dengan dimensi *d* berbeda dan regularisasi kecil, lalu bandingkan RMSE latih dan uji.
"""),
code('''
def simulasi_laten(m=80, n=50, rank=3, density=0.25, noise=0.5, seed=3):
    rng = np.random.default_rng(seed)
    P = rng.normal(0, 1, (m, rank)); Q = rng.normal(0, 1, (n, rank))
    nilai = 3.5 + 0.7 * P @ Q.T + rng.normal(0, noise, (m, n))
    nilai = np.clip(np.round(nilai), 1, 5)
    mask = rng.random((m, n)) < density
    return np.where(mask, nilai, np.nan)

Rs = simulasi_laten()
obs = np.argwhere(~np.isnan(Rs))
rng = np.random.default_rng(0)
perm = rng.permutation(len(obs))
n_uji = int(0.2 * len(obs))
uji, latih = obs[perm[:n_uji]], obs[perm[n_uji:]]
R_latih = np.full_like(Rs, np.nan)
R_latih[latih[:, 0], latih[:, 1]] = Rs[latih[:, 0], latih[:, 1]]

def rmse_mf(model, pairs, Rfull):
    e = [Rfull[u, i] - model.predict(u, i) for u, i in pairs]
    return float(np.sqrt(np.mean(np.square(e))))

print("rating latih:", len(latih), "| uji:", len(uji))
rows = []
for d in (1, 2, 3, 8, 24):
    m_ = BiasedMF(d=d, lr=0.02, reg=0.01, epochs=60, seed=0).fit(R_latih)
    rows.append({"d": d, "RMSE_latih": rmse_mf(m_, latih, Rs), "RMSE_uji": rmse_mf(m_, uji, Rs)})
hasil_d = pd.DataFrame(rows).set_index("d")
print(hasil_d.round(3))
'''),
md("""
## 3. Eksperimen B: regularisasi

Tetapkan dimensi besar (*d* = 24) lalu variasikan `reg`. Regularisasi menekan faktor agar tidak menghafal derau.
"""),
code('''
rows = []
for reg in (0.0, 0.02, 0.1, 0.3, 1.0):
    m_ = BiasedMF(d=24, lr=0.02, reg=reg, epochs=60, seed=0).fit(R_latih)
    rows.append({"reg": reg, "RMSE_latih": rmse_mf(m_, latih, Rs), "RMSE_uji": rmse_mf(m_, uji, Rs)})
hasil_reg = pd.DataFrame(rows).set_index("reg")
print(hasil_reg.round(3))

fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.2))
hasil_d.plot(ax=a, marker="o", title="Dimensi laten d (reg=0,01)")
hasil_reg.plot(ax=b, marker="o", logx=False, title="Regularisasi (d=24)")
for x in (a, b): x.set_ylabel("RMSE")
fig.tight_layout()

# klaim yang diuji: galat latih lebih kecil daripada galat uji (celah generalisasi)
assert (hasil_d["RMSE_uji"] > hasil_d["RMSE_latih"]).all()
'''),
md("""
**Interpretasi (data simulasi).** Galat latih lebih kecil daripada galat uji di semua konfigurasi, dan celahnya melebar saat model makin ekspresif: pada *d* = 24 model nyaris hafal data latih (galat ≈ 0,07) tetapi galat uji tetap sekitar 1,0. Pada eksperimen ini galat uji tidak naik tajam bersama *d*, karena pelatihan hanya 60 epoch dengan inisialisasi kecil (efek *early stopping*). Regularisasi menunjukkan bentuk U: tanpa regularisasi model menghafal, sedangkan `reg` terlalu besar membuat model *underfit* (galat latih dan uji sama-sama naik). Titik terbaik ada di nilai menengah.
"""),
md("""
## 4. Eksperimen C: lamanya pelatihan

Latih model besar tanpa regularisasi dengan jumlah epoch yang makin banyak.
"""),
code('''
rows = []
for ep in (5, 20, 60, 150, 400):
    m_ = BiasedMF(d=24, lr=0.02, reg=0.0, epochs=ep, seed=0).fit(R_latih)
    rows.append({"epochs": ep, "RMSE_latih": rmse_mf(m_, latih, Rs), "RMSE_uji": rmse_mf(m_, uji, Rs)})
hasil_ep = pd.DataFrame(rows).set_index("epochs")
print(hasil_ep.round(3))
assert hasil_ep["RMSE_latih"].is_monotonic_decreasing
assert hasil_ep["RMSE_uji"].idxmin() not in (hasil_ep.index[0], hasil_ep.index[-1])
'''),
md("""
**Interpretasi.** Galat latih turun terus hingga hampir nol, sedangkan galat uji mencapai titik terendah di epoch menengah lalu sedikit naik. Itulah alasan praktiknya memakai data validasi untuk menghentikan pelatihan lebih awal (*early stopping*), bukan melatih sampai galat latih habis.
"""),
footer([
 "Mengapa pembaruan `bu`, `bi`, `P`, `Q` memakai salinan `pu` sebelum diperbarui?",
 "Apa yang berubah jika `epochs` dinaikkan dari 60 menjadi 300 pada konfigurasi `d=24, reg=0.01`?",
 "Mengapa SGD per rating sulit diparalelkan, dan apa alternatifnya? (Lihat ALS di Bab 4.)",
]),
]

# ------------------------------------------------------------------ 2.3 hibrida
P = "notebooks/bab02_pendekatan_dasar/04_hibrida_sederhana.ipynb"
NB[P] = [
header(P, "Hibrida sederhana: berbobot dan beralih", 2, "Kode 2.3.1",
 ["Memahami dua cara menggabungkan prediksi CF dan CBF.",
  "Melihat mengapa hibrida membantu pada item yang jarang dinilai (cold-start item).",
  "Menyetel κ dan τ dari data validasi, bukan dari intuisi."],
 catatan="> Eksperimen B memakai **model galat yang diasumsikan** (galat CF mengecil seiring jumlah rating). Ini ilustrasi mekanisme, bukan pengukuran."),
md("## 1. Kode dari buku"),
code('''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def hibrida_berbobot(pred_cf, pred_cbf, n_item, kappa=10.0):
    """Bobot CF naik seiring jumlah rating item (Persamaan 2-13)."""
    a = n_item / (n_item + kappa)
    return a * pred_cf + (1 - a) * pred_cbf


def hibrida_beralih(pred_cf, pred_cbf, n_item, tau=5):
    """CF bila item cukup dinilai, selain itu CBF (Persamaan 2-12)."""
    return np.where(n_item >= tau, pred_cf, pred_cbf)
'''),
code('''
n_item = np.array([0, 2, 10, 50])            # jumlah rating tiap item
pred_cf = np.full(4, 4.5)                    # CF dan CBF dibuat konstan
pred_cbf = np.full(4, 3.5)                   # agar efek bobot terlihat
print("n_item  :", n_item)
print("berbobot:", np.round(hibrida_berbobot(pred_cf, pred_cbf, n_item), 2))
print("beralih :", hibrida_beralih(pred_cf, pred_cbf, n_item))
'''),
md("""
## 2. Eksperimen A: bagaimana bobot bergerak

Kurva bobot CF `a = n / (n + κ)` untuk beberapa κ. Pada n = κ bobotnya tepat 0,5.
"""),
code('''
n = np.arange(0, 101)
fig, ax = plt.subplots(figsize=(5.2, 3.2))
for kappa in (2, 10, 30):
    ax.plot(n, n / (n + kappa), label=f"κ={kappa}")
ax.step(n, (n >= 5).astype(float), "k--", where="post", label="beralih (τ=5)")
ax.set(xlabel="jumlah rating item (n)", ylabel="bobot CF")
ax.legend()
fig.tight_layout()
'''),
md("""
## 3. Eksperimen B: kapan hibrida menang?

Asumsi simulasi: kebenaran adalah rating sebenarnya; galat CF mengecil seiring jumlah rating item (simpangan baku ∝ 1/√(n+1)), sedangkan galat CBF tetap (simpangan baku 0,8). Kita bandingkan RMSE per kelompok popularitas item.
"""),
code('''
rng = np.random.default_rng(0)
N = 20000
n_rating = np.minimum(rng.zipf(1.6, N), 200)               # banyak item hanya sedikit rating
y = rng.normal(3.5, 0.8, N)
cf = y + rng.normal(0, 1.8 / np.sqrt(n_rating + 1))        # galat CF mengecil dengan n
cbf = y + rng.normal(0, 0.8, N)                            # galat CBF konstan

def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))

bins = [(0, 1), (2, 4), (5, 14), (15, 200)]
rows = []
for lo, hi in bins:
    m = (n_rating >= lo) & (n_rating <= hi)
    rows.append({"n_rating": f"{lo}-{hi}", "item_%": 100 * m.mean(),
                 "CF": rmse(cf[m], y[m]), "CBF": rmse(cbf[m], y[m]),
                 "berbobot": rmse(hibrida_berbobot(cf, cbf, n_rating)[m], y[m]),
                 "beralih": rmse(hibrida_beralih(cf, cbf, n_rating)[m], y[m])})
hasil = pd.DataFrame(rows).set_index("n_rating")
print(hasil.round(3))

# klaim yang diuji: pada kelompok jarang dinilai, hibrida berbobot lebih baik daripada CF saja
assert hasil.iloc[0]["berbobot"] < hasil.iloc[0]["CF"]
'''),
md("""
**Interpretasi (dari asumsi simulasi).** CF sangat buruk pada item yang nyaris tak punya rating, CBF stabil tetapi tidak membaik seiring bertambahnya rating. Hibrida beralih mengikuti yang terbaik di ujung-ujung rentang, tetapi bisa lebih buruk daripada hibrida berbobot di sekitar ambang τ karena memakai CF yang masih berisik. Hibrida berbobot unggul atas keduanya di semua kelompok. Itu bergantung pada asumsi simulasi bahwa galat CF dan CBF *independen*. Pada data nyata galat keduanya sering berkorelasi, sehingga keuntungannya lebih kecil.

Ini hanya menunjukkan mekanisme. Bentuk kurva galat CF dan CBF pada data nyata harus diukur.
"""),
md("## 4. Eksperimen C: menyetel κ pada data validasi"),
code('''
idx = rng.permutation(N); val, tes = idx[: N // 2], idx[N // 2:]
kappas = [0.5, 1, 2, 5, 10, 20, 50]
rm_val = [rmse(hibrida_berbobot(cf[val], cbf[val], n_rating[val], k), y[val]) for k in kappas]
best = kappas[int(np.argmin(rm_val))]
print("RMSE validasi per κ:", {k: round(float(v), 3) for k, v in zip(kappas, rm_val)})
print("κ terbaik (validasi):", best)
print("RMSE uji pada κ itu :", round(rmse(hibrida_berbobot(cf[tes], cbf[tes], n_rating[tes], best), y[tes]), 3))
'''),
footer([
 "Mengapa hibrida beralih bisa lebih buruk daripada CBF di sekitar n = τ?",
 "Apa yang berubah pada κ terbaik jika galat CBF diperbesar menjadi 1,5?",
 "Selain jumlah rating item, sinyal apa lagi yang cocok menentukan bobot CF?",
]),
]
