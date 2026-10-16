from nbkit import md, code
from common import header, footer

NB = {}

# ------------------------------------------------------------------ 3.1 ABSA -> representasi
P = "notebooks/bab03_sentiment_aware/01_absa_dan_representasi_aspek.ipynb"
NB[P] = [
header(P, "Dari teks ulasan ke fitur aspek: ekstraksi ABSA dan empat representasi", 3, "Kode 3.2.1 dan 3.3.1",
 ["Mengubah ulasan menjadi tabel kalimat–aspek dengan skor sentimen dan konfidens.",
  "Membandingkan empat cara merepresentasikan output ABSA sebagai fitur.",
  "Melihat mengapa rata-rata aspek dapat menghapus kontras polaritas."],
 catatan="> Klasifikator bintang di sini adalah **tiruan berbasis kamus kecil** (`star_logits_tiruan`), bukan model BERT. Ia ada agar pipeline bisa dijalankan tanpa mengunduh model. Angka yang keluar mirip pola pada Tabel 3.5, tetapi **tidak identik**."),
md("## 1. Kode dari buku: ekstraksi (Kode 3.2.1)"),
code('''
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import softmax

LEXICON = {                      # contoh kamus restoran (bahasa Indonesia)
    "food": ["makan", "masakan", "rasa", "hidangan"],
    "service": ["pelayan", "layan", "staf"],
    "price": ["harga", "mahal", "murah"],
    "ambiance": ["suasana", "interior", "musik"],
}


def split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\\s+", text) if s.strip()]


def match_aspects(sentence, lexicon=LEXICON):
    """Langkah 1: sebuah kalimat cocok dengan aspek bila memuat salah satu stem."""
    s = sentence.lower()
    return [a for a, kws in lexicon.items() if any(kw in s for kw in kws)]


def extract(reviews, star_logits):
    """reviews: {review_id: teks}. star_logits(list[str]) -> array (n, 5).
    Mengembalikan satu baris per (kalimat, aspek) dengan skor di [0, 1]."""
    rows = [(rid, a, s) for rid, text in reviews.items()
            for s in split_sentences(text) for a in match_aspects(s)]
    df = pd.DataFrame(rows, columns=["review_id", "aspect", "sentence"])
    if df.empty:
        return df.assign(score=[], conf_softmax=[])
    z = np.asarray(star_logits(df["sentence"].tolist()))          # logit 5 kelas
    p = softmax(z, axis=1)
    y_hat = p @ np.arange(1, 6)               # bintang harapan, [1, 5]
    df["score"] = ((y_hat - 3) / 2 + 1) / 2   # skala bintang [1, 5] ke [0, 1]
    df["conf_softmax"] = p.max(axis=1)        # Pers. (3-2), bahan 3.4
    return df
'''),
md("""
Klasifikator tiruan: menilai kutub dari kata kunci, lalu menghasilkan logit 5 kelas berbentuk lonceng di sekitar bintang yang dituju. Kalimat tanpa petunjuk kutub mendapat sebaran lebar (tidak yakin), sehingga konfidens softmax-nya rendah.
"""),
code('''
POS = ["enak", "luar biasa", "lezat", "ramah", "cepat", "murah", "nyaman", "bagus", "segar"]
NEG = ["lambat", "cuek", "buruk", "mahal", "kotor", "bising", "hambar", "lama"]
NET = ["biasa"]

def star_logits_tiruan(sentences):
    """Logit lima kelas tiruan dari kamus kutub. Hanya untuk demonstrasi."""
    out = []
    for s in sentences:
        t = s.lower()
        pol = sum(w in t for w in POS) - sum(w in t for w in NEG)
        kuat = 1.25 if ("sangat" in t or "luar biasa" in t) else 1.0
        ada_petunjuk = pol != 0 or any(w in t for w in NET)
        target = 3 + 2 * float(np.clip(pol, -1, 1)) * 0.85 * kuat
        sigma = 0.7 if ada_petunjuk else 1.6          # tanpa petunjuk: tidak yakin
        out.append(-((np.arange(1, 6) - np.clip(target, 1, 5)) ** 2) / (2 * sigma ** 2))
    return np.array(out)

ulasan = {
    "A": "Makanannya luar biasa enak. Pelayanannya sangat lambat. Pelayannya cuek.",
    "B": "Makanannya biasa saja. Pelayanannya juga biasa.",
}
sent = extract(ulasan, star_logits_tiruan)
sent.round(3)
'''),
md("""
Satu baris per pasangan (kalimat, aspek). Ulasan A memiliki satu kalimat tentang makanan dan dua tentang layanan, dengan kutub yang saling berlawanan.
"""),
md("## 2. Kode dari buku: empat representasi (Kode 3.3.1)"),
code('''
FLOOR, EVID_CAP = 0.05, 3  # lantai konfidens, batas bukti (jumlah kalimat)


def per_aspect_table(sent, s0, aspects):
    """Rata-rata skor dan jumlah kalimat per (ulasan, aspek).
    Hasil: s (NaN bila aspek tidak disebut) dan n (0 bila tidak disebut)."""
    g = sent.groupby(["review_id", "aspect"])["score"].agg(s="mean", n="size")
    grid = dict(index=s0.index, columns=aspects)
    s = g["s"].unstack("aspect").reindex(**grid)
    n = g["n"].unstack("aspect").reindex(**grid).fillna(0)
    return s.to_numpy(), n.to_numpy()


def confidence(s, n):
    """Pers. (3-3): margin = |2s - 1|, bukti = min(n / 3, 1), lantai 0,05."""
    margin = np.abs(2 * s - 1)
    evidence = np.minimum(n / EVID_CAP, 1.0)
    return np.maximum((margin + evidence) / 2, FLOOR)


def mean_score(s_filled, hit, s0):
    """Strategi 1: rata-rata hanya atas aspek yang disebut, lainnya s0."""
    count = hit.sum(axis=1)
    total = np.where(hit, s_filled, 0.0).sum(axis=1)
    return np.where(count > 0, total / np.maximum(count, 1), s0)


def conf_weighted_mean(s_filled, c, hit, s0):
    """Strategi 2: rata-rata berbobot konfidens, hanya aspek yang disebut."""
    w = np.where(hit, c, 0.0)
    den = w.sum(axis=1)
    weighted = (w * s_filled).sum(axis=1) / np.maximum(den, 1e-12)
    return np.where(den > 0, weighted, s0)


def build_representations(sent, s0, aspects):
    """sent: DataFrame [review_id, aspect, score] (satu baris = satu kalimat).
    s0: skor fallback seluruh ulasan (indeks = review_id).
    Keempat representasi berbagi satu antarmuka, sehingga tahap fusi dapat
    ditahan tetap saat representasinya dibandingkan."""
    s, n = per_aspect_table(sent, s0, aspects)
    hit = ~np.isnan(s)
    s_filled = np.where(hit, s, s0.to_numpy()[:, None])  # Pers. (3-1): fallback
    c = confidence(s_filled, n)                          # n = 0: hanya margin

    def frame(values, prefix):
        cols = [f"{prefix}_{a}" for a in aspects]
        return pd.DataFrame(values, index=s0.index, columns=cols)

    s0v, idx = s0.to_numpy(), s0.index
    mean = mean_score(s_filled, hit, s0v)
    cmean = conf_weighted_mean(s_filled, c, hit, s0v)
    return {
        "mean": pd.DataFrame({"s_mean": mean}, index=idx),
        "conf_mean": pd.DataFrame({"s_confmean": cmean}, index=idx),
        "concat": frame(s_filled, "s"),
        "concat_conf": pd.concat([frame(s_filled, "s"), frame(c, "c")], axis=1),
    }
'''),
code('''
aspects = list(LEXICON)
# s0: skor seluruh ulasan (cadangan) = klasifikator pada teks utuh
z0 = softmax(star_logits_tiruan(list(ulasan.values())), axis=1)
s0 = pd.Series(((z0 @ np.arange(1, 6) - 3) / 2 + 1) / 2, index=list(ulasan), name="s0")
rep = build_representations(sent, s0, aspects)
for nama, df in rep.items():
    print("==", nama)
    print(df.round(2).to_string())
'''),
md("""
**Yang perlu dilihat.** Pada strategi *mean* dan *conf_mean*, ulasan A (makanan luar biasa, layanan buruk) dan B (semua biasa) berujung pada angka yang sama-sama dekat netral. *Concat* mempertahankan kontras: `s_food` A tinggi, `s_service` A rendah, sedangkan B datar. Kolom `c_price` dan `c_ambiance` kecil karena aspek itu tidak disebut.
"""),
code('''
A, B = rep["mean"].loc["A", "s_mean"], rep["mean"].loc["B", "s_mean"]
print(f"mean A = {A:.3f}, mean B = {B:.3f}, selisih = {abs(A - B):.3f}")
sA = rep["concat"].loc["A"]
assert sA["s_food"] - sA["s_service"] > 0.5            # kontras tetap ada pada concat
# aspek yang tidak disebut tidak pernah mendapat konfidens > 0,5 (hanya suku margin yang tersisa)
cc = rep["concat_conf"]
assert (cc.loc["A", ["c_price", "c_ambiance"]] <= 0.5).all()
'''),
md("""
## 3. Eksperimen A: kontras polaritas runtuh

Bangkitkan 3.000 ulasan simulasi dengan skor aspek `food` dan `service` yang bebas satu sama lain. Kita ukur: pada ulasan yang aspeknya berlawanan, ke mana rata-rata jatuh?
"""),
code('''
rng = np.random.default_rng(0)
n = 3000
food = rng.beta(2, 2, n)                    # skor aspek, di [0, 1]
service = rng.beta(2, 2, n)
rata2 = (food + service) / 2

kontras = np.abs(food - service) > 0.6      # aspek berlawanan tajam
print(f"ulasan dengan aspek berlawanan tajam: {100 * kontras.mean():.1f}%")
print(f"  rata-rata mereka berada di 0,35-0,65: {100 * ((rata2 > .35) & (rata2 < .65))[kontras].mean():.1f}%")

fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.3))
a.scatter(food[kontras], service[kontras], s=6, c="tab:red", label="berlawanan")
a.scatter(food[~kontras], service[~kontras], s=3, c="lightgray", label="lainnya")
a.set(xlabel="skor food", ylabel="skor service", title="Concat: kontras terlihat")
a.legend(markerscale=2)
b.hist(rata2[kontras], bins=25, color="tab:red", alpha=.8)
b.set(xlabel="rata-rata aspek", title="Mean: ulasan berlawanan menumpuk di tengah")
fig.tight_layout()
assert ((rata2 > .35) & (rata2 < .65))[kontras].mean() > 0.6
'''),
md("""
**Interpretasi.** Pada ulasan yang aspeknya berlawanan tajam, rata-rata mayoritas jatuh di sekitar netral. Satu angka itu tidak dapat dibedakan dari ulasan yang benar-benar netral. Informasi yang hilang di tahap agregasi tidak bisa dikembalikan oleh model fusi sesudahnya. Inilah alasan buku menahan fusi tetap dan membandingkan *representasi*-nya.
"""),
md("""
## 4. Eksperimen B: "rata-rata" yang sebenarnya satu kalimat

Mekanisme kegagalan kedua pada buku: pada banyak ulasan hanya sedikit aspek yang disebut. Dengan peluang menyebut tiap aspek *p*, berapa persen ulasan yang "rata-ratanya" tersusun dari tidak lebih dari satu aspek?
"""),
code('''
rng = np.random.default_rng(1)
rows = []
for p in (0.2, 0.35, 0.5, 0.7):
    disebut = rng.random((20000, 4)) < p
    k = disebut.sum(axis=1)
    rows.append({"p_sebut_aspek": p, "0 aspek_%": 100 * (k == 0).mean(),
                 "1 aspek_%": 100 * (k == 1).mean(), ">=2 aspek_%": 100 * (k >= 2).mean()})
tabel = pd.DataFrame(rows).set_index("p_sebut_aspek")
print(tabel.round(1))
assert tabel.loc[0.35, "0 aspek_%"] + tabel.loc[0.35, "1 aspek_%"] > 50
'''),
md("""
**Interpretasi (simulasi).** Jika setiap aspek hanya disebut dengan peluang sekitar sepertiga, lebih dari separuh ulasan menyebut satu aspek atau tidak sama sekali. Untuk ulasan itu, "rata-rata multi-aspek" adalah skor satu kalimat, atau cadangan `s0`. Ia bukan ringkasan beberapa aspek. Menurut buku, inilah yang menyuntikkan derau pada dua varian agregasi.
"""),
footer([
 "Mengapa konfidens aspek yang tidak disebut tidak pernah melebihi 0,5 pada Persamaan (3-3)? Apa gunanya bagi model hilir?",
 "Kalimat tanpa kata kutub mendapat konfidens softmax rendah di klasifikator tiruan. Apa yang terjadi bila klasifikator sungguhan *overconfident* pada kalimat seperti itu?",
 "Ubah `LEXICON` ke domain hotel. Apa yang berubah pada tabel kalimat–aspek, dan apa yang tidak?",
]),
]

# ------------------------------------------------------------------ pembangkit data bersama (disalin ke notebook 02 dan 03)
SIM = '''
def simulasi_ulasan(m=600, n=200, K=4, density=0.15, noise=0.3, conf_informatif=True, seed=0):
    """Data ulasan tiruan: rating bergantung pada kualitas aspek item dan bobot aspek pengguna.
    Asumsi yang dibuat (bukan fakta empiris):
      - aspek berbeda punya kepentingan umum berbeda (w_global);
      - sentimen aspek yang DIEKSTRAK = sentimen sebenarnya + derau;
      - konfidens `c` bersifat informatif bila ia menurun saat derau naik (conf_informatif=True),
        atau acak murni (False)."""
    rng = np.random.default_rng(seed)
    mask = rng.random((m, n)) < density
    u, i = np.nonzero(mask)
    N = len(u)
    w_global = np.array([1.6, 0.9, 0.5, 0.2])[:K]           # pentingnya aspek secara umum
    a = rng.dirichlet(w_global * 6, m) * 3.6                 # bobot aspek tiap pengguna
    q = rng.normal(0, 1, (n, K))                             # kualitas aspek tiap item
    t = 1 / (1 + np.exp(-(q[i] + rng.normal(0, .5, (N, K)))))  # sentimen sebenarnya di (0, 1)
    bu, bi = rng.normal(0, .3, m), rng.normal(0, .3, n)
    P, Q = rng.normal(0, 1, (m, 3)), rng.normal(0, 1, (n, 3))
    r = (3.0 + bu[u] + bi[i] + 0.25 * (P[u] * Q[i]).sum(1)
         + (a[u] * (2 * t - 1)).sum(1) + rng.normal(0, .4, N))
    r = np.clip(np.round(r), 1, 5)
    sigma = noise * rng.uniform(0.3, 1.7, (N, K))            # derau ekstraksi bervariasi per sampel
    s = np.clip(t + rng.normal(0, sigma), 0, 1)              # sentimen terekstrak
    if conf_informatif:
        c = np.clip(0.97 - 1.4 * sigma + rng.normal(0, .05, (N, K)), 0.2, 0.97)
    else:
        c = rng.uniform(0.2, 0.97, (N, K))
    return dict(u=u, i=i, r=r, s=s, c=c, a=a, m=m, n=n, K=K)

def rmse(y, yh):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yh)) ** 2)))

def bagi(N, seed, frac=0.2):
    perm = np.random.default_rng(seed).permutation(N)
    return perm[int(frac * N):], perm[:int(frac * N)]        # indeks latih, uji
'''

# ------------------------------------------------------------------ 3.2 konfidens + MF aspek
P = "notebooks/bab03_sentiment_aware/02_konfidens_gerbang_mf_aspek.ipynb"
NB[P] = [
header(P, "Konfidens, gerbang sigmoid, dan MF dengan suku aspek", 3, "Kode 3.4.1 dan 3.5.1",
 ["Menghitung konfidens softmax dan melihat batasnya pada klasifikator lima kelas.",
  "Memahami gerbang sigmoid g(c) dan peran τ serta k.",
  "Membandingkan MF, MF+ABSA, dan MF+CA-ABSA pada kualitas ekstraksi yang berbeda."],
 catatan="> Eksperimen B dan C memakai **simulasi dengan asumsi eksplisit** (lihat docstring `simulasi_ulasan`). Hasilnya menunjukkan *kapan* gerbang membantu atau tidak, bukan seberapa besar manfaatnya pada dataset nyata."),
md("## 1. Kode dari buku: konfidens dan gerbang (Kode 3.4.1)"),
code('''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import softmax


def conf_softmax(z):
    """Pers. (3-2): c = max softmax(z). Untuk C kelas, c berada di [1/C, 1]."""
    return softmax(z, axis=1).max(axis=1)


def gate(c, tau=0.5, k=10.0):
    """Pers. (3-4): g(c) = 1 / (1 + exp(-k (c - tau)))."""
    return 1.0 / (1.0 + np.exp(-k * (c - tau)))
'''),
md("""
### Eksperimen A: apa arti konfidens softmax?

Untuk 5 kelas, konfidens minimum adalah 1/5 = 0,2 (tebakan seragam). Kita variasikan "ketajaman" logit dan lihat konfidensnya, lalu gambar gerbang untuk beberapa pasangan (τ, k).
"""),
code('''
rng = np.random.default_rng(0)
z0 = rng.normal(0, 1, (5000, 5))
rows = []
for skala in (0.1, 0.5, 1, 2, 4):
    c = conf_softmax(skala * z0)
    rows.append({"skala_logit": skala, "c_rata2": c.mean(), "c_min": c.min(), "c_maks": c.max(),
                 "g(c)_rata2": gate(c).mean()})
print(pd.DataFrame(rows).set_index("skala_logit").round(3))

cc = np.linspace(0.2, 1, 200)
fig, ax = plt.subplots(figsize=(5.2, 3.2))
for tau, k in ((0.5, 10), (0.5, 30), (0.7, 10), (0.3, 10)):
    ax.plot(cc, gate(cc, tau, k), label=f"τ={tau}, k={k}")
ax.set(xlabel="konfidens c", ylabel="g(c)")
ax.legend()
fig.tight_layout()
assert conf_softmax(0.1 * z0).min() >= 0.2 - 1e-9
'''),
md("""
**Interpretasi.** Konfidens softmax dibatasi dari bawah oleh 1/C. Logit yang tajam menghasilkan konfidens tinggi *tanpa jaminan* bahwa klasifikator benar, karena softmax tidak terkalibrasi. τ menggeser titik tengah gerbang dan *k* mengatur ketajamannya: *k* besar membuat gerbang hampir seperti sakelar.
"""),
md("## 2. Kode dari buku: MF dengan suku aspek (Kode 3.5.1)"),
code('''
class AspectMF:
    """r_hat = mu + b_u + b_i + p_u.q_i + sum_k a_u^k * s_ui^k  [Pers. 3-6]"""

    def __init__(self, n_users, n_items, n_aspects, d=8, lr=0.02, reg=0.02,
                 use_aspect=True, seed=0):
        rng = np.random.default_rng(seed)
        self.P = rng.normal(0, 0.1, (n_users, d))
        self.Q = rng.normal(0, 0.1, (n_items, d))
        self.bu, self.bi = np.zeros(n_users), np.zeros(n_items)
        self.A = np.zeros((n_users, n_aspects))
        self.mu, self.lr, self.reg = 0.0, lr, reg
        self.use_aspect, self.rng = use_aspect, rng

    def predict(self, u, i, s):
        out = self.mu + self.bu[u] + self.bi[i] + (self.P[u] * self.Q[i]).sum(1)
        return out + (self.A[u] * s).sum(1) if self.use_aspect else out

    def fit(self, u, i, r, s, epochs=40, batch=256):
        self.mu = r.mean()
        lr, lam = self.lr, self.reg
        for _ in range(epochs):
            for idx in np.array_split(self.rng.permutation(len(r)),
                                      max(1, len(r) // batch)):
                ub, ib, sb = u[idx], i[idx], s[idx]
                e = (r[idx] - self.predict(ub, ib, sb))[:, None]  # galat
                pu, qi = self.P[ub].copy(), self.Q[ib].copy()
                np.add.at(self.bu, ub, lr * (e[:, 0] - lam * self.bu[ub]))
                np.add.at(self.bi, ib, lr * (e[:, 0] - lam * self.bi[ib]))
                np.add.at(self.P, ub, lr * (e * qi - lam * pu))
                np.add.at(self.Q, ib, lr * (e * pu - lam * qi))
                if self.use_aspect:  # d r_hat / d a_u^k = s_ui^k
                    np.add.at(self.A, ub, lr * (e * sb - lam * self.A[ub]))
        return self
'''),
md("""
Tiga varian dibedakan oleh masukan suku aspek: **MF** (tanpa suku aspek), **MF+ABSA** (`s` apa adanya), dan **MF+CA-ABSA** (`g(c) · s`, sentimen yang tidak yakin diredam). Pada pembaruan SGD, `A` diperbarui sebanding dengan `e · s`, jadi `g(c)` ikut mengecilkan pembelajaran dari sampel yang tidak yakin. Itu turunan matematis, bukan klaim empiris.
"""),
md("## 3. Pembangkit data simulasi"),
code(SIM),
code('''
D = simulasi_ulasan(noise=0.2)
print("interaksi:", len(D["r"]), "| pengguna:", D["m"], "| item:", D["n"], "| aspek:", D["K"])
print("rata-rata konfidens: %.2f | rata-rata g(c): %.2f | porsi g(c) < 0,5: %.1f%%"
      % (D["c"].mean(), gate(D["c"]).mean(), 100 * (gate(D["c"]) < .5).mean()))
'''),
md("""
### Eksperimen B: kapan sentimen aspek membantu, dan kapan gerbang membantu?

Variasikan derau ekstraksi (`noise`). Untuk tiap tingkat derau, tiga seed, kita bandingkan RMSE uji empat model: MF, MF+ABSA, MF+CA-ABSA dengan konfidens **informatif**, dan MF+CA-ABSA dengan konfidens **acak** (tidak berkaitan dengan derau).
"""),
code('''
def evaluasi(noise, seeds=(0, 1, 2)):
    hasil = {"MF": [], "MF+ABSA": [], "MF+CA-ABSA (c informatif)": [], "MF+CA-ABSA (c acak)": []}
    for sd in seeds:
        Di = simulasi_ulasan(noise=noise, conf_informatif=True, seed=sd)
        Dr = simulasi_ulasan(noise=noise, conf_informatif=False, seed=sd)   # data sama, hanya c berbeda
        assert np.allclose(Di["s"], Dr["s"])
        tr, te = bagi(len(Di["r"]), sd)
        u, i, r = Di["u"], Di["i"], Di["r"]
        for nama, use, S in (("MF", False, Di["s"]),
                             ("MF+ABSA", True, Di["s"]),
                             ("MF+CA-ABSA (c informatif)", True, gate(Di["c"]) * Di["s"]),
                             ("MF+CA-ABSA (c acak)", True, gate(Dr["c"]) * Dr["s"])):
            m = AspectMF(Di["m"], Di["n"], Di["K"], use_aspect=use, seed=sd).fit(u[tr], i[tr], r[tr], S[tr])
            hasil[nama].append(rmse(r[te], m.predict(u[te], i[te], S[te])))
    return {k: float(np.mean(v)) for k, v in hasil.items()}

tabel = pd.DataFrame({noise: evaluasi(noise) for noise in (0.05, 0.2, 0.4, 0.7)}).T
tabel.index.name = "noise"
print(tabel.round(3))
'''),
code('''
fig, ax = plt.subplots(figsize=(6, 3.4))
for kol, gaya in zip(tabel.columns, ("k--", "o-", "s-", "^:")):
    ax.plot(tabel.index, tabel[kol], gaya, label=kol)
ax.set(xlabel="derau ekstraksi sentimen", ylabel="RMSE uji (rata-rata 3 seed)")
ax.legend(fontsize=7)
fig.tight_layout()

# klaim yang diuji (kualitatif): sentimen yang bersih membantu; manfaat itu menyusut seiring derau
gain = tabel["MF"] - tabel["MF+ABSA"]
assert gain.loc[0.05] > 0.03 and gain.loc[0.05] > gain.loc[0.7]
# konfidens acak tidak lebih baik daripada konfidens informatif secara rata-rata
assert tabel["MF+CA-ABSA (c informatif)"].mean() <= tabel["MF+CA-ABSA (c acak)"].mean()
'''),
md("""
**Interpretasi (simulasi dengan asumsi di atas).**

1. Sentimen aspek yang bersih (derau 0,05) menurunkan RMSE jauh di bawah MF. Manfaatnya menyusut saat derau naik. Pada derau tinggi MF+ABSA bahkan bisa lebih buruk daripada MF, karena suku aspek memasukkan derau ke prediksi.
2. Gerbang dengan konfidens **informatif** meredam derau itu. Pada derau tertinggi ia menyelamatkan model dari kerugian MF+ABSA. Pada derau rendah ia sedikit mengorbankan sinyal, karena `g(c) < 1` ikut mengecilkan sentimen yang sebenarnya benar.
3. Gerbang dengan konfidens **acak** tidak bisa membedakan sentimen yang andal dari yang tidak, sehingga hampir selalu lebih buruk daripada konfidens informatif. **Gerbang hanya berguna bila konfidens benar-benar membawa informasi tentang keandalan.** Pada klasifikator sungguhan, itu bergantung pada kalibrasi softmax, yang tidak dijamin.

Selisih antara MF+ABSA dan MF+CA-ABSA kecil (di bawah satu persen pada derau rendah dan menengah). Karena itu bab 3.8 memakai uji berpasangan dan banyak seed. Lihat notebook berikutnya.
"""),
md("""
### Eksperimen C: apakah bobot aspek pengguna terpelajar benar?

Karena datanya simulasi, kita tahu bobot aspek pengguna yang sebenarnya (`a`). Ukur korelasi antara `A` hasil pelatihan dan `a` yang sebenarnya.
"""),
code('''
rows = []
for noise in (0.05, 0.2, 0.4, 0.7):
    kor = {"ABSA": [], "CA-ABSA": []}
    for sd in (0, 1, 2):
        Dn = simulasi_ulasan(noise=noise, seed=sd)
        tr, _ = bagi(len(Dn["r"]), sd)
        for nama, S in (("ABSA", Dn["s"]), ("CA-ABSA", gate(Dn["c"]) * Dn["s"])):
            m = AspectMF(Dn["m"], Dn["n"], Dn["K"], seed=sd).fit(Dn["u"][tr], Dn["i"][tr], Dn["r"][tr], S[tr])
            kor[nama].append(np.corrcoef(m.A.ravel(), Dn["a"].ravel())[0, 1])
    rows.append({"noise": noise, **{k: float(np.mean(v)) for k, v in kor.items()}})
kor_tab = pd.DataFrame(rows).set_index("noise")
print(kor_tab.round(3))
assert kor_tab["ABSA"].is_monotonic_decreasing
'''),
md("""
**Interpretasi.** Pada simulasi ini bobot aspek pengguna makin sulit dipulihkan saat derau naik. Gerbang **tidak** memperbaiki pemulihan bobot itu. Ia bahkan sedikit menurunkannya, karena `g(c) · s` mengecilkan sinyal yang dipakai untuk menaksir `a`. Jadi manfaat gerbang pada RMSE (Eksperimen B) tidak otomatis berarti pemahaman preferensi pengguna lebih baik. Itu hasil simulasi ini, bukan hukum umum. Kesimpulan seperti itu perlu diuji pada data nyata dengan label preferensi aspek yang valid, jika ada.
"""),
md("""
### Pemeriksaan gradien: apakah pembaruan SGD sesuai Persamaan (3-8)?

Buku menyatakan bahwa pembaruan ditulis eksplisit agar dapat diperiksa dengan beda hingga. Di sini satu langkah SGD pada satu rating (kelas `AspectMF` apa adanya) dibandingkan dengan gradien numerik dari fungsi rugi `½ e² + ½ λ (‖θ‖²)` dengan beda hingga terpusat. Pada satu sampel, `fit` menyetel `mu = r`, jadi `mu` ikut menjadi konstanta rugi.
"""),
code('''
def rugi(m, u, i, r, s):
    e = r - m.predict(np.array([u]), np.array([i]), s)[0]
    reg = (m.bu[u] ** 2 + m.bi[i] ** 2 + (m.P[u] ** 2).sum() + (m.Q[i] ** 2).sum() + (m.A[u] ** 2).sum())
    return 0.5 * e ** 2 + 0.5 * m.reg * reg

def cek_gradien(seed=0, h=1e-6):
    rng = np.random.default_rng(seed)
    nU, nI, nA, u, i = 3, 4, 3, 1, 2
    r = np.array([3.7]); s = rng.normal(0, 1, (1, nA))
    m = AspectMF(nU, nI, nA, d=4, lr=1.0, reg=0.05, seed=seed)
    # keadaan awal acak agar semua suku non-trivial
    m.bu[:] = rng.normal(0, .3, nU); m.bi[:] = rng.normal(0, .3, nI); m.A[:] = rng.normal(0, .3, (nU, nA))
    m.mu = float(r[0])
    sebelum = {k: getattr(m, k).copy() for k in ("bu", "bi", "P", "Q", "A")}
    # gradien numerik di titik awal (mu = r, seperti yang akan diset fit)
    def fd(nama, idx):
        arr = getattr(m, nama)
        a0 = arr[idx]; arr[idx] = a0 + h; lp = rugi(m, u, i, r[0], s); arr[idx] = a0 - h; lm = rugi(m, u, i, r[0], s); arr[idx] = a0
        return (lp - lm) / (2 * h)
    galat = 0.0
    num = {}
    for nama, baris in (("bu", u), ("bi", i), ("P", u), ("Q", i), ("A", u)):
        arr = getattr(m, nama)
        idxs = [baris] if arr.ndim == 1 else [(baris, j) for j in range(arr.shape[1])]
        num[nama] = np.array([fd(nama, ix) for ix in idxs])
    # satu langkah SGD sungguhan (lr = 1 -> delta = -gradien)
    m.fit(np.array([u]), np.array([i]), r, s, epochs=1, batch=1)
    for nama, baris in (("bu", u), ("bi", i), ("P", u), ("Q", i), ("A", u)):
        delta = getattr(m, nama)[baris] - sebelum[nama][baris]
        galat = max(galat, float(np.max(np.abs(np.atleast_1d(delta) + num[nama]))))
    return galat

g_maks = max(cek_gradien(sd) for sd in range(5))
print(f"galat maksimum |pembaruan SGD + gradien numerik|: {g_maks:.2e}")
assert g_maks < 1e-6
'''),
md("""
**Interpretasi.** Pembaruan di `fit` identik dengan gradien rugi (sampai ketelitian beda hingga), termasuk suku aspek `dr̂/da = s`. Ini memeriksa *implementasi* terhadap persamaan, bukan kualitas model. Perhatikan bahwa pemeriksaan ini memakai satu sampel dan `lr = 1`; pada pelatihan sebenarnya, `np.add.at` menjumlahkan kontribusi sampel dalam satu batch, yang secara matematis setara dengan menjumlahkan gradien mereka.
"""),
footer([
 "Mengapa gerbang memakai `g(c) · s` dan bukan menarik `s` ke 0,5 (netral) saat konfidens rendah? Apa konsekuensinya bagi tanda sumbangan aspek?",
 "Ubah `conf_informatif=False` dan ulangi Eksperimen B pada derau 0,7. Apa yang berubah?",
 "Bagaimana Anda akan memeriksa bahwa konfidens softmax klasifikator BERT sungguhan informatif? (Petunjuk: kurva kalibrasi.)",
]),
]

# ------------------------------------------------------------------ 3.3 stacking + uji
P = "notebooks/bab03_sentiment_aware/03_stacking_dan_uji_berpasangan.ipynb"
NB[P] = [
header(P, "Stacking out-of-fold dan uji berpasangan", 3, "Kode 3.6.1 dan 3.8.1",
 ["Memahami mengapa fitur stacking harus berasal dari prediksi out-of-fold.",
  "Membandingkan representasi sentimen (mean, concat, concat+konfidens) pada fusi pohon keputusan.",
  "Membaca uji berpasangan, bootstrap berpasangan, dan p gabungan Fisher dengan benar."],
 catatan="> Seperti notebook sebelumnya, data adalah simulasi dengan asumsi eksplisit. Selisih antar representasi di sini **kecil**, dan justru kasus seperti itu yang membutuhkan uji statistik."),
md("## 1. Kode dari buku: stacking (Kode 3.6.1)"),
code('''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor


def oof_predictions(fit_predict, train_idx, n_splits=5, seed=0):
    """Prediksi out-of-fold: setiap baris train diprediksi oleh model yang
    TIDAK pernah melihat baris itu. fit_predict(tr, te) melatih model baru
    pada indeks `tr` dan mengembalikan prediksi untuk indeks `te`."""
    out = np.zeros(len(train_idx))
    for tr, te in KFold(n_splits, shuffle=True, random_state=seed).split(train_idx):
        out[te] = fit_predict(train_idx[tr], train_idx[te])
    return out


def fit_fusion(cf_oof, sent_train, y_train, max_depth=10, seed=0):
    """Pohon keputusan di atas [prediksi CF out-of-fold | blok sentimen].
    Blok sentimen: 1 kolom (global/mean), K (concat) atau 2K (concat+conf)."""
    X = np.column_stack([cf_oof, sent_train])
    tree = DecisionTreeRegressor(max_depth=max_depth, random_state=seed).fit(X, y_train)
    return lambda cf_te, sent_te: tree.predict(np.column_stack([cf_te, sent_te]))
'''),
md("## 2. Kode dari buku: uji berpasangan (Kode 3.8.1)"),
code('''
def rmse_gain_pct(se_base, se_new):
    """Penurunan RMSE relatif (%) ; se_* = galat kuadrat per sampel uji."""
    return 100 * (1 - np.sqrt(se_new.mean()) / np.sqrt(se_base.mean()))


def paired_bootstrap_ci(se_base, se_new, n_boot=2000, alpha=0.05, seed=0):
    """CI persentil untuk penurunan RMSE; sampel uji diacak ULANG BERPASANGAN."""
    rng = np.random.default_rng(seed)
    n = len(se_base)
    gains = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        gains[b] = rmse_gain_pct(se_base[idx], se_new[idx])
    return np.percentile(gains, [100 * alpha / 2, 100 * (1 - alpha / 2)])


def per_seed_summary(se_base_by_seed, se_new_by_seed):
    """Wilcoxon per seed (pada galat kuadrat per sampel) + p gabungan Fisher.
    Semua seed memakai set uji yang sama, jadi p gabungan hanyalah ringkasan."""
    p = [stats.wilcoxon(b, n).pvalue for b, n in zip(se_base_by_seed, se_new_by_seed)]
    fisher = stats.combine_pvalues(p, method="fisher").pvalue
    return {"p_per_seed": p, "n_sig": int(sum(x < 0.05 for x in p)), "p_fisher": fisher}
'''),
md("""
## 3. Data simulasi dan MF dasar

Kelas `AspectMF` sama seperti pada notebook 02. Di sini ia dipakai tanpa suku aspek (`use_aspect=False`) sebagai model CF dasar.
"""),
code('''
class AspectMF:
    """r_hat = mu + b_u + b_i + p_u.q_i + sum_k a_u^k * s_ui^k  [Pers. 3-6]"""

    def __init__(self, n_users, n_items, n_aspects, d=8, lr=0.02, reg=0.02,
                 use_aspect=True, seed=0):
        rng = np.random.default_rng(seed)
        self.P = rng.normal(0, 0.1, (n_users, d))
        self.Q = rng.normal(0, 0.1, (n_items, d))
        self.bu, self.bi = np.zeros(n_users), np.zeros(n_items)
        self.A = np.zeros((n_users, n_aspects))
        self.mu, self.lr, self.reg = 0.0, lr, reg
        self.use_aspect, self.rng = use_aspect, rng

    def predict(self, u, i, s):
        out = self.mu + self.bu[u] + self.bi[i] + (self.P[u] * self.Q[i]).sum(1)
        return out + (self.A[u] * s).sum(1) if self.use_aspect else out

    def fit(self, u, i, r, s, epochs=40, batch=256):
        self.mu = r.mean()
        lr, lam = self.lr, self.reg
        for _ in range(epochs):
            for idx in np.array_split(self.rng.permutation(len(r)),
                                      max(1, len(r) // batch)):
                ub, ib, sb = u[idx], i[idx], s[idx]
                e = (r[idx] - self.predict(ub, ib, sb))[:, None]  # galat
                pu, qi = self.P[ub].copy(), self.Q[ib].copy()
                np.add.at(self.bu, ub, lr * (e[:, 0] - lam * self.bu[ub]))
                np.add.at(self.bi, ib, lr * (e[:, 0] - lam * self.bi[ib]))
                np.add.at(self.P, ub, lr * (e * qi - lam * pu))
                np.add.at(self.Q, ib, lr * (e * pu - lam * qi))
                if self.use_aspect:  # d r_hat / d a_u^k = s_ui^k
                    np.add.at(self.A, ub, lr * (e * sb - lam * self.A[ub]))
        return self
'''),
code(SIM),
code('''
def cf_dasar(D, tr, te, seed):
    """Prediksi CF (tanpa sentimen): OOF untuk baris latih, dan penuh untuk baris uji."""
    u, i, r, s = D["u"], D["i"], D["r"], D["s"]
    def fit_predict(a, b):
        m = AspectMF(D["m"], D["n"], D["K"], use_aspect=False, seed=seed).fit(u[a], i[a], r[a], s[a])
        return m.predict(u[b], i[b], None)
    cf_oof = oof_predictions(fit_predict, tr, seed=seed)
    m_full = AspectMF(D["m"], D["n"], D["K"], use_aspect=False, seed=seed).fit(u[tr], i[tr], r[tr], s[tr])
    return cf_oof, m_full.predict(u[tr], i[tr], None), m_full.predict(u[te], i[te], None)

def representasi(D):
    return {"mean": D["s"].mean(1, keepdims=True),
            "concat": D["s"],
            "concat_conf": np.column_stack([D["s"], D["c"]])}

D = simulasi_ulasan(noise=0.1, seed=0)
tr, te = bagi(len(D["r"]), 0)
cf_oof, cf_in, cf_te = cf_dasar(D, tr, te, seed=0)
y_te = D["r"][te]
print("RMSE CF saja (uji):", round(rmse(y_te, cf_te), 3))
print("RMSE CF pada data latih (in-sample): %.3f | out-of-fold: %.3f"
      % (rmse(D["r"][tr], cf_in), rmse(D["r"][tr], cf_oof)))
'''),
md("""
Perhatikan: prediksi *in-sample* jauh lebih akurat daripada *out-of-fold* karena model sudah melihat baris itu. Jika pohon fusi dilatih pada prediksi in-sample, ia belajar mempercayai CF terlalu banyak.
"""),
md("""
### Eksperimen A: out-of-fold versus in-sample

Latih pohon fusi dengan dua jenis fitur CF, lalu uji pada data uji yang sama.
"""),
code('''
rep = representasi(D)
rows = []
for nama, X in rep.items():
    f_oof = fit_fusion(cf_oof, X[tr], D["r"][tr], max_depth=5)
    f_in = fit_fusion(cf_in, X[tr], D["r"][tr], max_depth=5)
    rows.append({"representasi": nama,
                 "RMSE (fitur CF out-of-fold)": rmse(y_te, f_oof(cf_te, X[te])),
                 "RMSE (fitur CF in-sample)": rmse(y_te, f_in(cf_te, X[te]))})
bocor = pd.DataFrame(rows).set_index("representasi")
print(bocor.round(3))
assert (bocor["RMSE (fitur CF out-of-fold)"] < bocor["RMSE (fitur CF in-sample)"]).all()
'''),
md("""
**Interpretasi.** Fitur CF out-of-fold menghasilkan RMSE uji yang lebih baik untuk ketiga representasi. Pada fitur in-sample, distribusi fitur saat pelatihan (terlalu akurat) berbeda dengan saat uji (kurang akurat): ini bentuk kebocoran. Skema lipatan menjaga distribusi keduanya sama.
"""),
md("""
### Eksperimen B: representasi sentimen dan kedalaman pohon

Rata-rata tiga seed pada derau ekstraksi 0,1 dan 0,3. Kedalaman pohon 3, 5, dan 10 (nilai awal di buku).
"""),
code('''
def eval_grid(noise, depths=(3, 5, 10), seeds=(0, 1, 2)):
    hasil = {}
    for sd in seeds:
        Dn = simulasi_ulasan(noise=noise, seed=sd)
        trn, ten = bagi(len(Dn["r"]), sd)
        co, ci, ct = cf_dasar(Dn, trn, ten, seed=sd)
        hasil.setdefault("CF saja", []).append(rmse(Dn["r"][ten], ct))
        for nama, X in representasi(Dn).items():
            for dep in depths:
                f = fit_fusion(co, X[trn], Dn["r"][trn], max_depth=dep, seed=sd)
                hasil.setdefault((nama, dep), []).append(rmse(Dn["r"][ten], f(ct, X[ten])))
    return {k: float(np.mean(v)) for k, v in hasil.items()}

semua = {noise: eval_grid(noise) for noise in (0.1, 0.3)}
for noise, h in semua.items():
    t = pd.Series({k: v for k, v in h.items() if k != "CF saja"}).unstack()
    t.index.name = "representasi"; t.columns.name = "max_depth"
    print(f"noise = {noise} | RMSE CF saja = {h['CF saja']:.3f}")
    print(t.round(3), "\\n")
'''),
md("""
**Interpretasi (simulasi).**

1. Menambahkan sentimen ke CF menurunkan RMSE dibandingkan CF saja, pada kedalaman yang wajar.
2. **Kedalaman pohon lebih menentukan daripada pilihan representasi.** Pada kedalaman 10 (nilai awal di kode buku) semua varian memburuk karena pohon menghafal derau. Pada simulasi ini, kedalaman sekitar 5 lebih baik. Itu perlu disetel pada data validasi.
3. Selisih antar representasi kecil. `concat` sedikit unggul atas `mean` pada derau rendah, dan blok konfidens tidak menambah akurasi pada kedalaman 3 dan 5. Pada data nyata di buku pola ini berbeda antar domain, jadi jangan menyimpulkan peringkat representasi dari satu simulasi.
"""),
md("""
### Eksperimen C: uji berpasangan untuk selisih kecil

Bandingkan `concat` dengan `mean` (kedalaman 5, derau 0,1). Data dan set uji tetap, **seed model** divariasikan (inisialisasi MF, lipatan OOF, pohon). Itu setara dengan "sepuluh seed pada set uji yang sama" pada buku, dengan lima seed agar cepat.
"""),
code('''
D = simulasi_ulasan(noise=0.1, seed=0)
tr, te = bagi(len(D["r"]), 0)
y_te = D["r"][te]
se_mean, se_concat, se_cf = [], [], []
for sd in range(5):
    co, ci, ct = cf_dasar(D, tr, te, seed=sd)
    rep = representasi(D)
    for nama, X, tujuan in (("mean", rep["mean"], se_mean), ("concat", rep["concat"], se_concat)):
        f = fit_fusion(co, X[tr], D["r"][tr], max_depth=5, seed=sd)
        tujuan.append((y_te - f(ct, X[te])) ** 2)
    se_cf.append((y_te - ct) ** 2)

# (1) perbaikan besar: stacking(mean) terhadap CF saja
# (2) perbaikan kecil: concat terhadap mean
for judul, base, new in (("mean terhadap CF saja", se_cf, se_mean), ("concat terhadap mean", se_mean, se_concat)):
    ringkas = per_seed_summary(base, new)
    gain = rmse_gain_pct(np.mean(base, axis=0), np.mean(new, axis=0))
    ci = paired_bootstrap_ci(np.mean(base, axis=0), np.mean(new, axis=0))
    print(f"[{judul}]")
    print(f"  penurunan RMSE (rata-rata seed): {gain:.2f}% | CI 95% bootstrap berpasangan: [{ci[0]:.2f}, {ci[1]:.2f}]")
    print(f"  Wilcoxon p per seed: {[f'{p:.3g}' for p in ringkas['p_per_seed']]}")
    print(f"  seed signifikan (p < 0,05): {ringkas['n_sig']}/5 | p gabungan Fisher: {ringkas['p_fisher']:.3g}")
'''),
md("""
**Cara membaca.**

- Perbaikan besar (mean terhadap CF saja) jelas: CI jauh dari nol dan semua seed signifikan.
- Perbaikan kecil (concat terhadap mean) perlu hati-hati. Lihat apakah CI melewati nol dan berapa seed yang signifikan. Hasil bisa bergeser antar seed. Karena semua seed memakai **set uji yang sama**, p gabungan Fisher hanya merangkum, dan **bukan** bukti independen: p yang sangat kecil di sana tidak berarti efeknya besar.
- Signifikan secara statistik tidak sama dengan bermakna secara praktis. Laporkan besar efek (persen penurunan RMSE) bersama CI.
"""),
footer([
 "Mengapa `fit_predict` harus melatih model *baru* pada setiap lipatan, bukan memakai ulang model yang sama?",
 "Pada Eksperimen C, apa yang berubah jika set uji dibangkitkan ulang pada tiap seed? Mengapa p gabungan Fisher lebih sah dalam kasus itu?",
 "Tambahkan `min_samples_leaf` pada `DecisionTreeRegressor`. Apakah ia menggantikan penyetelan `max_depth`?",
]),
]
