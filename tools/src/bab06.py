import json
from nbkit import md, code
from common import header, footer

_CB = {x[1]: x[2] for x in json.load(open(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.dirname(__file__)), "codeblocks.json"), encoding="utf-8"))}
NB = {}

# ================================================================ 6.1 metrik ranking
P = "notebooks/bab06_evaluasi/01_metrik_ranking_dan_jebakan_evaluasi.ipynb"
NB[P] = [
header(P, "Metrik ranking dan jebakan evaluasi", 6, "Kode 6.x (metrik prediksi dan ketepatan daftar)",
 ["Menghitung MAE/RMSE, Hit Rate, Precision, Recall, F1, MRR, MAP, dan nDCG, dan memeriksanya pada contoh yang bisa dihitung tangan.",
  "Melihat bagaimana K dan jenis metrik dapat mengubah urutan model.",
  "Memahami mengapa evaluasi dengan negatif tersampel dapat menyesatkan."]),
md("## 1. Kode dari buku"),
code(_CB["b6_ranking_metrics.py"].split('import numpy as np\n', 1)[1].join(["import numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\n", ""])),
md("""
## 2. Contoh yang dapat dihitung tangan

Daftar rekomendasi `[a, b, c, d, e]`. Item relevan: `{b, e, z}` (satu relevan, `z`, tidak muncul sama sekali). Pada K = 5:

- Precision = 2/5 = 0,4; Recall = 2/3; MRR = 1/2 (relevan pertama di posisi 2).
- AP = (1/min(3,5)) · (1/2 + 2/5) = 0,3.
- nDCG (biner) = (1/log₂3 + 1/log₂6) / (1 + 1/log₂3 + 1/log₂4).
"""),
code('''
ranked = ["a", "b", "c", "d", "e"]
rel = {"b", "e", "z"}
print("P@5", precision(ranked, rel, 5), "| R@5", recall(ranked, rel, 5), "| MRR", reciprocal_rank(ranked, rel, 5))
print("AP@5", average_precision(ranked, rel, 5), "| HR@5", hit_rate(ranked, rel, 5))
rel_g = {"b": 1, "e": 1, "z": 1}
manual = (1/np.log2(3) + 1/np.log2(6)) / (1 + 1/np.log2(3) + 1/np.log2(4))
print("nDCG@5", ndcg(ranked, rel_g, 5), "| manual", manual)

assert abs(precision(ranked, rel, 5) - 0.4) < 1e-12
assert abs(recall(ranked, rel, 5) - 2/3) < 1e-12
assert abs(reciprocal_rank(ranked, rel, 5) - 0.5) < 1e-12
assert abs(average_precision(ranked, rel, 5) - 0.3) < 1e-12
assert abs(ndcg(ranked, rel_g, 5) - manual) < 1e-12
'''),
md("""
## 3. Simulasi: tiga model, banyak metrik

Dunia: 500 item dengan popularitas Zipf, 300 pengguna dengan selera laten. Item relevan bagi pengguna = 5 item uji yang disampel dari selera pengguna dengan peluang naik terhadap popularitas. Tiga model:

- **Populer**: peringkat berdasarkan popularitas global.
- **Personal-berderau**: skor selera sebenarnya + derau besar.
- **Personal-jernih**: skor selera sebenarnya + derau kecil.

(Model "personal" di sini bukan algoritma nyata; mereka hanya pembuat skor dengan kualitas berbeda yang dikendalikan, supaya perilaku *metrik* terlihat.)
"""),
code('''
rng = np.random.default_rng(0)
N_I, N_U, N_REL = 500, 300, 5
pop = 1.0 / np.arange(1, N_I + 1) ** 0.8
pop = rng.permutation(pop / pop.sum())                       # popularitas global per item
taste = rng.normal(0, 1, (N_U, N_I))                        # selera laten pengguna
logit_rel = 1.2 * taste + 1.0 * np.log(pop / pop.mean())     # peluang relevan: selera + popularitas
relevan = []
for u in range(N_U):
    p = np.exp(logit_rel[u] - logit_rel[u].max()); p /= p.sum()
    relevan.append(set(rng.choice(N_I, N_REL, replace=False, p=p)))

def peringkat(skor_fn):
    return [list(np.argsort(-skor_fn(u))) for u in range(N_U)]

model = {
    "Populer":          peringkat(lambda u: pop),
    "Personal-berderau": peringkat(lambda u: taste[u] + rng.normal(0, 2.5, N_I)),
    "Personal-jernih":  peringkat(lambda u: logit_rel[u] + rng.normal(0, 0.5, N_I)),
}

def tabel_metrik(model, K):
    rows = {}
    for nm, R in model.items():
        rows[nm] = {m.__name__: np.mean([m(R[u], relevan[u], K) for u in range(N_U)])
                    for m in (hit_rate, precision, recall, reciprocal_rank, average_precision)}
        rows[nm]["ndcg"] = np.mean([ndcg(R[u], {i: 1 for i in relevan[u]}, K) for u in range(N_U)])
    return pd.DataFrame(rows).T

for K in (5, 20, 100):
    print(f"\\n=== K = {K} ==="); print(tabel_metrik(model, K).round(3).to_string())
'''),
code('''
rows = []
for K in (1, 5, 10, 20, 50, 100):
    t = tabel_metrik(model, K)
    for nm in model:
        rows.append({"K": K, "model": nm, "ndcg": t.loc[nm, "ndcg"], "recall": t.loc[nm, "recall"]})
res = pd.DataFrame(rows)
fig, ax = plt.subplots(1, 2, figsize=(8.5, 3.2))
for nm, g in res.groupby("model"):
    ax[0].plot(g.K, g.ndcg, marker="o", label=nm); ax[1].plot(g.K, g.recall, marker="o", label=nm)
ax[0].set(xlabel="K", ylabel="nDCG@K", xscale="log"); ax[1].set(xlabel="K", ylabel="Recall@K", xscale="log")
ax[0].legend(fontsize=7); fig.tight_layout()

t10 = tabel_metrik(model, 10)
assert t10.loc["Personal-jernih", "ndcg"] > t10.loc["Populer", "ndcg"]
assert t10.loc["Personal-jernih", "ndcg"] > t10.loc["Personal-berderau", "ndcg"]
'''),
md("""
**Interpretasi.** Model jernih unggul di semua K, dan model berderau paling buruk. Hit Rate menjenuh: pada K = 100 ia sudah 0,99 (Populer) dan 1,00 (jernih), sehingga hampir tidak membedakan keduanya, padahal recall@100 (0,58 vs 0,75) dan nDCG masih membedakan jelas. Jadi metrik yang menjenuh pada K besar kehilangan daya pembeda; pilih K dan metrik yang sesuai skenario tampilan. Perhatikan juga bahwa model Populer *tidak sebodoh itu* di sini karena relevansi dalam simulasi memang dipengaruhi popularitas (parameter `1.0 * log(pop)`); pada dataset nyata, seberapa kuat bias popularitas pada data uji adalah hal yang harus diukur, bukan diasumsikan.
"""),
md("""
## 4. Jebakan: evaluasi dengan negatif tersampel

Praktik umum: untuk tiap item uji, ambil 99 item negatif acak lalu peringkatkan 100 kandidat itu. Ini murah, tetapi apakah urutan model sama dengan evaluasi peringkat penuh (seluruh katalog)? Bandingkan Hit Rate@10 pada kedua protokol.
"""),
code('''
def hr_penuh(skor_fn, K=10):
    h = []
    for u in range(N_U):
        s = skor_fn(u)
        ranked = np.argsort(-s)[:K]
        h += [i in set(ranked) for i in relevan[u]]
    return float(np.mean(h))

def hr_sampel(skor_fn, K=10, n_neg=99, seed=0):
    r = np.random.default_rng(seed); h = []
    for u in range(N_U):
        s = skor_fn(u)
        non_rel = np.array([i for i in range(N_I) if i not in relevan[u]])
        for i in relevan[u]:
            neg = r.choice(non_rel, n_neg, replace=False)
            rank = (s[neg] > s[i]).sum()          # peringkat item uji di antara 100 kandidat
            h.append(rank < K)
    return float(np.mean(h))

skor = {
    "Populer": lambda u: pop,
    "Personal-berderau": lambda u, _r=np.random.default_rng(5): taste[u] + _r.normal(0, 2.5, N_I),
    "Personal-jernih": lambda u, _r=np.random.default_rng(6): logit_rel[u] + _r.normal(0, 0.5, N_I),
}
cmp_ = pd.DataFrame({nm: {"HR@10 penuh": hr_penuh(f), "HR@10 sampel (99 neg.)": hr_sampel(f)} for nm, f in skor.items()}).T
print(cmp_.round(3))
assert (cmp_["HR@10 sampel (99 neg.)"] > cmp_["HR@10 penuh"]).all()      # sampel selalu menggembungkan angka
'''),
md("""
**Interpretasi.** Evaluasi tersampel menggembungkan angka mutlak secara besar-besaran: lawan yang dihadapi hanya 99 item, bukan 499. Itu sebabnya angka dari protokol tersampel **tidak boleh dibandingkan** dengan angka dari protokol penuh atau dengan paper lain yang memakai protokol berbeda. Urutan antar model bisa tetap sama pada simulasi ini, tetapi literatur melaporkan kasus yang urutannya berbalik. Prinsip aman: laporkan protokol, gunakan peringkat penuh bila memungkinkan, dan jangan membandingkan angka lintas protokol.
"""),
footer([
 "Pada contoh tangan, bagaimana nilai AP@5 berubah bila item `z` dihitung sebagai relevan yang 'tidak muncul' vs. dikeluarkan dari himpunan relevan? Mengapa penyebut `min(|rel|, K)` dipilih?",
 "Naikkan `N_REL` dari 5 ke 50. Metrik mana yang paling terpengaruh, dan mengapa?",
 "Ganti negatif acak pada evaluasi tersampel dengan negatif yang disampel sesuai popularitas. Bagaimana jarak antar protokol berubah?",
]),
]

# ================================================================ 6.2 beyond + afektif
P = "notebooks/bab06_evaluasi/02_metrik_beyond_accuracy_dan_afektif.ipynb"
NB[P] = [
header(P, "Metrik di luar akurasi dan keselarasan emosi", 6, "Kode 6.x (coverage, Gini, novelty, ILD, EAS, QHR, KL afektif)",
 ["Menghitung coverage, Gini, novelty, ILD, EAS, QHR, dan KL afektif.",
  "Melihat pertukaran akurasi dan keragaman/keselarasan pada simulasi.",
  "Mengenali metrik yang bisa 'dicurangi' (EAS tinggi dengan rekomendasi netral)."]),
md("## 1. Kode dari buku"),
code("import pandas as pd\nimport matplotlib.pyplot as plt\n" + _CB["b6_beyond_affective.py"].replace('"""Metrik di luar akurasi dan keselarasan emosi (Pers. 6-6 s.d. 6-12)."""\n', "")),
md("""
## 2. Dunia simulasi

Katalog 400 item dengan koordinat valence–arousal di empat kuadran dan popularitas Zipf. Tiap pengguna punya mood saat ini `m` (salah satu pusat kuadran) dan relevansi stabil per item. Empat pembuat daftar (top-10 dari 50 kandidat relevansi tertinggi):

| Pembuat daftar | Cara memilih |
|---|---|
| **Populer** | 10 item paling populer, tanpa personalisasi |
| **Relevansi-saja** | 10 skor relevansi tertinggi |
| **Selaras-mood** | relevansi + bobot besar untuk kedekatan ke mood |
| **Netral** | item yang koordinatnya paling dekat ke pusat (0,0) |
"""),
code('''
rng = np.random.default_rng(0)
N_I, N_U, K = 400, 300, 10
PUSAT = np.array([[.6, .6], [-.6, .6], [-.6, -.6], [.6, -.6]])
va = np.clip(PUSAT[rng.integers(0, 4, N_I)] + rng.normal(0, .25, (N_I, 2)), -1, 1)
pop_cnt = rng.permutation(1000.0 / np.arange(1, N_I + 1) ** 0.9)         # jumlah interaksi per item
pop_cnt = np.maximum(pop_cnt, 1.0)
n_users_total = 1000

rel_u = rng.normal(0, 1, (N_U, N_I)) + 0.5 * np.log(pop_cnt / pop_cnt.mean())
mood_u = PUSAT[rng.integers(0, 4, N_U)]
# distribusi kuadran riwayat per pengguna (untuk KL): sebagian besar di kuadran mood-nya
p_hist = np.array([0.1 + 0.6 * (np.arange(4) == quadrant(m)) for m in mood_u])
p_hist /= p_hist.sum(1, keepdims=True)

def daftar(strategi):
    out = []
    for u in range(N_U):
        cand = np.argsort(-rel_u[u])[:50]
        if strategi == "Populer":
            L = list(np.argsort(-pop_cnt)[:K])
        elif strategi == "Relevansi-saja":
            L = list(cand[:K])
        elif strategi == "Selaras-mood":
            a = np.array([align(va[i], mood_u[u]) for i in cand])
            r = (rel_u[u][cand] - rel_u[u][cand].min()) / np.ptp(rel_u[u][cand])
            L = list(cand[np.argsort(-(0.4 * r + 0.6 * a))[:K]])
        else:                                                                # Netral
            L = list(cand[np.argsort(np.linalg.norm(va[cand], axis=1))[:K]])
        out.append(L)
    return out

strategi = ["Populer", "Relevansi-saja", "Selaras-mood", "Netral"]
L = {s: daftar(s) for s in strategi}

def ringkas(Ls):
    expo = np.bincount(np.concatenate(Ls), minlength=N_I)
    # relevansi (rata-rata z-skor relevansi item yang direkomendasikan) sebagai proksi akurasi
    rel_rata = np.mean([rel_u[u][Ls[u]].mean() for u in range(N_U)])
    return {"relevansi rata-rata": rel_rata,
            "coverage": coverage(Ls, N_I), "Gini eksposur": gini(expo),
            "novelty": np.mean([novelty(l, dict(enumerate(pop_cnt)), n_users_total) for l in Ls]),
            "ILD": np.mean([ild(l, va) for l in Ls]),
            "EAS": np.mean([eas(Ls[u], va, mood_u[u]) for u in range(N_U)]),
            "QHR": np.mean([qhr(Ls[u], va, mood_u[u]) for u in range(N_U)]),
            "KL afektif": np.mean([affective_kl(Ls[u], va, p_hist[u]) for u in range(N_U)])}

tab = pd.DataFrame({s: ringkas(L[s]) for s in strategi}).T
print(tab.round(3).to_string())
'''),
md("""
## 3. Membaca tabel dan menguji klaim
"""),
code('''
# Klaim-klaim di bawah dibuktikan oleh data simulasi, bukan diasumsikan.
assert tab.loc["Populer", "coverage"] < tab.loc["Relevansi-saja", "coverage"]       # populer: cakupan sempit
assert tab.loc["Populer", "Gini eksposur"] > tab.loc["Relevansi-saja", "Gini eksposur"]
assert tab.loc["Populer", "novelty"] < tab.loc["Relevansi-saja", "novelty"]         # populer: tidak baru
assert tab.loc["Relevansi-saja", "relevansi rata-rata"] > tab.loc["Selaras-mood", "relevansi rata-rata"]
assert tab.loc["Selaras-mood", "QHR"] > tab.loc["Relevansi-saja", "QHR"]
assert tab.loc["Selaras-mood", "EAS"] > tab.loc["Relevansi-saja", "EAS"]
# Jebakan: daftar netral menang di EAS secara murah, tetapi kalah di QHR dan keragaman afektif
assert tab.loc["Netral", "ILD"] < tab.loc["Relevansi-saja", "ILD"]
assert tab.loc["Netral", "QHR"] < tab.loc["Selaras-mood", "QHR"]

fig, ax = plt.subplots(figsize=(7.5, 3.4))
z = (tab - tab.min()) / (tab.max() - tab.min() + 1e-12)
im = ax.imshow(z.T.values, cmap="viridis", aspect="auto")
ax.set_xticks(range(len(z.index)), z.index, fontsize=8); ax.set_yticks(range(z.shape[1]), z.columns, fontsize=8)
for i in range(z.shape[1]):
    for j in range(z.shape[0]):
        ax.text(j, i, f"{tab.iloc[j, i]:.2f}", ha="center", va="center", color="w", fontsize=7)
ax.set_title("Metrik per pembuat daftar (warna dinormalisasi per baris)")
fig.tight_layout()
'''),
md("""
**Interpretasi (dari angka di atas).**

- **Populer** memiliki coverage dan novelty terendah serta Gini tertinggi: tidak personal, menumpuk eksposur pada sedikit item.
- **Relevansi-saja** paling relevan, tetapi tidak mempedulikan mood sehingga EAS dan QHR-nya lebih rendah daripada **Selaras-mood**.
- **Selaras-mood** menukar sedikit relevansi dengan keselarasan emosi yang jauh lebih baik. Pertukaran ini adalah inti Bab 5.
- **Netral** adalah jebakan: item di dekat pusat valence–arousal memiliki jarak sedang ke semua mood sehingga EAS-nya terlihat lumayan tanpa benar-benar cocok dengan mood siapa pun. QHR (kecocokan kuadran) dan ILD (keragaman) membongkarnya. Karena itu satu metrik afektif saja tidak cukup; laporkan beberapa yang saling mengimbangi. KL afektif terendah pada Selaras-mood: distribusi kuadran daftarnya paling dekat dengan riwayat pengguna (yang dalam simulasi ini condong ke kuadran mood).
"""),
md("""
## 4. Eksperimen: kurva pertukaran bobot afektif

Variasikan bobot mood `w` pada skor `(1-w)·relevansi + w·keselarasan`. Plot relevansi vs EAS/QHR.
"""),
code('''
def daftar_w(w):
    out = []
    for u in range(N_U):
        cand = np.argsort(-rel_u[u])[:50]
        a = np.array([align(va[i], mood_u[u]) for i in cand])
        r = (rel_u[u][cand] - rel_u[u][cand].min()) / np.ptp(rel_u[u][cand])
        out.append(list(cand[np.argsort(-((1 - w) * r + w * a))[:K]]))
    return out

rows = []
for w in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
    Ls = daftar_w(w)
    rows.append({"w": w, "relevansi": np.mean([rel_u[u][Ls[u]].mean() for u in range(N_U)]),
                 "EAS": np.mean([eas(Ls[u], va, mood_u[u]) for u in range(N_U)]),
                 "QHR": np.mean([qhr(Ls[u], va, mood_u[u]) for u in range(N_U)]),
                 "ILD": np.mean([ild(Ls[u], va) for u in range(N_U)])})
kurva = pd.DataFrame(rows).set_index("w")
print(kurva.round(3))
assert kurva["relevansi"].is_monotonic_decreasing
assert kurva["QHR"].iloc[-1] > kurva["QHR"].iloc[0]
assert kurva["ILD"].iloc[-1] < kurva["ILD"].iloc[0]
'''),
md("""
**Interpretasi.** Relevansi turun monoton ketika bobot afektif naik, sedangkan QHR dan EAS naik. ILD turun karena daftar makin mengumpul di satu area valence–arousal. Jadi mengejar keselarasan emosi memiliki harga ganda: relevansi dan keragaman. Pemilihan `w` adalah keputusan produk, bukan hanya statistik; kurva ini membantu membuat keputusan itu eksplisit.
"""),
footer([
 "Mengapa `affective_kl` mencampur `q` dengan `p` (smoothing `eps`)? Apa yang terjadi bila `eps = 0` dan satu kuadran tidak pernah direkomendasikan?",
 "Rancang pembuat daftar lain yang menaikkan EAS tetapi merusak QHR. Apakah ia lolos dari asersi?",
 "Bagaimana mengubah novelty agar tidak bergantung pada `n_users_total` yang tidak diketahui pada sistem baru?",
]),
]

# ================================================================ 6.3 signifikansi
P = "notebooks/bab06_evaluasi/03_uji_signifikansi_dan_koreksi_holm.ipynb"
NB[P] = [
header(P, "Uji signifikansi, bootstrap, Holm, dan besar efek", 6, "Kode 6.x (uji berpasangan)",
 ["Menjalankan uji berpasangan (Wilcoxon, uji-t), bootstrap CI, dan koreksi Holm.",
  "Melihat bahwa p kecil tidak berarti efek besar (d_z) dan sebaliknya.",
  "Mengamati inflasi positif palsu pada perbandingan jamak tanpa koreksi."]),
md("## 1. Kode dari buku"),
code("import pandas as pd\nimport matplotlib.pyplot as plt\n" + _CB["b6_significance.py"].replace('"""Uji berpasangan, bootstrap, koreksi Holm, dan besar efek (Pers. 6-13, 6-14)."""\n', "")),
md("""
## 2. Skor per pengguna

Metrik per pengguna (misalnya nDCG@10) biasanya tidak normal: banyak nol, ekor panjang. Kita buat dua model dengan skor per pengguna dari campuran nol dan Beta, di mana B memiliki sedikit perbaikan nyata.
"""),
code('''
rng = np.random.default_rng(0)

def skor_pengguna(n, rata, p_nol=0.5, rng=rng):
    """Campuran: p_nol pengguna dapat skor 0; sisanya Beta (rata-rata sekitar `rata`). Rata-rata keseluruhan lebih rendah karena nol."""
    nol = rng.random(n) < p_nol
    b = rng.beta(2, 2 / rata - 2 + 1e-9, n) if rata < 1 else rng.beta(2, 2, n)
    return np.where(nol, 0.0, b)

def varian(base, efek, r):
    """Model B = A + efek + derau per pengguna. Pengguna dengan skor 0 tetap 0 (tidak ada yang bisa
    diperbaiki), dan skor di-clip ke [0, 1]. Penanganan nol ini penting: meng-clip nol akan
    membuat B > A secara artifisial walau efek = 0."""
    return np.where(base > 0, np.clip(base + r.normal(efek, 0.10, len(base)), 0, 1), 0.0)

n = 500
A = skor_pengguna(n, 0.30)
B = varian(A, 0.02, rng)
print("rata-rata A:", A.mean().round(4), "| B:", B.mean().round(4), "| proporsi skor nol A:", (A == 0).mean().round(2))
hasil = paired_tests(A, B)
ci = bootstrap_ci(A, B)
print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in hasil.items()})
print("CI bootstrap 95% untuk selisih rata-rata:", ci.round(4))
'''),
md("""
## 3. p kecil ≠ efek besar

Selisih sebenarnya sama-sama kecil. Perbesar `n` dan lihat `p` serta `d_z` bergerak berbeda: `p` menyusut cepat terhadap `n`, sedangkan `d_z` mencerminkan efek per pengguna dan relatif stabil.
"""),
code('''
rows = []
for n_ in (50, 200, 1000, 5000, 20000):
    r_ = np.random.default_rng(1)
    a_ = skor_pengguna(n_, 0.30, rng=r_)
    b_ = varian(a_, 0.01, r_)                                 # efek sejati kecil: 0.01
    h = paired_tests(a_, b_)
    rows.append({"n": n_, "selisih": h["mean_diff"], "p_t": h["p_ttest"], "p_wilcoxon": h["p_wilcoxon"], "d_z": h["d_z"]})
skala = pd.DataFrame(rows).set_index("n")
print(skala.round(4))
assert skala["p_t"].iloc[-1] < skala["p_t"].iloc[0]
assert skala["d_z"].abs().max() < 0.5          # efek sebenarnya tetap kecil di semua n
'''),
md("""
**Interpretasi.** Dengan data cukup banyak, efek yang tak berarti secara praktis pun menjadi "signifikan". Itu sebabnya buku merekomendasikan melaporkan selisih rata-rata dengan CI dan `d_z`, bukan `p` saja. Dalam simulasi ini, `d_z` tetap kecil (di bawah 0,5) pada semua `n`, sedangkan `p` menyusut. Perhatikan juga bahwa pada `n` kecil, `p` besar belum berarti tidak ada efek: itu berarti datanya kurang kuat untuk memutuskan (daya uji rendah).
"""),
md("""
## 4. Perbandingan jamak: inflasi positif palsu dan Holm

Bayangkan memperbandingkan 20 varian model terhadap baseline. Pada simulasi ini **tidak ada** varian yang lebih baik (hipotesis nol benar untuk semuanya). Berapa varian yang "signifikan" pada α = 0,05 tanpa dan dengan Holm? Ulangi 200 percobaan.
"""),
code('''
def satu_percobaan(seed, m=20, n=300):
    r = np.random.default_rng(seed)
    base = skor_pengguna(n, 0.30, rng=r)
    p = []
    for _ in range(m):
        v = varian(base, 0.0, r)                                    # tidak ada efek sejati
        p.append(paired_tests(base, v)["p_wilcoxon"])
    return np.array(p)

mentah, terkoreksi = [], []
for s in range(200):
    p = satu_percobaan(s)
    mentah.append((p < 0.05).any()); terkoreksi.append(any(holm(p)))
fwer_mentah, fwer_holm = np.mean(mentah), np.mean(terkoreksi)
print(f"peluang >=1 positif palsu: tanpa koreksi {fwer_mentah:.2f} | dengan Holm {fwer_holm:.2f}")
print(f"teori tanpa koreksi: 1-(0.95)^20 = {1 - 0.95**20:.2f}")
assert fwer_mentah > 0.5
assert fwer_holm < 0.12
'''),
md("""
**Interpretasi.** Menjalankan 20 pembandingan pada α = 0,05 hampir pasti menghasilkan setidaknya satu "temuan" palsu (teori: ~64%). Holm menekan peluang itu mendekati α (family-wise error rate) tanpa mengasumsikan independensi. Dalam penelitian nyata, jumlah perbandingan yang diuji, termasuk yang dibuang, harus dilaporkan. Catatan: simulasi memakai derau independen antar varian; pada varian yang saling berkorelasi, perilakunya berbeda.
"""),
md("""
## 5. Daya uji: seberapa sering efek nyata terdeteksi?
"""),
code('''
def daya(efek, n, ulang=200):
    det = 0
    for s in range(ulang):
        r = np.random.default_rng(1000 + s)
        a_ = skor_pengguna(n, 0.30, rng=r); b_ = varian(a_, efek, r)
        det += paired_tests(a_, b_)["p_wilcoxon"] < 0.05
    return det / ulang

grid = pd.DataFrame({f"efek={e}": [daya(e, n_) for n_ in (50, 200, 800)] for e in (0.0, 0.01, 0.03)}, index=(50, 200, 800))
grid.index.name = "n"
print(grid.round(2))
assert grid["efek=0.0"].max() < 0.12            # tanpa efek: ~alpha
assert grid["efek=0.03"].iloc[-1] > grid["efek=0.03"].iloc[0]
'''),
md("""
**Interpretasi.** Kolom `efek=0.0` memperlihatkan tingkat positif palsu uji tunggal: sekitar α. Untuk efek kecil, `n` kecil hampir tidak pernah mendeteksinya. Jadi hasil "tidak signifikan" pada sampel kecil bukan bukti ketiadaan efek.
"""),
footer([
 "Mengapa Wilcoxon sering dipilih untuk metrik per pengguna? Kapan uji-t sah walau data tidak normal?",
 "Ubah `skor_pengguna` menjadi tanpa nol (`p_nol=0`). Bagaimana hasil Wilcoxon vs uji-t berubah?",
 "Pada Bagian 4, buat varian yang saling berkorelasi (derau bersama). Bagaimana FWER tanpa koreksi berubah?",
]),
]
