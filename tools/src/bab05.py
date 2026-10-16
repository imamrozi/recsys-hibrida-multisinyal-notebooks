from nbkit import md, code
from common import header, footer

NB = {}

# ------------------------------------------------------------------ 5.1 VA
P = "notebooks/bab05_emosi_sesi/01_koordinat_valence_arousal.ipynb"
NB[P] = [
header(P, "Koordinat valence–arousal item dari tag dan teks", 5, "Kode 5.2.1",
 ["Memetakan kata ke koordinat valence–arousal di [-1, 1]² dan menghitung keyakinan (κ).",
  "Memahami bagaimana bukti, konsistensi, dan cakupan kata memengaruhi κ.",
  "Menggabungkan sumber tag dan teks dengan rata-rata berbobot keyakinan."],
 catatan="> **Leksikon di sini adalah leksikon mainan** (±36 kata, nilai tiruan pada skala 1–9 ala Warriner). Nilai itu **bukan** data Warriner atau NRC yang sebenarnya, yang berlisensi sendiri dan tidak disertakan. Untuk riset, ganti `LEKSIKON_MENTAH` dengan leksikon asli yang Anda unduh sesuai lisensinya."),
md("## 1. Kode dari buku"),
code('''
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def warriner_to_unit(x):          # Pers. (5-1): skala 1..9 -> [-1, 1]
    return (np.asarray(x, float) - 5.0) / 4.0

def nrc_to_unit(x):               # Pers. (5-1): skala 0..1 -> [-1, 1]
    return 2.0 * np.asarray(x, float) - 1.0

def score_tokens(tokens, lexicon, n_cap=10):
    """lexicon: {kata: (valence, arousal)} sudah di [-1, 1].
    Mengembalikan (v, kappa): rata-rata VA kata tercakup dan keyakinannya."""
    hits = np.array([lexicon[t] for t in tokens if t in lexicon])
    if len(hits) == 0:
        return np.zeros(2), 0.0
    v = hits.mean(axis=0)
    evidence = min(len(hits) / n_cap, 1.0)                  # banyaknya bukti
    spread = hits.std(axis=0).mean() if len(hits) > 1 else 0.5
    consistency = 1.0 - min(spread, 1.0)                    # kesepakatan bukti
    return v, evidence * consistency

def from_tags(tags, lexicon):
    return score_tokens([t.lower().strip() for t in tags], lexicon)

def from_text(text, lexicon):
    tokens = re.findall(r"[a-z']+", text.lower())
    v, kappa = score_tokens(tokens, lexicon)
    coverage = sum(t in lexicon for t in tokens) / max(len(tokens), 1)
    return v, kappa * min(coverage / 0.2, 1.0)              # Pers. (5-2)

def fuse(sources):
    """sources: list[(v, kappa)]. Pers. (5-3): rata-rata berbobot keyakinan,
    keyakinan gabungan 1 - prod(1 - kappa_s)."""
    k = np.array([s[1] for s in sources])
    if k.sum() == 0:
        return np.zeros(2), 0.0                             # cadangan: netral
    v = (k[:, None] * np.array([s[0] for s in sources])).sum(0) / k.sum()
    return np.clip(v, -1, 1), float(1 - np.prod(1 - k))
'''),
md("## 2. Leksikon mainan"),
code('''
# (valence, arousal) pada skala 1..9. NILAI TIRUAN untuk demonstrasi, bukan Warriner asli.
LEKSIKON_MENTAH = {
    "happy": (8.2, 6.5), "joyful": (8.5, 6.2), "cheerful": (8.0, 5.9), "party": (7.9, 7.0),
    "dance": (7.6, 6.9), "energetic": (7.0, 7.8), "thrilling": (7.1, 7.9), "exciting": (7.8, 7.6),
    "love": (8.7, 5.9), "summer": (7.4, 5.2), "morning": (6.5, 4.3),
    "tender": (7.2, 3.5), "peaceful": (7.6, 2.6), "calm": (7.2, 2.1), "relax": (7.4, 2.4),
    "gentle": (7.0, 2.8), "soothing": (7.3, 2.3), "dreamy": (6.8, 3.2),
    "sad": (2.1, 3.9), "lonely": (2.2, 3.6), "gloomy": (2.4, 3.0), "melancholy": (2.8, 3.2),
    "tired": (3.0, 2.6), "bored": (3.2, 2.5), "dark": (3.0, 4.8), "rain": (4.9, 3.8),
    "angry": (2.3, 7.5), "furious": (1.9, 8.0), "tense": (2.9, 7.0), "aggressive": (2.6, 7.6),
    "anxious": (2.7, 6.6), "scary": (2.5, 7.3), "heavy": (3.4, 5.4),
    "night": (5.4, 3.3), "road": (5.5, 4.5), "city": (5.6, 5.0),
}
LEKSIKON = {k: tuple(warriner_to_unit(v)) for k, v in LEKSIKON_MENTAH.items()}

def kuadran(v):
    """0: senang-bergairah, 1: sedih-bergairah, 2: sedih-tenang, 3: senang-tenang."""
    if v[0] >= 0:
        return 0 if v[1] >= 0 else 3
    return 1 if v[1] >= 0 else 2

print("contoh:", {k: tuple(np.round(LEKSIKON[k], 2)) for k in ("calm", "furious", "party")})
'''),
md("""
## 3. Eksperimen A: tag, teks, dan gabungan

Enam trek contoh. Perhatikan κ untuk tiap sumber, dan apa yang terjadi ketika sumber saling bertentangan atau tidak ada.
"""),
code('''
trek = {
    "Sunrise Run":   (["happy", "energetic", "summer"], "an upbeat track full of energy and joy for the morning run"),
    "Slow Rain":     (["sad", "rain", "melancholy"], "a gentle and gloomy song for a rainy night, tender and lonely"),
    "Midnight Rage": (["angry", "aggressive", "dark"], "furious heavy riffs, tense and scary from the first bar"),
    "Lake Evening":  (["calm", "peaceful"], "a calm and soothing piece to relax and breathe slowly"),
    "Mixed Signals": (["happy", "sad", "angry", "calm"], "a song about a city road at night"),
    "Tanpa Tag":     ([], "dreamy and gentle sounds, a soothing night to relax"),
}
rows, koordinat = [], {}
for nama, (tags, teks) in trek.items():
    vt, kt = from_tags(tags, LEKSIKON)
    vx, kx = from_text(teks, LEKSIKON)
    vf, kf = fuse([(vt, kt), (vx, kx)])
    koordinat[nama] = vf
    rows.append({"trek": nama, "v_tag": vt[0], "a_tag": vt[1], "k_tag": kt,
                 "v_teks": vx[0], "a_teks": vx[1], "k_teks": kx,
                 "v_gab": vf[0], "a_gab": vf[1], "k_gab": kf, "kuadran": kuadran(vf)})
tabel = pd.DataFrame(rows).set_index("trek")
print(tabel.round(2).to_string())
'''),
code('''
fig, ax = plt.subplots(figsize=(4.8, 4.4))
ax.axhline(0, c="lightgray"); ax.axvline(0, c="lightgray")
for nama, v in koordinat.items():
    ax.scatter(*v, s=60 * (0.3 + tabel.loc[nama, "k_gab"] * 2))
    ax.annotate(nama, v, textcoords="offset points", xytext=(5, 4), fontsize=8)
ax.set(xlim=(-1, 1), ylim=(-1, 1), xlabel="valence", ylabel="arousal",
       title="Koordinat gabungan (ukuran titik = κ)")
fig.tight_layout()

# klaim yang diuji: trek koheren memiliki κ gabungan lebih tinggi daripada trek yang tags-nya bertentangan
assert tabel.loc["Midnight Rage", "k_gab"] > tabel.loc["Mixed Signals", "k_gab"]
assert tabel.loc["Tanpa Tag", "k_tag"] == 0.0       # tanpa tag: tidak ada bukti dari sumber tag
'''),
md("""
**Interpretasi.** Trek yang tag-nya sepakat (Midnight Rage, Lake Evening) memiliki koordinat yang jelas dan κ lebih besar. *Mixed Signals* memuat tag yang saling bertentangan sehingga rata-ratanya mendekati pusat dan κ-nya rendah. Pada *Tanpa Tag* sumber tag nol, tetapi sumber teks menutupinya. Titik yang kecil pada plot adalah titik yang tidak boleh dipercaya penuh oleh hilir, misalnya oleh re-ranking di notebook berikutnya.
"""),
md("""
## 4. Eksperimen B: bukti, konsistensi, dan batas n_cap

Pada `score_tokens`, κ = bukti × konsistensi. Bukti naik linier sampai `n_cap` kata. Bagaimana κ berubah bila kita menambah tag yang sejenis, dan bila kita menambah tag yang bertentangan?
"""),
code('''
tenang = ["calm", "peaceful", "relax", "gentle", "soothing", "dreamy", "tender"]
campur = ["calm", "furious", "happy", "sad", "party", "lonely", "angry", "peaceful"]
rows = []
for k in range(1, 8):
    rows.append({"jumlah_tag": k,
                 "κ (semua tenang)": score_tokens(tenang[:k], LEKSIKON)[1],
                 "κ (campur aduk)": score_tokens(campur[:k], LEKSIKON)[1]})
hasil_k = pd.DataFrame(rows).set_index("jumlah_tag")
print(hasil_k.round(3))
assert (hasil_k["κ (semua tenang)"].iloc[-1] > hasil_k["κ (campur aduk)"].iloc[-1])
'''),
md("""
**Interpretasi.** Satu tag mendapat κ rendah karena buktinya sedikit (konsistensi bernilai 0,5 sebagai nilai awal). Tag yang sejenis menaikkan κ. Tag yang saling bertentangan menaikkan bukti tetapi menurunkan konsistensi, sehingga κ-nya tidak naik. Ini sifat yang diinginkan: lebih banyak data tidak otomatis lebih meyakinkan jika datanya saling membantah.
"""),
md("""
## 5. Eksperimen C: sifat penggabungan

Keyakinan gabungan `1 − ∏(1 − κ_s)` tidak pernah lebih kecil dari κ sumber terbaik, dan bernilai 0 bila semua sumber tidak punya bukti. Uji pada banyak pasangan acak.
"""),
code('''
rng = np.random.default_rng(0)
ok_maks, ok_nol = True, True
for _ in range(2000):
    s = [(rng.uniform(-1, 1, 2), rng.uniform(0, 1)) for _ in range(rng.integers(1, 4))]
    v, kf = fuse(s)
    ok_maks &= kf >= max(x[1] for x in s) - 1e-12
    assert -1 <= v.min() and v.max() <= 1
print("κ gabungan >= κ terbaik pada 2000 percobaan acak:", ok_maks)
v0, k0 = fuse([(np.array([0.9, 0.9]), 0.0), (np.array([-0.9, 0.1]), 0.0)])
print("tanpa bukti sama sekali ->", v0, k0)
assert ok_maks and k0 == 0.0 and not v0.any()
'''),
footer([
 "Mengapa `spread` diberi nilai 0,5 ketika hanya ada satu kata yang tercakup? Apa akibatnya bila nilainya 0?",
 "Ganti `n_cap` menjadi 3 dan 30. Bagaimana κ trek berubah, dan mengapa pengaturan itu bergantung pada panjang tag di dataset Anda?",
 "Apa risiko memakai koordinat dari teks ulasan pengguna untuk item musik, dibanding dari tag editorial?",
]),
]

# ------------------------------------------------------------------ 5.2 mood sesi + rerank
P = "notebooks/bab05_emosi_sesi/02_mood_sesi_dan_reranking.ipynb"
NB[P] = [
header(P, "Mood sesi (EWMA) dan re-ranking sadar-emosi", 5, "Kode 5.5.1 dan 5.5.2",
 ["Melacak mood sesi dengan EWMA berbobot keyakinan dan mendeteksi peralihan.",
  "Menyetel bobot afektif λ pada re-ranking, dan melihat bahwa λ optimum bergantung pada seberapa kuat mood memengaruhi perilaku.",
  "Mengukur kerusakan akibat koordinat item yang tidak andal, dan peran κ."],
 catatan="> Seluruh sesi dan perilaku pengguna adalah **simulasi** dengan asumsi eksplisit (lihat `sesi`). Ini ilustrasi mekanisme re-ranking, bukan klaim tentang pengguna nyata. Bahwa mood memengaruhi pilihan (parameter `gamma`) adalah **asumsi** yang kita variasikan, bukan fakta."),
md("## 1. Kode dari buku: mood sesi (Kode 5.5.1)"),
code('''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ewma_step(m_prev, v, alpha, kappa=1.0):
    """Satu langkah EWMA; kappa kecil -> item meragukan menggeser mood sedikit."""
    if m_prev is None:
        return np.asarray(v, float)
    a = alpha * kappa
    return a * np.asarray(v, float) + (1 - a) * m_prev

def update_mood(state, v, kappa=1.0, a_main=0.5, a_short=0.6, a_long=0.2, theta=0.35):
    """state: dict berisi 'm', 'short', 'long' (None di awal sesi).
    Mengembalikan state baru, mood teramati, dan tanda peralihan."""
    s = {k: ewma_step(state.get(k), v, a, kappa)
         for k, a in (("m", a_main), ("short", a_short), ("long", a_long))}
    shift = np.linalg.norm(s["short"] - s["long"]) > theta
    mood = s["short"] if shift else s["m"]        # ikuti EWMA pendek saat beralih
    return s, mood, bool(shift)

def target_mood(m, m_goal=None, gamma=0.0):
    """Pers. (5-13): gamma = 0 -> selaras; gamma > 0 -> regulasi ke m_goal."""
    if m_goal is None or gamma == 0:
        return m
    return m + gamma * (np.asarray(m_goal, float) - m)
'''),
md("## 2. Kode dari buku: re-ranking (Kode 5.5.2)"),
code('''
DIAG = 2 * np.sqrt(2)                       # jarak maksimum di [-1, 1]^2

def alignment(V, m):
    """Pers. (5-14): a = 1 - ||v - m|| / (2 sqrt 2); V: (n, 2), m: (2,)."""
    return 1.0 - np.linalg.norm(V - m, axis=1) / DIAG

def rerank(cand_ids, r, V, m_star, K=10, lam=0.3, g_bar_u=None, lam_max=0.6, kappa=None):
    """cand_ids, r: kandidat dan skor Tahap A; V: koordinat VA kandidat."""
    r = np.asarray(r, float)
    r_t = (r - r.min()) / max(r.max() - r.min(), 1e-12)          # Pers. (5-15)
    a = alignment(np.asarray(V, float), np.asarray(m_star, float))
    if kappa is not None:                     # item meragukan -> a ditarik ke netral
        a = kappa * a + (1 - kappa) * 0.5
    if g_bar_u is not None:                   # Pers. (5-17), perluasan konseptual
        lam = lam_max * (1.0 - g_bar_u)
    s = (1 - lam) * r_t + lam * a                                  # Pers. (5-16)
    order = np.argsort(-s)[:K]
    return [cand_ids[j] for j in order], s[order], a[order]
'''),
md("""
## 3. Simulasi sesi

Asumsi yang dibuat:

- 400 item di empat kuadran valence–arousal.
- Tiap pengguna punya utilitas stabil per item (`u`). Pilihan item pada langkah *t* mengikuti softmax dari `beta · (u + gamma · keselarasan(item, mood_t))`. `gamma` = seberapa kuat mood memengaruhi pilihan.
- Mood sebenarnya konstan pada satu kuadran, lalu (pada sesi "peralihan") pindah ke kuadran lain di tengah sesi.
- Sistem hanya melihat koordinat item yang dikonsumsi, dengan derau `sig_c` (koordinat tidak sempurna).
- Tahap A (mood-buta) memberi skor `u + derau`; re-ranking mengambil 50 kandidat teratas dan memilih 10.
"""),
code('''
N_ITEM = 400
PUSAT = np.array([[.6, .6], [-.6, .6], [-.6, -.6], [.6, -.6]])

def katalog(seed=0):
    rng = np.random.default_rng(seed)
    return np.clip(PUSAT[rng.integers(0, 4, N_ITEM)] + rng.normal(0, .25, (N_ITEM, 2)), -1, 1)

def sesi(rng, V, T=10, beta=3.0, gamma=1.0, sig_c=0.2, shift=True):
    """Satu sesi: riwayat T item + 1 item target. Mengembalikan utilitas stabil,
    koordinat teramati (berderau) dari riwayat, target, dan mood sebenarnya saat target."""
    u = rng.normal(0, 1, N_ITEM)
    c0 = rng.integers(0, 4); c1 = (c0 + rng.integers(1, 4)) % 4
    moods = [PUSAT[c1] if (shift and t >= T // 2) else PUSAT[c0] for t in range(T + 1)]
    def pilih(m):
        sc = beta * (u + gamma * alignment(V, m))
        p = np.exp(sc - sc.max()); p /= p.sum()
        return rng.choice(N_ITEM, p=p)
    hist = [pilih(m) for m in moods[:T]]
    target = pilih(moods[T])
    V_obs = np.clip(V[hist] + rng.normal(0, sig_c, (T, 2)), -1, 1)
    return u, hist, V_obs, target, moods[T], moods[:T]

V = katalog()
rng = np.random.default_rng(0)
u_, hist_, Vobs_, target_, m_true_, _ = sesi(rng, V)
print("riwayat:", hist_, "| target:", target_, "| mood sebenarnya saat target:", m_true_)
'''),
md("""
## 4. Eksperimen A: melacak mood dengan EWMA

Pada sesi tanpa peralihan dan dengan peralihan, ukur galat pelacakan di akhir sesi (jarak antara estimasi mood dan mood sebenarnya) untuk beberapa `a_main`.
"""),
code('''
def estimasi_mood(V_obs, a_main=0.5, theta=0.35, kappas=None):
    st, mood, flags = {}, None, []
    for k, v in enumerate(V_obs):
        st, mood, sh = update_mood(st, v, 1.0 if kappas is None else kappas[k], a_main=a_main, theta=theta)
        flags.append(sh)
    return mood, flags

def galat_pelacakan(a_main, shift, n_sesi=400, sig_c=0.4, seed=0):
    """Deteksi peralihan dimatikan (theta=5) agar efek a_main terlihat murni.
    gamma=3 dan T=14: pilihan cukup mencerminkan mood agar pelacakan bermakna."""
    rng = np.random.default_rng(seed)
    g = []
    for _ in range(n_sesi):
        _, _, V_obs, _, m_true, _ = sesi(rng, V, T=14, gamma=3.0, shift=shift, sig_c=sig_c)
        mood, _ = estimasi_mood(V_obs, a_main=a_main, theta=5.0)
        g.append(np.linalg.norm(mood - m_true))
    return float(np.mean(g))

rows = [{"a_main": a, "tanpa peralihan": galat_pelacakan(a, False), "dengan peralihan": galat_pelacakan(a, True)}
        for a in (0.1, 0.3, 0.5, 0.8)]
lacak = pd.DataFrame(rows).set_index("a_main")
print(lacak.round(3))
assert lacak.loc[0.1, "tanpa peralihan"] < lacak.loc[0.8, "tanpa peralihan"]      # stabil: EWMA lambat unggul
assert lacak.loc[0.1, "dengan peralihan"] > lacak.loc[0.3, "dengan peralihan"]    # peralihan: EWMA lambat tertinggal
'''),
md("""
**Interpretasi (simulasi).** Ada pertukaran klasik. `a_main` kecil meredam derau koordinat, tetapi lambat mengikuti peralihan. `a_main` besar cepat mengikuti peralihan, tetapi ikut terbawa derau. Pada sesi stabil, `a_main` kecil paling akurat; pada sesi dengan peralihan, `a_main = 0,1` justru jauh lebih buruk daripada 0,3 karena terlambat mengikuti. Deteksi peralihan (`theta`) di buku ada untuk menutup kelemahan itu: saat EWMA pendek dan panjang berjauhan, sistem mengikuti EWMA pendek.
"""),
code('''
def galat_dengan_deteksi(theta, shift, n_sesi=400, sig_c=0.4, seed=0):
    rng = np.random.default_rng(seed)
    g = []
    for _ in range(n_sesi):
        _, _, V_obs, _, m_true, _ = sesi(rng, V, T=14, gamma=3.0, shift=shift, sig_c=sig_c)
        mood, _ = estimasi_mood(V_obs, a_main=0.1, theta=theta)       # a_main lambat, bergantung pada deteksi
        g.append(np.linalg.norm(mood - m_true))
    return float(np.mean(g))

rows = [{"theta": th, "tanpa peralihan": galat_dengan_deteksi(th, False), "dengan peralihan": galat_dengan_deteksi(th, True)}
        for th in (0.15, 0.35, 0.6, 5.0)]
deteksi = pd.DataFrame(rows).set_index("theta")
print(deteksi.round(3))
assert deteksi.loc[0.35, "dengan peralihan"] < deteksi.loc[5.0, "dengan peralihan"]    # deteksi membantu saat ada peralihan
assert deteksi.loc[0.35, "tanpa peralihan"] > deteksi.loc[5.0, "tanpa peralihan"]      # tetapi menambah alarm palsu saat stabil
'''),
md("""
**Interpretasi.** `theta = 5` setara dengan mematikan deteksi (jarak maksimum hanya 2,83). Pada sesi dengan peralihan, `theta` kecil menurunkan galat jauh dibanding EWMA lambat saja. Harganya terlihat pada sesi tanpa peralihan: `theta` kecil memicu alarm palsu karena derau, sehingga galat naik. Nilai `theta` adalah kompromi antara kepekaan dan alarm palsu.
"""),
md("""
## 5. Eksperimen B: bobot keyakinan κ pada EWMA

Setengah item punya koordinat yang buruk (derau 1,0, κ = 0,2) dan setengahnya baik (derau 0,15, κ = 0,9). Bandingkan EWMA tanpa κ dengan EWMA berbobot κ.
"""),
code('''
def galat_kappa(pakai_kappa, n_sesi=500, seed=0):
    """gamma=3, T=14 seperti di atas."""
    rng = np.random.default_rng(seed)
    g = []
    for _ in range(n_sesi):
        u, hist, _, _, m_true, _ = sesi(rng, V, T=14, gamma=3.0, shift=False)
        buruk = rng.random(len(hist)) < 0.5
        sig = np.where(buruk, 1.0, 0.15)[:, None]
        V_obs = np.clip(V[hist] + rng.normal(0, 1, (len(hist), 2)) * sig, -1, 1)
        kap = np.where(buruk, 0.2, 0.9) if pakai_kappa else None
        mood, _ = estimasi_mood(V_obs, a_main=0.5, kappas=kap)
        g.append(np.linalg.norm(mood - m_true))
    return float(np.mean(g))

e0, e1 = galat_kappa(False), galat_kappa(True)
print(f"galat pelacakan tanpa κ: {e0:.3f} | dengan κ: {e1:.3f}")
assert e1 < e0
'''),
md("""
**Interpretasi.** Bila κ memang informatif (item ber-derau tinggi diberi κ rendah), EWMA berbobot κ melacak mood lebih baik. Jika κ tidak berkaitan dengan kualitas koordinat, manfaat ini hilang. Sama seperti gerbang konfidens di Bab 3, κ hanya berguna jika benar-benar membawa informasi.
"""),
md("""
## 6. Eksperimen C: berapa bobot λ yang pantas untuk re-ranking?

Ukuran: *hit rate@10* dari item target berikutnya, pada 800 sesi per sel. Variasikan λ dan kekuatan pengaruh mood terhadap perilaku (`gamma`). `gamma = 0` berarti mood tidak memengaruhi pilihan sama sekali.
"""),
code('''
def hit_rate(lam, gamma, sig_c=0.2, n_sesi=800, seed=1, kappa_fn=None):
    rng = np.random.default_rng(seed)
    hit = 0
    for _ in range(n_sesi):
        u, hist, V_obs, target, m_true, _ = sesi(rng, V, gamma=gamma, sig_c=sig_c)
        r = u + rng.normal(0, 0.7, N_ITEM)                  # Tahap A, buta mood
        cand = np.argsort(-r)[:50]
        mood, _ = estimasi_mood(V_obs)
        top, _, _ = rerank(cand, r[cand], V[cand], mood, K=10, lam=lam)
        hit += target in top
    return hit / n_sesi

lams = (0.0, 0.2, 0.4, 0.6, 0.8)
tab_lam = pd.DataFrame({f"gamma={g}": [hit_rate(l, g) for l in lams] for g in (0.0, 1.0, 2.0)}, index=lams)
tab_lam.index.name = "lambda"
print(tab_lam.round(3))

fig, ax = plt.subplots(figsize=(5.4, 3.3))
tab_lam.plot(ax=ax, marker="o")
ax.set(xlabel="λ (bobot keselarasan afektif)", ylabel="hit rate@10")
fig.tight_layout()

# klaim yang diuji
assert tab_lam["gamma=0.0"].iloc[-1] < tab_lam["gamma=0.0"].iloc[0]          # mood tidak relevan: re-ranking merusak
assert tab_lam["gamma=1.0"].max() > tab_lam["gamma=1.0"].iloc[0] + 0.01       # mood relevan: ada λ yang membantu
assert tab_lam["gamma=1.0"].idxmax() not in (lams[0], lams[-1])               # λ optimum bukan di ekstrem
'''),
md("""
**Interpretasi (simulasi dengan asumsi di atas).**

1. Bila mood **tidak** memengaruhi perilaku (`gamma = 0`), re-ranking tidak memberi manfaat yang berarti (selisih kecil pada λ rendah masih dalam rentang derau simulasi), dan λ besar menurunkan hit rate karena menggeser rekomendasi menjauhi preferensi stabil. Re-ranking afektif bukan perbaikan universal.
2. Bila mood berpengaruh sedang (`gamma = 1`), ada λ menengah yang memberi kenaikan nyata, dan λ yang terlalu besar mengabaikan relevansi sehingga hasil menurun lagi. Bentuk U terbalik ini yang menjadi alasan λ harus disetel pada validasi.
3. Bila mood sangat berpengaruh (`gamma = 2`), λ optimum bergeser ke nilai yang lebih besar.

Karena `gamma` pada data nyata tidak diketahui, λ **tidak boleh** ditetapkan dari intuisi. Perhatikan juga bahwa `gamma` di sini adalah asumsi simulasi: apakah pengguna nyata berperilaku demikian adalah pertanyaan empiris.
"""),
md("""
## 7. Eksperimen D: koordinat item yang tidak andal

Pada Eksperimen C koordinat item (`V`) dipakai langsung. Di sini, sebagian item mendapat koordinat yang salah (acak) dan kita bandingkan re-ranking tanpa κ dengan re-ranking yang menarik keselarasan item meragukan ke netral (`kappa`).
"""),
code('''
def hit_rate_koordinat_salah(pakai_kappa, lam=0.4, gamma=1.0, frac_salah=0.4, n_sesi=800, seed=2):
    rng_k = np.random.default_rng(99)
    salah = rng_k.random(N_ITEM) < frac_salah                       # item dengan koordinat tidak andal
    V_dipercaya = np.where(salah[:, None], rng_k.uniform(-1, 1, (N_ITEM, 2)), V)   # koordinat yang diketahui sistem
    kappa_item = np.where(salah, 0.1, 0.9)
    rng = np.random.default_rng(seed)
    hit = 0
    for _ in range(n_sesi):
        u, hist, V_obs, target, m_true, _ = sesi(rng, V, gamma=gamma)
        r = u + rng.normal(0, 0.7, N_ITEM)
        cand = np.argsort(-r)[:50]
        mood, _ = estimasi_mood(V_obs)
        top, _, _ = rerank(cand, r[cand], V_dipercaya[cand], mood, K=10, lam=lam,
                           kappa=kappa_item[cand] if pakai_kappa else None)
        hit += target in top
    return hit / n_sesi

dasar = hit_rate(0.0, 2.0)
rows = [{"porsi koordinat salah": f,
         "tanpa re-ranking": dasar,
         "re-rank tanpa κ": hit_rate_koordinat_salah(False, lam=0.6, gamma=2.0, frac_salah=f),
         "re-rank dengan κ": hit_rate_koordinat_salah(True, lam=0.6, gamma=2.0, frac_salah=f)}
        for f in (0.0, 0.4, 0.7, 0.9)]
tab_salah = pd.DataFrame(rows).set_index("porsi koordinat salah")
print(tab_salah.round(3))
assert tab_salah.loc[0.9, "re-rank dengan κ"] > tab_salah.loc[0.9, "re-rank tanpa κ"] + 0.02
assert tab_salah.loc[0.9, "re-rank tanpa κ"] < tab_salah.loc[0.9, "tanpa re-ranking"] + 0.01   # tanpa κ: manfaat lenyap
'''),
md("""
**Interpretasi (simulasi, gamma = 2, λ = 0,6).** Seiring bertambahnya porsi koordinat yang salah, manfaat re-ranking tanpa κ menyusut dan pada porsi 0,9 tidak lebih baik daripada tanpa re-ranking. Dengan κ, keselarasan item yang diragukan ditarik ke netral sehingga sebagian besar manfaat bertahan. Pada porsi kecil, selisih keduanya berada dalam rentang derau simulasi (±0,02), jadi jangan menafsirkan perbedaan kecil itu. Kualitas koordinat tetap hulu dari semuanya.
"""),
footer([
 "Mengapa `rerank` menormalkan skor Tahap A ke [0, 1] sebelum dicampur dengan keselarasan?",
 "Pada Eksperimen C, apa yang terjadi bila Tahap A bukan `u + derau 0,7`, tetapi `u + derau 0,1` (Tahap A jauh lebih akurat)? Prediksikan dulu, lalu jalankan.",
 "`target_mood` dengan `gamma > 0` mengarahkan pengguna ke mood tujuan. Kapan ini pantas dipakai, dan apa risiko etisnya (manipulasi emosi)?",
]),
]

# ------------------------------------------------------------------ 5.3 gated two-tower (torch)
P = "notebooks/bab05_emosi_sesi/03_two_tower_gerbang_personal.ipynb"
BUTUH = {P: "torch"}
NB[P] = [
header(P, "Two-tower dengan gerbang personal (PyTorch)", 5, "Kode 5.4.1",
 ["Membaca arsitektur two-tower dengan gerbang vektor antara preferensi jangka panjang dan sinyal sesi.",
  "Melatih dengan loss softmax in-batch + koreksi logQ + regularisasi entropi gerbang.",
  "Melihat apakah gerbang yang dipelajari mengikuti struktur data simulasi."],
 catatan="> **Peringatan.** Notebook ini membutuhkan PyTorch (sudah tersedia di Google Colab). Hasil asersi di bawah dirancang untuk simulasi kecil di CPU, tetapi tolok ukur kuantitatifnya dikonfirmasi lewat eksekusi otomatis di GitHub Actions, bukan dijamin lintas versi PyTorch. Kelompok pengguna 'konsisten' vs 'pengikut sesi' dalam simulasi adalah **asumsi**."),
md("## 1. Kode dari buku"),
code('''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

torch.manual_seed(0); np.random.seed(0)


class ItemTower(nn.Module):
    def __init__(self, n_items, n_feat, d=64):
        super().__init__()
        self.emb = nn.Embedding(n_items, d)
        self.mlp = nn.Sequential(nn.Linear(d + n_feat + 2, d), nn.ReLU(), nn.Linear(d, d))
    def forward(self, item_ids, feats, va):            # va: (n, 2) = v_i
        x = torch.cat([self.emb(item_ids), feats, va], dim=-1)
        return F.normalize(self.mlp(x), dim=-1)

class GatedUserTower(nn.Module):
    def __init__(self, n_users, n_ctx, d=64, eta=3.0):
        super().__init__()
        self.user = nn.Embedding(n_users, d)
        self.long = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
        self.sess = nn.Linear(d, d)
        self.gate = nn.Linear(d + n_ctx + d, d)       # Pers. (5-7), gerbang bervektor
        self.eta = eta
    def forward(self, user_ids, ctx, sess_item_vecs, sess_mask):
        e_u = self.user(user_ids)
        h_L = self.long(e_u)
        T = sess_item_vecs.size(1)                      # (B, T, d), item terbaru di akhir
        age = torch.arange(T - 1, -1, -1, device=e_u.device).float()
        w = torch.exp(-age / self.eta) * sess_mask      # bobot kebaruan (Tabel 5.3)
        w = w / w.sum(1, keepdim=True).clamp_min(1e-9)
        h_S = self.sess((w.unsqueeze(-1) * sess_item_vecs).sum(1))
        g = torch.sigmoid(self.gate(torch.cat([e_u, ctx, h_S], dim=-1)))
        h_u = g * h_L + (1 - g) * h_S                   # Pers. (5-8)
        return F.normalize(h_u, dim=-1), g

def loss_fn(h_u, h_i, log_q, g, tau=0.05, beta_g=0.01):
    """h_u, h_i: (B, d) pasangan positif sejajar; log_q: (B,) log frekuensi item."""
    logits = h_u @ h_i.T / tau - log_q.unsqueeze(0)    # Pers. (5-5) + koreksi logQ
    labels = torch.arange(h_u.size(0), device=h_u.device)
    l_rec = F.cross_entropy(logits, labels)             # Pers. (5-6)
    g_bar = g.mean(0).clamp(1e-6, 1 - 1e-6)
    omega = (g_bar * g_bar.log() + (1 - g_bar) * (1 - g_bar).log()).sum()  # Pers. (5-10)
    return l_rec + beta_g * omega
'''),
md("""
## 2. Data simulasi

Dunia kecil dengan 300 item di 6 kelompok selera (cluster) dan koordinat valence–arousal per item. Dua tipe pengguna:

- **Konsisten** (50%): item berikutnya diambil dari cluster favorit tetap, apa pun sesinya. Sinyal sesi tidak membantu.
- **Pengikut sesi** (50%): item berikutnya mengikuti cluster item terakhir di sesi. Preferensi jangka panjang tidak membantu.

Model **tidak diberi tahu** tipe pengguna. Jika arsitekturnya masuk akal, gerbang `g` (bobot jangka panjang) seharusnya lebih besar untuk pengguna konsisten.
"""),
code('''
N_ITEM, N_USER, N_CL, T, D = 300, 400, 6, 6, 32
rng = np.random.default_rng(0)
item_cl = rng.integers(0, N_CL, N_ITEM)
items_by_cl = [np.where(item_cl == c)[0] for c in range(N_CL)]
va = np.clip(rng.normal(0, .5, (N_CL, 2))[item_cl] + rng.normal(0, .1, (N_ITEM, 2)), -1, 1).astype("float32")
feats = np.eye(N_CL, dtype="float32")[item_cl]                      # fitur item: one-hot cluster
fav = rng.integers(0, N_CL, N_USER)
konsisten = rng.random(N_USER) < 0.5

def contoh(n):
    u = rng.integers(0, N_USER, n)
    sess = np.zeros((n, T), int); pos = np.zeros(n, int)
    for k in range(n):
        cl_s = rng.integers(0, N_CL, T)
        sess[k] = [rng.choice(items_by_cl[c]) for c in cl_s]
        cl_t = fav[u[k]] if konsisten[u[k]] else cl_s[-1]
        pos[k] = rng.choice(items_by_cl[cl_t])
    return u, sess, pos

u_tr, s_tr, p_tr = contoh(6000)
u_te, s_te, p_te = contoh(1000)
freq = np.bincount(p_tr, minlength=N_ITEM) + 1.0
log_q_all = torch.tensor(np.log(freq / freq.sum()), dtype=torch.float32)
print("contoh latih:", len(u_tr), "| porsi pengguna konsisten:", konsisten.mean().round(2))
'''),
md("## 3. Pelatihan"),
code('''
item_ids_all = torch.arange(N_ITEM)
F_all, VA_all = torch.tensor(feats), torch.tensor(va)

def latih(epochs=15, bs=256, beta_g=0.01, lr=3e-3, seed=0):
    torch.manual_seed(seed)
    it = ItemTower(N_ITEM, N_CL, D)
    ut = GatedUserTower(N_USER, 1, D)
    opt = torch.optim.Adam(list(it.parameters()) + list(ut.parameters()), lr=lr)
    ctx_dummy = lambda n: torch.zeros(n, 1)
    riw = []
    for ep in range(epochs):
        perm = np.random.default_rng(seed + ep).permutation(len(u_tr))
        tot = 0
        for b in range(0, len(perm), bs):
            ix = perm[b:b + bs]
            u = torch.tensor(u_tr[ix]); s = torch.tensor(s_tr[ix]); p = torch.tensor(p_tr[ix])
            sv = it(s.reshape(-1), F_all[s.reshape(-1)], VA_all[s.reshape(-1)]).reshape(len(ix), T, D)
            h_u, g = ut(u, ctx_dummy(len(ix)), sv, torch.ones(len(ix), T))
            h_i = it(p, F_all[p], VA_all[p])
            loss = loss_fn(h_u, h_i, log_q_all[p], g, beta_g=beta_g)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(ix)
        riw.append(tot / len(perm))
    return it, ut, riw

@torch.no_grad()
def evaluasi(it, ut, k=10):
    u = torch.tensor(u_te); s = torch.tensor(s_te); p = torch.tensor(p_te)
    sv = it(s.reshape(-1), F_all[s.reshape(-1)], VA_all[s.reshape(-1)]).reshape(len(u), T, D)
    h_u, g = ut(u, torch.zeros(len(u), 1), sv, torch.ones(len(u), T))
    h_all = it(item_ids_all, F_all, VA_all)
    top = (h_u @ h_all.T).topk(k, dim=1).indices
    hit = (top == p.unsqueeze(1)).any(1).float().numpy()
    return hit.mean(), g.mean(1).numpy()

it, ut, riw = latih()
hit, g_rata = evaluasi(it, ut)
g_kons = g_rata[konsisten[u_te]].mean(); g_sess = g_rata[~konsisten[u_te]].mean()
acak = 10 / N_ITEM
print(f"loss awal -> akhir: {riw[0]:.3f} -> {riw[-1]:.3f}")
print(f"hit@10 = {hit:.3f} (acak = {acak:.3f})")
print(f"rata-rata gerbang jangka panjang: konsisten {g_kons:.3f} | pengikut sesi {g_sess:.3f}")

fig, ax = plt.subplots(1, 2, figsize=(8, 3))
ax[0].plot(riw, marker="o"); ax[0].set(xlabel="epoch", ylabel="loss", title="Kurva latih")
ax[1].hist(g_rata[konsisten[u_te]], bins=20, alpha=.6, label="konsisten")
ax[1].hist(g_rata[~konsisten[u_te]], bins=20, alpha=.6, label="pengikut sesi")
ax[1].set(xlabel="rata-rata g (bobot jangka panjang)", title="Gerbang per tipe"); ax[1].legend()
fig.tight_layout()

assert riw[-1] < riw[0]
assert hit > 3 * acak
'''),
md("""
**Cara membaca hasil.** Pertama, loss turun dan hit@10 jauh di atas acak: model mempelajari struktur cluster. Kedua, bandingkan rata-rata gerbang antar tipe pengguna. Jika gerbang memang menangkap struktur data, pengguna konsisten bernilai `g` lebih tinggi. Hasil ini **tidak dijamin** pada setiap seed dan setiap versi PyTorch; periksa angka yang keluar, dan lihat ablasi di bawah sebelum menarik kesimpulan.
"""),
md("""
## 4. Ablasi: apakah gerbang membantu?

Bandingkan dengan dua pembanding degenerat yang dibuat dengan membekukan gerbang: hanya jangka panjang (`g = 1`) dan hanya sesi (`g = 0`), lewat subclass sederhana. Ini menjawab pertanyaan inti: pada data campuran, apakah gerbang yang dipelajari mengalahkan kedua ekstrem?
"""),
code('''
class GerbangTetap(GatedUserTower):
    def __init__(self, *a, nilai=1.0, **k):
        super().__init__(*a, **k); self.nilai = nilai
    def forward(self, user_ids, ctx, sess_item_vecs, sess_mask):
        e_u = self.user(user_ids); h_L = self.long(e_u)
        T_ = sess_item_vecs.size(1)
        age = torch.arange(T_ - 1, -1, -1).float()
        w = torch.exp(-age / self.eta) * sess_mask
        w = w / w.sum(1, keepdim=True).clamp_min(1e-9)
        h_S = self.sess((w.unsqueeze(-1) * sess_item_vecs).sum(1))
        g = torch.full_like(h_L, self.nilai)
        return F.normalize(g * h_L + (1 - g) * h_S, dim=-1), g

def latih_ut(kelas, seed=0, epochs=15, **kw):
    torch.manual_seed(seed)
    it_ = ItemTower(N_ITEM, N_CL, D); ut_ = kelas(N_USER, 1, D, **kw)
    opt = torch.optim.Adam(list(it_.parameters()) + list(ut_.parameters()), lr=3e-3)
    for ep in range(epochs):
        perm = np.random.default_rng(seed + ep).permutation(len(u_tr))
        for b in range(0, len(perm), 256):
            ix = perm[b:b + 256]
            u = torch.tensor(u_tr[ix]); s = torch.tensor(s_tr[ix]); p = torch.tensor(p_tr[ix])
            sv = it_(s.reshape(-1), F_all[s.reshape(-1)], VA_all[s.reshape(-1)]).reshape(len(ix), T, D)
            h_u, g = ut_(u, torch.zeros(len(ix), 1), sv, torch.ones(len(ix), T))
            loss = loss_fn(h_u, it_(p, F_all[p], VA_all[p]), log_q_all[p], g, beta_g=0.0)
            opt.zero_grad(); loss.backward(); opt.step()
    return it_, ut_

hasil = {}
for nama, kelas, kw in (("gerbang dipelajari", GatedUserTower, {}),
                        ("hanya jangka panjang (g=1)", GerbangTetap, {"nilai": 1.0}),
                        ("hanya sesi (g=0)", GerbangTetap, {"nilai": 0.0})):
    a, b = latih_ut(kelas, **kw)
    hasil[nama] = evaluasi(a, b)[0]
for k, v in hasil.items():
    print(f"{k:30s} hit@10 = {v:.3f}")
assert hasil["gerbang dipelajari"] > hasil["hanya jangka panjang (g=1)"] - 0.02
'''),
md("""
**Interpretasi (perkiraan; periksa terhadap keluaran Anda).** Pada data campuran, kedua ekstrem masing-masing gagal pada separuh pengguna, sehingga gerbang yang dipelajari seharusnya setidaknya sebanding dengan yang terbaik dari keduanya dan idealnya lebih tinggi. Asersi di atas sengaja longgar (toleransi 0,02) karena simulasi kecil dan satu seed. Jika gerbang tidak unggul pada eksekusi Anda, itu informasi yang sah: tambah epoch, ubah `eta`, atau jalankan beberapa seed sebelum menyimpulkan.
"""),
footer([
 "Mengapa koreksi logQ perlu pada negatif in-batch? Apa yang terjadi bila item populer terlalu sering menjadi negatif?",
 "Setel `beta_g` ke 0, 0,01, dan 1. Bagaimana distribusi gerbang berubah? Apakah entropi tinggi atau rendah yang diharapkan?",
 "Tambahkan konteks nyata (misalnya jam hari) ke `ctx` yang sekarang nol. Perubahan apa yang Anda harapkan pada gerbang?",
]),
]
