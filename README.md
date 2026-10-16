# Kode pendamping: Sistem Rekomendasi Hibrida Multi-Sinyal

Notebook interaktif untuk buku *Sistem Rekomendasi Hibrida Multi-Sinyal: Rating, Sentimen, dan Emosi*.
Setiap notebook dapat dijalankan **langsung di Google Colab** (klik badge di dalam notebook, tanpa instalasi)
dan berfokus pada **simulasi dan insight**: Anda mengubah parameter, melihat hasilnya, lalu membandingkannya dengan penjelasan di buku.

## Penting dibaca dulu

- **Semua data adalah simulasi atau data mainan.** Angka yang keluar menunjukkan *perilaku metode*, bukan hasil empiris pada dataset nyata, dan **tidak boleh dikutip sebagai bukti**. Angka di buku berasal dari eksperimen pada data nyata yang dijelaskan di Bab 1 dan Bab 3–6.
- Kode bersifat **ilustratif dan belum production-ready** (tanpa validasi masukan, tanpa uji beban).
- Setiap klaim di sel penjelasan didukung oleh `assert` di sel kode di atasnya. Jika sebuah asersi gagal setelah Anda mengubah parameter, itu informasi, bukan kerusakan.
- Leksikon emosi di Bab 5 adalah leksikon **mainan**; ganti dengan Warriner/NRC asli yang Anda unduh sesuai lisensinya.

## Daftar notebook

| Bab | Notebook | Yang dieksplorasi |
|---|---|---|
| 1 | `bab01_lanskap/01_dataset_kejarangan_longtail` | k-core, kejarangan, long-tail, baseline popularitas |
| 2 | `bab02_pendekatan_dasar/01_content_based_tfidf` | Content-based dengan TF-IDF |
| 2 | `bab02_pendekatan_dasar/02_collaborative_filtering_knn` | kNN: cosine mentah vs terpusat |
| 2 | `bab02_pendekatan_dasar/03_matrix_factorization_sgd` | Faktorisasi matriks: dimensi, regularisasi, epoch |
| 2 | `bab02_pendekatan_dasar/04_hibrida_sederhana` | Hibrida berbobot vs switching |
| 3 | `bab03_sentiment_aware/01_absa_dan_representasi_aspek` | ABSA mainan dan empat representasi aspek |
| 3 | `bab03_sentiment_aware/02_konfidens_gerbang_mf_aspek` | Gerbang konfidens dan MF-aspek |
| 3 | `bab03_sentiment_aware/03_stacking_dan_uji_berpasangan` | Stacking, kebocoran OOF, uji berpasangan |
| 4 | `bab04_hybrid_cf/01_als_dengan_bias` | ALS dengan bias |
| 4 | `bab04_hybrid_cf/02_fitur_statistik_out_of_fold` | Fitur statistik out-of-fold dan kebocoran |
| 5 | `bab05_emosi_sesi/01_koordinat_valence_arousal` | Koordinat VA dari tag/teks dan keyakinan κ |
| 5 | `bab05_emosi_sesi/02_mood_sesi_dan_reranking` | Mood sesi (EWMA) dan re-ranking afektif |
| 5 | `bab05_emosi_sesi/03_two_tower_gerbang_personal` | Two-tower dengan gerbang personal (PyTorch) |
| 6 | `bab06_evaluasi/01_metrik_ranking_dan_jebakan_evaluasi` | Metrik ranking dan evaluasi tersampel |
| 6 | `bab06_evaluasi/02_metrik_beyond_accuracy_dan_afektif` | Coverage, Gini, novelty, ILD, EAS, QHR, KL |
| 6 | `bab06_evaluasi/03_uji_signifikansi_dan_koreksi_holm` | Wilcoxon, bootstrap, Holm, besar efek |

Bab 7 (penutup) tidak memiliki kode.

## Menjalankan

**Colab (disarankan):** buka notebook di GitHub, klik badge *Buka di Colab* di bagian atas.

**Lokal:**

```bash
git clone https://github.com/imamrozi/recsys-hibrida-multisinyal-notebooks
cd recsys-hibrida-multisinyal-notebooks
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Semua notebook berjalan di CPU. Sebagian besar selesai dalam hitungan detik; beberapa (Bab 3 nomor 2 dan 3, Bab 5 nomor 2) memerlukan sekitar 20–40 detik.

## Status verifikasi

| Notebook | Dieksekusi dan asersi diuji |
|---|---|
| Semua kecuali satu | Ya, di lingkungan pengembangan dan oleh GitHub Actions |
| `bab05_emosi_sesi/03_two_tower_gerbang_personal` | Diuji oleh GitHub Actions saja (membutuhkan PyTorch). Periksa status CI. Hasil kuantitatif dapat bergeser antar versi PyTorch. |

## Struktur

- `notebooks/`: notebook per bab (satu notebook per konsep).
- `tools/`: skrip yang membangun notebook dari sumber (`tools/src/bab*.py`). Bukan bagian yang perlu dijalankan pembaca.
- `.github/workflows/notebooks.yml`: eksekusi otomatis semua notebook.

## Lisensi dan sitasi

- Kode: **MIT** (`LICENSE`).
- Teks penjelasan: **CC BY 4.0** (`LICENSE-TEKS.md`).
- Sitasi: lihat `CITATION.cff` (tombol *Cite this repository* di GitHub).
