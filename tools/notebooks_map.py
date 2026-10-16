REPO = "imamrozi/recsys-hibrida-multisinyal-notebooks"
REPO_URL = "https://github.com/" + REPO
PAGES_BASE = "https://imamrozi.github.io/recsys-hibrida-multisinyal-notebooks/k/"
def short_url(num): return PAGES_BASE + num + "/"
def colab_url(nb):   # nb: 'bab05_emosi_sesi/02_mood_sesi_dan_reranking'
    return "https://colab.research.google.com/github/%s/blob/main/notebooks/%s.ipynb" % (REPO, nb)

NB = {  # path -> (judul, isi)
 'bab01_lanskap/01_dataset_kejarangan_longtail': ('Dataset, kejarangan, dan long-tail', 'k-core, kejarangan, kurva long-tail, dan baseline popularitas.'),
 'bab02_pendekatan_dasar/01_content_based_tfidf': ('Content-based dengan TF-IDF', 'profil pengguna dan skor kemiripan dengan representasi TF-IDF.'),
 'bab02_pendekatan_dasar/02_collaborative_filtering_knn': ('Collaborative filtering kNN', 'kosinus mentah versus terpusat pada kejarangan yang berbeda.'),
 'bab02_pendekatan_dasar/03_matrix_factorization_sgd': ('Faktorisasi matriks dengan SGD', 'pengaruh dimensi laten, regularisasi, dan jumlah epoch.'),
 'bab02_pendekatan_dasar/04_hibrida_sederhana': ('Hibrida sederhana', 'hibrida berbobot versus switching di bawah model galat yang diasumsikan.'),
 'bab03_sentiment_aware/01_absa_dan_representasi_aspek': ('ABSA dan representasi aspek', 'empat cara merepresentasikan keluaran ABSA, dan runtuhnya kontras polaritas.'),
 'bab03_sentiment_aware/02_konfidens_gerbang_mf_aspek': ('Konfidens, gerbang, dan MF-aspek', 'kapan gerbang konfidens membantu, pemulihan bobot aspek, dan pemeriksaan gradien.'),
 'bab03_sentiment_aware/03_stacking_dan_uji_berpasangan': ('Stacking dan uji berpasangan', 'kebocoran out-of-fold, kedalaman pohon, bootstrap, Wilcoxon, dan Fisher.'),
 'bab04_hybrid_cf/01_als_dengan_bias': ('ALS dengan bias', 'kontribusi tiap komponen, jumlah iterasi, dan kisi dimensi × regularisasi.'),
 'bab04_hybrid_cf/02_fitur_statistik_out_of_fold': ('Fitur statistik out-of-fold', 'kebocoran pada fitur rata-rata, penghalusan, dan hasil per jumlah rating pengguna.'),
 'bab05_emosi_sesi/01_koordinat_valence_arousal': ('Koordinat valence–arousal', 'koordinat dari tag dan teks, keyakinan κ, dan penggabungan sumber.'),
 'bab05_emosi_sesi/02_mood_sesi_dan_reranking': ('Mood sesi dan re-ranking', 'pelacakan EWMA, deteksi peralihan, bobot λ, dan koordinat item yang tak andal.'),
 'bab05_emosi_sesi/03_two_tower_gerbang_personal': ('Two-tower dengan gerbang personal', 'PyTorch: pelatihan, diagnostik gerbang, dan ablasi gerbang.'),
 'bab06_evaluasi/01_metrik_ranking_dan_jebakan_evaluasi': ('Metrik ranking dan jebakan evaluasi', 'metrik ketepatan daftar, pengaruh K, dan evaluasi dengan negatif tersampel.'),
 'bab06_evaluasi/02_metrik_beyond_accuracy_dan_afektif': ('Metrik di luar akurasi dan afektif', 'coverage, Gini, novelty, ILD, EAS, QHR, KL afektif, dan kurva pertukaran.'),
 'bab06_evaluasi/03_uji_signifikansi_dan_koreksi_holm': ('Uji signifikansi dan koreksi Holm', 'p versus besar efek, perbandingan jamak, dan daya uji.'),
}
# nomor kode di buku -> notebook
KODE = {
 '1.5.1':'bab01_lanskap/01_dataset_kejarangan_longtail','1.5.2':'bab01_lanskap/01_dataset_kejarangan_longtail',
 '2.1.1':'bab02_pendekatan_dasar/01_content_based_tfidf','2.2.1':'bab02_pendekatan_dasar/02_collaborative_filtering_knn',
 '2.2.2':'bab02_pendekatan_dasar/03_matrix_factorization_sgd','2.3.1':'bab02_pendekatan_dasar/04_hibrida_sederhana',
 '3.2.1':'bab03_sentiment_aware/01_absa_dan_representasi_aspek','3.3.1':'bab03_sentiment_aware/01_absa_dan_representasi_aspek',
 '3.4.1':'bab03_sentiment_aware/02_konfidens_gerbang_mf_aspek','3.5.1':'bab03_sentiment_aware/02_konfidens_gerbang_mf_aspek',
 '3.6.1':'bab03_sentiment_aware/03_stacking_dan_uji_berpasangan','3.8.1':'bab03_sentiment_aware/03_stacking_dan_uji_berpasangan',
 '4.2.1':'bab04_hybrid_cf/01_als_dengan_bias','4.4.1':'bab04_hybrid_cf/02_fitur_statistik_out_of_fold',
 '5.2.1':'bab05_emosi_sesi/01_koordinat_valence_arousal','5.4.1':'bab05_emosi_sesi/03_two_tower_gerbang_personal',
 '5.5.1':'bab05_emosi_sesi/02_mood_sesi_dan_reranking','5.5.2':'bab05_emosi_sesi/02_mood_sesi_dan_reranking',
 '6.3.1':'bab06_evaluasi/01_metrik_ranking_dan_jebakan_evaluasi','6.5.1':'bab06_evaluasi/02_metrik_beyond_accuracy_dan_afektif',
 '6.6.1':'bab06_evaluasi/03_uji_signifikansi_dan_koreksi_holm',
}
