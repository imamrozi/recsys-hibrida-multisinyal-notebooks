REPO = "imamrozi/recsys-hibrida-multisinyal-notebooks"
BUKU = "Sistem Rekomendasi Hibrida Multi-Sinyal: Rating, Sentimen, dan Emosi"

def header(path, judul, bab, kode, tujuan, catatan=None):
    """Sel markdown pembuka seragam: judul, badge Colab, tujuan belajar, peringatan."""
    from nbkit import md
    badge = ("[![Buka di Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
             "(https://colab.research.google.com/github/%s/blob/main/%s)" % (REPO, path))
    t = "\n".join("- " + x for x in tujuan)
    extra = ("\n\n" + catatan) if catatan else ""
    return md(f"""
# {judul}

{badge}

**Buku pendamping:** *{BUKU}* · Bab {bab} · {kode}

**Tujuan belajar**

{t}

> **Catatan.** Semua data di notebook ini **simulasi** dan dibuat agar mudah dijalankan (CPU, hitungan detik). Angka yang keluar menunjukkan *perilaku metode*, bukan hasil empiris pada dataset nyata, dan tidak boleh dikutip sebagai bukti. Kode bersifat ilustratif dan **belum production-ready** (tanpa validasi masukan, tanpa uji beban).{extra}
""")

def footer(pertanyaan):
    from nbkit import md
    q = "\n".join(f"{i}. {x}" for i, x in enumerate(pertanyaan, 1))
    return md(f"## Pertanyaan refleksi\n\n{q}")
