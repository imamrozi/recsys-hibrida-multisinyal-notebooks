"""Membuat halaman pengalihan pendek untuk GitHub Pages (docs/).
   docs/k/<nomor kode>/index.html  -> Colab notebook yang bersangkutan.
   Dijalankan: python tools/build_pages.py"""
import os, sys, html
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import notebooks_map as m

DOCS = os.path.join(ROOT, "docs")
GH = "https://github.com/%s/blob/main/notebooks/%s.ipynb"

def halaman(num, nb):
    judul, isi = m.NB[nb]
    colab, gh = m.colab_url(nb), GH % (m.REPO, nb)
    return f"""<!doctype html>
<html lang="id"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Kode {num}: {html.escape(judul)}</title>
<meta http-equiv="refresh" content="0; url={colab}">
<style>body{{font:16px/1.5 system-ui,sans-serif;max-width:34rem;margin:3rem auto;padding:0 1rem;color:#1f2937}}a{{color:#2a79d1}}</style>
</head><body>
<h1>Kode {num}</h1>
<p><strong>{html.escape(judul)}</strong>: {html.escape(isi)}</p>
<p>Mengalihkan ke Google Colab. Jika tidak berpindah otomatis, <a href="{colab}">buka notebook di Colab</a>
atau <a href="{gh}">lihat berkasnya di GitHub</a> (tanpa akun Google).</p>
<script>location.replace({colab!r});</script>
</body></html>
"""

def indeks():
    rev = {}
    for k, nb in m.KODE.items(): rev.setdefault(nb, []).append(k)
    baris = "".join(
        f"<tr><td>{', '.join(f'<a href=\"k/{k}/\">{k}</a>' for k in rev.get(nb, []))}</td><td>{html.escape(j)}</td><td>{html.escape(i)}</td></tr>"
        for nb, (j, i) in m.NB.items())
    return f"""<!doctype html>
<html lang="id"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Kode pendamping: Sistem Rekomendasi Hibrida Multi-Sinyal</title>
<style>body{{font:16px/1.5 system-ui,sans-serif;max-width:48rem;margin:2.5rem auto;padding:0 1rem;color:#1f2937}}
table{{border-collapse:collapse;width:100%}}td,th{{border-bottom:1px solid #d5dceb;padding:.4rem .5rem;text-align:left;vertical-align:top}}a{{color:#2a79d1}}</style>
</head><body>
<h1>Kode pendamping</h1>
<p><em>Sistem Rekomendasi Hibrida Multi-Sinyal: Rating, Sentimen, dan Emosi.</em> Pilih nomor kode seperti tercetak di buku untuk membuka notebook-nya di Google Colab.</p>
<table><tr><th>Kode</th><th>Notebook</th><th>Yang dieksplorasi</th></tr>{baris}</table>
<p>Repositori: <a href="{m.REPO_URL}">{m.REPO_URL.replace('https://','')}</a>. Semua data adalah simulasi; kode ilustratif dan belum siap produksi.</p>
</body></html>
"""

if __name__ == "__main__":
    os.makedirs(DOCS, exist_ok=True)
    open(os.path.join(DOCS, "index.html"), "w", encoding="utf-8").write(indeks())
    open(os.path.join(DOCS, ".nojekyll"), "w").write("")
    for num, nb in m.KODE.items():
        d = os.path.join(DOCS, "k", num); os.makedirs(d, exist_ok=True)
        open(os.path.join(d, "index.html"), "w", encoding="utf-8").write(halaman(num, nb))
    print("halaman:", len(m.KODE) + 1)
