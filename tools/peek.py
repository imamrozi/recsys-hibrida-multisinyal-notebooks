"""Alat bantu pengembangan: jalankan sel sampai sebelum sel gagal dan cetak keluaran."""
import sys, json, io, contextlib
sys.path.insert(0, "tools")
import matplotlib; matplotlib.use("Agg")
nb = json.load(open(sys.argv[1]))
ns = {"__name__": "__main__"}
for c in nb["cells"]:
    if c["cell_type"] != "code": continue
    src = "".join(c["source"])
    try:
        exec(compile(src, "cell", "exec"), ns)
    except Exception as e:
        print("GAGAL:", type(e).__name__, e); break
