"""Cetak keluaran teks sebuah notebook yang sudah dieksekusi (untuk pemeriksaan)."""
import json, sys
nb = json.load(open(sys.argv[1]))
for c in nb["cells"]:
    if c["cell_type"] != "code": continue
    for o in c["outputs"]:
        if o["output_type"] == "stream": print("".join(o["text"]), end="")
        elif o["output_type"] == "execute_result": print("".join(o["data"]["text/plain"]))
        elif o["output_type"] == "display_data": print("[gambar]")
