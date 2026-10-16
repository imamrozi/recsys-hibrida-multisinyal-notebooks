"""Pembangun dan pelari notebook tanpa ketergantungan Jupyter.

build(): menulis berkas .ipynb (nbformat 4) dari daftar sel.
run():   mengeksekusi sel kode berurutan dalam satu namespace, menangkap stdout,
         nilai ekspresi terakhir, dan gambar matplotlib sebagai keluaran sel.
Berkas keluaran memakai format standar sehingga dapat dibuka di Jupyter/Colab
dan dieksekusi ulang dengan `jupyter nbconvert --execute` (dipakai di CI).
"""
import ast, base64, contextlib, io, json, textwrap, traceback

def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(text).strip("\n")}

def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": textwrap.dedent(text).strip("\n")}

def build(path, cells):
    nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 4,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}}}
    for c in nb["cells"]:
        c["source"] = c["source"].splitlines(keepends=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
        f.write("\n")

def run(path, verbose=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nb = json.load(open(path, encoding="utf-8"))
    ns = {"__name__": "__main__"}
    count = 0
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        count += 1
        outs, buf = [], io.StringIO()
        tree = ast.parse(src)
        last = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last = ast.Expression(tree.body.pop().value)
        try:
            with contextlib.redirect_stdout(buf):
                exec(compile(tree, path, "exec"), ns)
                val = eval(compile(last, path, "eval"), ns) if last is not None else None
        except Exception:
            raise RuntimeError("Galat pada sel kode #%d di %s:\n%s\n--- sel ---\n%s" % (count, path, traceback.format_exc(), src))
        if buf.getvalue():
            outs.append({"output_type": "stream", "name": "stdout", "text": buf.getvalue().splitlines(keepends=True)})
        if val is not None:
            data = {"text/plain": repr(val).splitlines(keepends=True)}
            if hasattr(val, "_repr_html_"):
                try: data["text/html"] = val._repr_html_().splitlines(keepends=True)
                except Exception: pass
            outs.append({"output_type": "execute_result", "metadata": {}, "execution_count": count, "data": data})
        for n in plt.get_fignums():
            fig = plt.figure(n); b = io.BytesIO()
            fig.savefig(b, format="png", dpi=110, bbox_inches="tight")
            outs.append({"output_type": "display_data", "metadata": {},
                         "data": {"image/png": base64.b64encode(b.getvalue()).decode(), "text/plain": ["<Figure>"]}})
        plt.close("all")
        cell["outputs"], cell["execution_count"] = outs, count
        if verbose:
            print("".join(buf.getvalue().splitlines(keepends=True)))
    json.dump(nb, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return nb
