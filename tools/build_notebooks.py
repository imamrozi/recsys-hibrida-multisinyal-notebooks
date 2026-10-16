"""Membangun semua notebook dari tools/src/bab*.py, lalu (opsional) mengeksekusinya.
Pemakaian:  python tools/build_notebooks.py [--run] [--only bab01]
"""
import importlib, os, sys, glob, time
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, os.path.join(HERE, "src")]
import nbkit

def _ada(mod):
    try:
        __import__(mod); return True
    except ImportError:
        return False

def main():
    run = "--run" in sys.argv
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
    for f in sorted(glob.glob(os.path.join(HERE, "src", "bab*.py"))):
        name = os.path.basename(f)[:-3]
        if only and name != only:
            continue
        mod = importlib.import_module(name)
        for rel, cells in mod.NB.items():
            path = os.path.join(ROOT, rel)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            nbkit.build(path, cells)
            butuh = getattr(mod, 'BUTUH', {}).get(rel)
            if run and butuh and not _ada(butuh):
                print('LEWAT %-69s (butuh %s; diverifikasi oleh CI)' % (rel, butuh))
            elif run:
                t = time.time()
                nbkit.run(path)
                print("OK  %-70s %.1fs" % (rel, time.time() - t))
            else:
                print("tulis", rel)

if __name__ == "__main__":
    main()
