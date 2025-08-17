import os, zipfile, sys, pathlib

def make_zip(src_dir: str, out_path: str):
    src_dir = os.path.abspath(src_dir)
    with zipfile.ZipFile(out_path, 'w', zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(src_dir):
            for f in files:
                abspath = os.path.join(root, f)
                rel = os.path.relpath(abspath, src_dir)
                z.write(abspath, rel)

if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "."
    out = sys.argv[2] if len(sys.argv) > 2 else "tsmom-parity-backtest.zip"
    make_zip(src, out)
    print("Wrote", out)
