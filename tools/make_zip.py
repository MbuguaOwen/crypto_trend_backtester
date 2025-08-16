import os, zipfile, datetime

def make_zip(base_dir: str):
    date_str = datetime.datetime.utcnow().strftime("%Y%m%d")
    name = f"tsmom-parity-backtest-{date_str}.zip"
    out_path = os.path.join(os.path.dirname(base_dir), name)
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(base_dir):
            for f in files:
                p = os.path.join(root, f)
                arc = os.path.relpath(p, os.path.dirname(base_dir))
                z.write(p, arc)
    return out_path

if __name__ == "__main__":
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    z = make_zip(base)
    print(z)
