r"""Download REHAB26-ViewAngles (Zenodo record 23018128, CC BY-NC 4.0) into the server dataset folder.
Resumable (HTTP Range), verifies the Zenodo-published MD5 for every file, writes download_manifest.json."""
import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path

DEST = Path(r"rehabal_work\datasets\rehab26_viewangles")
BASE = "https://zenodo.org/api/records/23018128/files/{}/content"
FILES = {  # key: (size, md5) as listed by the Zenodo API on 2026-10-13
    "Error_types.csv": (934, "2f89da48c2c26b4868a59deaa4fb3ed9"),
    "Segmentation.txt": (2019, "502475f709390d96b599da65fa45ffd3"),
    "joint_names.txt": (511, "2d6999cedf3775d3e9dded2e87b8d035"),
    "Segmentation.csv": (83334, "3d1cf6b552f60ef03eb8f5db977c139f"),
    "jointSequences-mediapipe.zip": (686028292, "303ba8e1c236ee52dac7c0b8ac23789f"),
    "videos.zip": (8433934055, "f21dc596f4ef84e21060a43a65c2f30b"),
}


def md5_file(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def fetch(name, size):
    target = DEST / name
    part = DEST / (name + ".part")
    if target.exists() and target.stat().st_size == size:
        return "exists"
    attempt = 0
    while True:
        attempt += 1
        have = part.stat().st_size if part.exists() else 0
        if have >= size:
            break
        req = urllib.request.Request(BASE.format(name), headers={"User-Agent": "rehabal-eval/1.0"})
        if have:
            req.add_header("Range", "bytes=%d-" % have)
        try:
            with urllib.request.urlopen(req, timeout=120) as r, open(part, "ab" if have else "wb") as f:
                if have and r.status != 206:
                    f.seek(0); f.truncate(); have = 0
                t0 = time.time(); got = 0
                while True:
                    b = r.read(8 * 1024 * 1024)
                    if not b:
                        break
                    f.write(b); got += len(b)
                    if got % (256 * 1024 * 1024) < 8 * 1024 * 1024:
                        print(json.dumps({"file": name, "bytes": have + got, "of": size,
                                          "mbps": round(got / 1e6 / max(time.time() - t0, 1e-6), 1)}), flush=True)
        except Exception as e:  # noqa: BLE001
            print(json.dumps({"file": name, "retry": attempt, "error": repr(e)}), flush=True)
            if attempt > 30:
                raise
            time.sleep(min(60, 5 * attempt))
            continue
        if part.stat().st_size >= size:
            break
    os.replace(part, target)
    return "downloaded"


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    manifest = {"record": "https://zenodo.org/records/23018128", "license": "CC BY-NC 4.0", "files": {}}
    for name, (size, md5) in FILES.items():
        status = fetch(name, size)
        digest = md5_file(DEST / name)
        ok = digest == md5
        manifest["files"][name] = {"size": size, "md5_expected": md5, "md5_actual": digest, "match": ok, "status": status}
        print(json.dumps({"file": name, "status": status, "md5_ok": ok}), flush=True)
        if not ok:
            raise SystemExit("MD5 mismatch " + name)
    manifest["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(DEST / "download_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print("ALL_DOWNLOADS_VERIFIED", flush=True)


if __name__ == "__main__":
    main()
