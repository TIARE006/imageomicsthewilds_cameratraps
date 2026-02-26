from pathlib import Path
import csv, re

ROOT = Path("thewilds_cameratraps_full")     # 全量目录
OUT  = Path("index.csv")

CAM_RE = re.compile(r"^TW\d+-CT\d+$")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

rows = []
for p in ROOT.rglob("*"):
    if p.suffix not in IMG_EXTS:
        continue

    parts = p.parts
    site_cam = ""
    session = ""

    # 在路径中找 TWxx-CTxx 的位置
    for i, part in enumerate(parts):
        if CAM_RE.match(part):
            site_cam = part
            # session 就是 camera 目录的下一层（如果存在）
            if i + 1 < len(parts):
                session = parts[i + 1]
            break

    rows.append((str(p), site_cam, session))

print("found images:", len(rows))

with OUT.open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["path", "site_cam", "session"])
    w.writerows(rows)

print("wrote:", OUT.resolve())