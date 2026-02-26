import re
from pathlib import Path
import pandas as pd
from PIL import Image, ExifTags

IDX = "index.csv"
OUT = "events.csv"

# 适配 TW01/TW03：NSCF0002_250630121802_0022.JPG 里的 YYMMDDHHMMSS
RE = re.compile(r"_(\d{12})_")

# EXIF tag id 映射
TAGS = ExifTags.TAGS
NAME2ID = {v: k for k, v in TAGS.items()}
DTO_ID = NAME2ID.get("DateTimeOriginal", None)
DT_ID  = NAME2ID.get("DateTime", None)

def ts_from_filename(name: str):
    m = RE.search(name)
    if not m:
        return None
    s = m.group(1)  # YYMMDDHHMMSS
    year = 2000 + int(s[0:2])
    mo   = int(s[2:4]); d  = int(s[4:6])
    hh   = int(s[6:8]); mm = int(s[8:10]); ss = int(s[10:12])
    return pd.Timestamp(year=year, month=mo, day=d, hour=hh, minute=mm, second=ss)

def ts_from_exif(path: Path):
    try:
        img = Image.open(path)
        exif = img.getexif()
        if not exif:
            return None
        val = None
        if DTO_ID is not None and DTO_ID in exif:
            val = exif.get(DTO_ID)
        elif DT_ID is not None and DT_ID in exif:
            val = exif.get(DT_ID)
        if val is None:
            return None
        # 常见格式 "YYYY:MM:DD HH:MM:SS"
        return pd.to_datetime(str(val), format="%Y:%m:%d %H:%M:%S", errors="coerce")
    except Exception:
        return None

df = pd.read_csv(IDX)

rows = []
missing = 0
exif_used = 0
for _, r in df.iterrows():
    p = Path(r["path"])
    ts = ts_from_filename(p.name)
    if ts is None:
        ts = ts_from_exif(p)
        if ts is not None and not pd.isna(ts):
            exif_used += 1
    if ts is None or pd.isna(ts):
        missing += 1
        continue
    rows.append({"timestamp": ts, "site_cam": r["site_cam"], "session": r["session"], "path": r["path"]})

ev = pd.DataFrame(rows).sort_values(["site_cam", "timestamp"]).reset_index(drop=True)
ev.to_csv(OUT, index=False)
print(f"wrote {OUT} rows={len(ev)} missing_ts={missing} exif_used={exif_used}")
print(ev.groupby('site_cam').size())