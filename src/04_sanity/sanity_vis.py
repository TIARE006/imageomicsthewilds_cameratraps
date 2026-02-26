from ultralytics import YOLO
import pandas as pd
from pathlib import Path
import shutil

CSV = "baseline_predictions_full.csv"
CONF = 0.10
IMGSZ = 1280

df = pd.read_csv(CSV)
model = YOLO("yolov8n.pt")

out = Path("sanity_vis")
out.mkdir(exist_ok=True)

pick = []
for cam, g in df[df.is_animal == 1].groupby("site_cam"):
    pick.extend(g.sample(n=min(4, len(g)), random_state=0).to_dict("records"))

for r in pick:
    img_path = r["path"]
    cam = r["site_cam"]

    res = model.predict(img_path, conf=CONF, imgsz=IMGSZ, save=True, verbose=False)
    saved = Path(res[0].save_dir) / Path(img_path).name  # runs/detect/.../xxx.jpg

    if saved.exists():
        shutil.copy(saved, out / f"{cam}__{Path(img_path).name}")

print("saved vis to", out.resolve())