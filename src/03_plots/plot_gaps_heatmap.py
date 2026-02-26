import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 读 1-min counts，然后汇总到 1h
df = pd.read_csv("counts_over_time.csv", parse_dates=["tbin"])
df["hour"] = df["tbin"].dt.floor("h")
h = df.groupby(["site_cam", "hour"])["count"].sum().reset_index()

# 统一时间范围：用所有 camera 的全局 min/max（也可以改成你指定的开始/结束）
t0 = h["hour"].min().floor("h")
t1 = h["hour"].max().ceil("h")
full_hours = pd.date_range(t0, t1, freq="h")

cams = sorted(h["site_cam"].unique())
outdir = Path("gap_outputs")
outdir.mkdir(exist_ok=True)

# 产出一个“补齐后的全表”（后面博士要啥都能从这张表做）
rows = []
for cam in cams:
    g = h[h["site_cam"] == cam].set_index("hour").reindex(full_hours)
    g["count"] = g["count"].fillna(0).astype(int)
    g["site_cam"] = cam
    g = g.reset_index().rename(columns={"index": "hour"})
    g["received"] = (g["count"] > 0).astype(int)
    rows.append(g)

full = pd.concat(rows, ignore_index=True)
full.to_csv(outdir / "counts_over_time_1h_fullgrid.csv", index=False)

# 计算 gaps：连续 received==0 的小时段（每段给 start/end/duration）
gap_rows = []
for cam in cams:
    g = full[full["site_cam"] == cam].sort_values("hour").reset_index(drop=True)
    z = (g["received"] == 0).to_numpy()
    if z.sum() == 0:
        continue
    # 找连续0段
    starts = np.where((z == 1) & np.concatenate(([True], z[:-1] == 0)))[0]
    ends   = np.where((z == 1) & np.concatenate((z[1:] == 0, [True])))[0]
    for s, e in zip(starts, ends):
        gap_rows.append({
            "site_cam": cam,
            "start_hour": g.loc[s, "hour"],
            "end_hour": g.loc[e, "hour"],
            "duration_hours": int(e - s + 1),
        })

gaps = pd.DataFrame(gap_rows)
gaps.to_csv(outdir / "gaps_1h_fullrange.csv", index=False)

# 画 heatmap：日期 x 小时（颜色=每小时图片数；同时可看缺失）
# 为了可读：每个 cam 单独一张图
for cam in cams:
    g = full[full["site_cam"] == cam].copy()
    g["date"] = g["hour"].dt.date
    g["hod"]  = g["hour"].dt.hour  # hour-of-day

    pivot = g.pivot_table(index="hod", columns="date", values="count", aggfunc="sum", fill_value=0)
    pivot = pivot.reindex(range(24), fill_value=0)  # 0..23

    plt.figure()
    plt.imshow(pivot.values, aspect="auto")  # 不指定颜色
    plt.yticks(range(24), range(24))
    plt.xticks(range(pivot.shape[1]), [str(d) for d in pivot.columns], rotation=45, ha="right")
    plt.xlabel("Date")
    plt.ylabel("Hour of day")
    plt.title(f"{cam}: counts heatmap (1h bins, full-range)")
    plt.tight_layout()
    plt.savefig(outdir / f"heatmap_1h_{cam}.png", dpi=200)
    plt.close()

print("wrote:", outdir / "counts_over_time_1h_fullgrid.csv")
print("wrote:", outdir / "gaps_1h_fullrange.csv")
print("wrote heatmaps to:", outdir.resolve())