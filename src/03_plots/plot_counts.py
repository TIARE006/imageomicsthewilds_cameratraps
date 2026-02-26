import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("counts_over_time.csv", parse_dates=["tbin"])

# 把 1min bins 汇总成 1h bins
df["hour"] = df["tbin"].dt.floor("h")
h = df.groupby(["site_cam", "hour"])["count"].sum().reset_index()

cams = sorted(h["site_cam"].unique())

# 画一张总览图（4条线）
plt.figure()
for cam in cams:
    g = h[h["site_cam"] == cam].sort_values("hour")
    plt.plot(g["hour"], g["count"], label=cam)
plt.xlabel("Time (hour)")
plt.ylabel("Images per hour")
plt.title("Camera-trap arrivals: counts over time (1h bins)")
plt.legend()
plt.tight_layout()
plt.savefig("counts_over_time_1h_allcams.png", dpi=200)
plt.close()

# 每个 camera 单独一张（更清晰）
for cam in cams:
    g = h[h["site_cam"] == cam].sort_values("hour")
    plt.figure()
    plt.plot(g["hour"], g["count"])
    plt.xlabel("Time (hour)")
    plt.ylabel("Images per hour")
    plt.title(f"{cam}: counts over time (1h bins)")
    plt.tight_layout()
    plt.savefig(f"counts_over_time_1h_{cam}.png", dpi=200)
    plt.close()

print("wrote: counts_over_time_1h_allcams.png and per-cam PNGs")