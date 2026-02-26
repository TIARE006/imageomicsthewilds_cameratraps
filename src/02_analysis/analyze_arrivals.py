import pandas as pd

EV = "events.csv"
BIN = "1min"   # 先用 1min，之后你可改成 "5min" 或 "1H"

ev = pd.read_csv(EV, parse_dates=["timestamp"])
print("rows=", len(ev), "cams=", ev["site_cam"].nunique(), "sessions=", ev["session"].nunique())
print("\ncounts by camera:\n", ev["site_cam"].value_counts())

# 1) arrival / trigger counts over time (binned)
ev["tbin"] = ev["timestamp"].dt.floor(BIN)
binned = ev.groupby(["site_cam", "tbin"]).size().rename("count").reset_index()
binned.to_csv("counts_over_time.csv", index=False)
print(f"\nwrote counts_over_time.csv (bin={BIN})")

# 2) gap analysis: bins with 0 count
gaps = []
for cam, g in binned.groupby("site_cam"):
    tmin, tmax = g["tbin"].min(), g["tbin"].max()
    full = pd.date_range(tmin, tmax, freq=BIN)
    have = set(g["tbin"])
    for t in full:
        if t not in have:
            gaps.append({"site_cam": cam, "tbin": t})
gaps = pd.DataFrame(gaps, columns=["site_cam","tbin"])
gaps.to_csv("gaps.csv", index=False)
print("wrote gaps.csv gaps=", len(gaps))

# 3) periodic sampling detection (minute-of-hour + second distributions)
ev["minute"] = ev["timestamp"].dt.minute
ev["second"] = ev["timestamp"].dt.second
minute_dist = ev.groupby(["site_cam", "minute"]).size().rename("count").reset_index()
second_dist = ev.groupby(["site_cam", "second"]).size().rename("count").reset_index()
minute_dist.to_csv("minute_of_hour_dist.csv", index=False)
second_dist.to_csv("second_dist.csv", index=False)
print("wrote minute_of_hour_dist.csv and second_dist.csv")

top_min = (minute_dist.sort_values(["site_cam","count"], ascending=[True,False])
           .groupby("site_cam").head(5))
print("\nTop minutes per camera (if you see strong spikes at 0 => on-the-hour sampling):\n", top_min)