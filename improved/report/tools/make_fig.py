"""열화 강도별 정확도 곡선 — 4개 모델, seed 0·1·2 평균 ± 표준편차 (per_condition.csv 에서 생성)."""
import csv, collections, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

kr = [f.name for f in fm.fontManager.ttflist if "Noto Sans CJK" in f.name]
plt.rcParams["font.family"] = kr[0] if kr else "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False
TEXT, TEXT2, GRID, SURF = "#191919", "#666666", "#e6e6e6", "#ffffff"

R = list(csv.DictReader(open("per_condition.csv")))
acc = collections.defaultdict(list)                 # (model, corruption, severity) -> [acc per seed]
for r in R:
    acc[(r["model"], r["corruption"], int(r["severity"]))].append(float(r["accuracy"]) * 100)

# 계열: 회색 2개(기준) + 색 2개(증강 계열). 색만으로 구분하지 않도록 선 모양·마커도 다르게.
SERIES = [("base_v2",  "베이스라인",     "#666666", "-",  "o"),
          ("clahe_v2", "CLAHE만",        "#9c9c9c", (0, (4, 2)), "s"),
          ("aug_v2",   "증강만",         "#f47230", "-",  "^"),
          ("both_v2",  "CLAHE + 증강",   "#257de0", "-",  "D")]
PANELS = [("gaussian_noise", "노이즈"), ("gaussian_blur", "블러"), ("contrast", "대비 저하"),
          ("jpeg", "JPEG 압축"), ("motion_blur", "모션블러")]

fig, axes = plt.subplots(2, 3, figsize=(11, 6.6), dpi=200, facecolor=SURF)
fig.subplots_adjust(left=0.065, right=0.985, top=0.80, bottom=0.085, wspace=0.16, hspace=0.42)
xs = np.arange(6)
for ax, (corr, title) in zip(axes.flat, PANELS):
    for model, label, color, ls, mk in SERIES:
        m = np.array([np.mean(acc[(model, "clean", 0)])] + [np.mean(acc[(model, corr, s)]) for s in range(1, 6)])
        sd = np.array([np.std(acc[(model, "clean", 0)])] + [np.std(acc[(model, corr, s)]) for s in range(1, 6)])
        ax.fill_between(xs, m - sd, m + sd, color=color, alpha=0.10, linewidth=0)
        ax.plot(xs, m, color=color, linestyle=ls, linewidth=1.8, marker=mk, markersize=6.5,
                markeredgecolor=SURF, markeredgewidth=1.3, solid_capstyle="round", solid_joinstyle="round", label=label, zorder=3)
    ax.axhline(62.5, color="#b3b3b3", linewidth=0.8, zorder=1)
    ax.set_ylim(50, 95); ax.set_xlim(-0.25, 5.25)
    ax.set_xticks(xs); ax.set_xticklabels(["원본", "1", "2", "3", "4", "5"], color=TEXT2, fontsize=8.5)
    ax.set_yticks([50, 60, 70, 80, 90]); ax.set_yticklabels(["50%", "60%", "70%", "80%", "90%"], color=TEXT2, fontsize=8.5)
    ax.grid(axis="y", color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title(title, loc="left", fontsize=10.5, color=TEXT, pad=6, fontweight="bold")
    ax.set_xlabel("열화 강도", fontsize=8.5, color=TEXT2, labelpad=3)
axes[1, 0].text(-0.2, 63.4, "전부 폐렴으로 답할 때의 정확도 62.5%", fontsize=7.5, color=TEXT2, ha="left", va="bottom")

# 6번째 칸: 범례 + 평균 표
lg = axes[1, 2]; lg.axis("off")
h, l = axes[0, 0].get_legend_handles_labels()
lg.legend(h, l, loc="upper left", bbox_to_anchor=(0.0, 1.02), frameon=False, fontsize=9.5, labelcolor=TEXT,
          handlelength=2.6, labelspacing=0.7, borderaxespad=0)
lg.text(0.0, 0.40, "음영은 seed 3개의 표준편차", fontsize=8.5, color=TEXT2, transform=lg.transAxes)
lg.text(0.0, 0.28, "윗줄 3개: 학습에 쓴 열화", fontsize=8.5, color=TEXT2, transform=lg.transAxes)
lg.text(0.0, 0.16, "아랫줄 2개: 학습에 쓰지 않은 열화", fontsize=8.5, color=TEXT2, transform=lg.transAxes)

fig.text(0.065, 0.945, "열화 증강을 쓰면 강한 열화에서도 정확도가 유지된다. CLAHE는 증강과 함께일 때만 효과가 있다",
         fontsize=13, color=TEXT, fontweight="bold", ha="left")
fig.text(0.065, 0.895, "흉부 X-ray 폐렴 분류, test 624장 정확도 · 원본 CNN + BatchNorm 기준 · seed 0·1·2 평균",
         fontsize=9.5, color=TEXT2, ha="left")
fig.savefig("robustness_curves.png", facecolor=SURF)
print("font:", plt.rcParams["font.family"])
