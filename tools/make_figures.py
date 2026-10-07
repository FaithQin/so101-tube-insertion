"""Render the write-up figure set from `figures_data.py` (numbers) into
analysis/figures/. Palette and marks follow the dataviz method: two categorical
slots for the arms (ACT-A blue, ACT-B orange), aqua for the v3 dataset, gray for
context; thin marks, hairline recessive grid, direct labels, a legend for every
two-series chart. Palette validated Sep 2 (adjacent CVD dE 9.2, normal 27.6).

Run with an environment that has matplotlib (the patched lerobot env does not):
    <figenv>/bin/python tools/make_figures.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import figures_data as fd  # noqa: E402

OUT = fd.ROOT / "analysis" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
A, B, V3C, V2C = "#2a78d6", "#eb6834", "#1baf7a", "#898781"
COLOR = {"ACT-A": A, "ACT-B": B}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.labelcolor": MUTED, "text.color": INK, "font.size": 9.5,
})


def frame(ax, title, subtitle, ygrid=True):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if ygrid:
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontsize=12.5, color=INK, fontweight="semibold", pad=18)
    ax.text(0, 1.03, subtitle, transform=ax.transAxes, fontsize=9, color=INK2, va="bottom")


def fig1_stage_split(rows):
    cum = fd.stage_cumulative(rows)
    labels = ["reached the tube", "held it", "carried it to the rack", "seated it"]
    x = np.arange(len(fd.STAGES))
    w = 0.3
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200)
    for i, pol in enumerate(("ACT-A", "ACT-B")):
        vals = [100 * cum[pol][s] for s in fd.STAGES]
        bars = ax.bar(x + (i - 0.5) * (w + 0.04), vals, width=w, color=COLOR[pol],
                      edgecolor=SURFACE, linewidth=2, label=pol)
        for b_, v in zip(bars, vals):
            ax.text(b_.get_x() + b_.get_width() / 2, v + 2, f"{v:.0f}%", ha="center", va="bottom",
                    fontsize=9, color=INK2)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100%"])
    frame(ax, "Where each arm's 20 trials got to",
          "v2 paired eval, Aug 25 2026 · share of trials reaching at least each stage · live scoring, every trial on video")
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_stage_split.png")
    plt.close(fig)
    return cum


def fig2_close_height(rows):
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(8, 3.8), dpi=200)
    ypos = {"ACT-A": 1.0, "ACT-B": 0.0}
    for r in rows:
        y = ypos[r["policy"]] + rng.uniform(-0.13, 0.13)
        c = COLOR[r["policy"]]
        if r["success"]:
            ax.scatter(r["close_offset_deg"], y, s=62, color=c, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        else:
            ax.scatter(r["close_offset_deg"], y, s=62, facecolor=SURFACE, edgecolor=c, linewidth=1.8, zorder=3)
    ax.axvline(0, color=AXIS, linewidth=1, zorder=1)
    ax.text(0.15, 1.42, "demonstrations' median close height", fontsize=8.5, color=INK2, va="center")
    ax.set_yticks([1.0, 0.0], ["ACT-A", "ACT-B"])
    ax.tick_params(axis="y", colors=INK2, length=0)
    ax.set_ylim(-0.5, 1.6)
    ax.set_xlabel("commanded close height at first grasp, degrees above the demonstrations' median (about 1 cm per 3°)")
    handles = [
        Line2D([], [], marker="o", color=A, markersize=8, linestyle="", markeredgecolor=SURFACE, label="ACT-A seated"),
        Line2D([], [], marker="o", markerfacecolor=SURFACE, markeredgecolor=A, markeredgewidth=1.8, markersize=8, linestyle="", label="ACT-A failed"),
        Line2D([], [], marker="o", color=B, markersize=8, linestyle="", markeredgecolor=SURFACE, label="ACT-B seated"),
        Line2D([], [], marker="o", markerfacecolor=SURFACE, markeredgecolor=B, markeredgewidth=1.8, markersize=8, linestyle="", label="ACT-B failed"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper right", fontsize=8.5, ncol=2)
    frame(ax, "The close height decided the grasp",
          "v2, 40 trials · every ACT-A seat closed within 3.5° of the demonstrations · ACT-B closed high in 19 of 20", ygrid=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_close_height.png")
    plt.close(fig)


def fig3_durations(d2, d3):
    s2, s3 = fd.duration_summary(d2), fd.duration_summary(d3)
    rng = np.random.default_rng(1)
    fig, ax = plt.subplots(figsize=(8, 3.8), dpi=200)
    for y, d, c, name, s in ((1.0, d2, V2C, "v2, generic", s2), (0.0, d3, V3C, "v3, recovery-targeted", s3)):
        ys = y + rng.uniform(-0.14, 0.14, size=len(d))
        ax.scatter(d.values, ys, s=40, color=c, edgecolor=SURFACE, linewidth=1.2, zorder=3)
        ax.plot([s["median"]] * 2, [y - 0.3, y + 0.3], color=INK, linewidth=2, zorder=4)
        ax.plot([s["p90"]] * 2, [y - 0.22, y + 0.22], color=INK2, linewidth=1, zorder=4)
        ax.text(s["median"], y + 0.36, f"median {s['median']:.1f} s", ha="center", fontsize=8.5, color=INK2)
        ax.text(s["p90"], y - 0.44, f"p90 {s['p90']:.1f} s", ha="center", fontsize=8.5, color=INK2)
    tail = d2[d2 > 1.5 * s2["median"]]
    ax.annotate(f"{len(tail)} episodes more than 50% above the median:\nthe recovery demonstrations",
                xy=(float(tail.min()), 1.1), xytext=(float(d2.max()), 1.62),
                fontsize=8.5, color=INK2, ha="right", va="center",
                arrowprops=dict(arrowstyle="-", color=AXIS, linewidth=0.8))
    ax.text(float(d3.max()) + 0.3, 0.0, "no episode above 13.8 s", fontsize=8.5, color=INK2, va="center")
    ax.set_yticks([1.0, 0.0], ["v2  (generic)", "v3  (recovery-targeted)"])
    ax.tick_params(axis="y", colors=INK2, length=0)
    ax.set_ylim(-0.7, 1.9)
    ax.set_xlabel("episode duration, seconds (50 demonstrations per dataset)")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    frame(ax, "The recovery dataset has no long tail",
          "corrective behaviour shows up as long episodes · the decisiveness rule removed every one of them from v3", ygrid=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_durations.png")
    plt.close(fig)
    return s2, s3


def fig4_final_positions(pts):
    img = plt.imread(fd.ROOT / "tools" / "v2_training_start_reference.png")
    fig, ax = plt.subplots(figsize=(8, 4.9), dpi=200)
    ax.imshow(img)
    ax.set_xlim(-12, 1292)
    ax.set_ylim(772, -12)
    ax.set_axis_off()
    sx, sy = fd.SEAT_PX
    ax.scatter([sx], [sy], s=140, marker="s", facecolor="none", edgecolor=INK, linewidth=1.6, zorder=4)
    ax.text(sx + 14, sy - 12, "seat", fontsize=9, color=INK, va="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=SURFACE, edgecolor="none", alpha=0.85))
    tx, ty = fd.START_PX
    ax.scatter([tx], [ty], s=90, marker="x", color=INK, linewidth=1.6, zorder=4)
    near = [p for p in pts if np.hypot(p["x"] - tx, p["y"] - ty) < 40]
    far = [p for p in pts if p not in near]
    for p in pts:
        ax.scatter(p["x"], p["y"], s=95, color=COLOR[p["policy"]], edgecolor=SURFACE, linewidth=1.8, zorder=5)
    for i, p in enumerate(sorted(far, key=lambda q: q["x"])):
        dx, dy = (22, -30) if i % 2 == 0 else (22, 34)
        ax.annotate(p["label"], xy=(p["x"], p["y"]), xytext=(p["x"] + dx, p["y"] + dy), fontsize=8.5, color=INK,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor=SURFACE, edgecolor="none", alpha=0.9),
                    arrowprops=dict(arrowstyle="-", color=INK2, linewidth=0.8), zorder=6)
    ax.annotate(f"{len(near)} of {len(pts)} ended at the start mark:\nthe tube never moved",
                xy=(tx, ty), xytext=(tx - 70, ty - 120), fontsize=8.5, color=INK, ha="center",
                bbox=dict(boxstyle="round,pad=0.25", facecolor=SURFACE, edgecolor="none", alpha=0.9),
                arrowprops=dict(arrowstyle="-", color=INK2, linewidth=0.8), zorder=6)
    handles = [Line2D([], [], marker="o", color=A, markersize=8, linestyle="", markeredgecolor=SURFACE, label="ACT-A final cap position"),
               Line2D([], [], marker="o", color=B, markersize=8, linestyle="", markeredgecolor=SURFACE, label="ACT-B final cap position")]
    ax.legend(handles=handles, frameon=True, facecolor=SURFACE, edgecolor="none", loc="upper right", fontsize=8.5)
    ax.set_title("Where the tube ended up, v3 first-contact probes", loc="left", fontsize=12.5, color=INK,
                 fontweight="semibold", pad=40)
    ax.text(0, 1.02, "Aug 29–30 2026 · 8 valid trials (ACT-A 6, ACT-B 2), drawn on the training reference frame\n"
            "the three that carried the tube dropped it in three different places",
            transform=ax.transAxes, fontsize=9, color=INK2, va="bottom")
    fig.subplots_adjust(top=0.80, bottom=0.01, left=0.01, right=0.99)
    fig.savefig(OUT / "fig4_v3_final_positions.png")
    plt.close(fig)
    return len(near), len(far)


def main():
    rows = fd.v2_trials()
    cum = fig1_stage_split(rows)
    fig2_close_height(rows)
    d2, d3 = fd.episode_durations(fd.V2), fd.episode_durations(fd.V3)
    s2, s3 = fig3_durations(d2, d3)
    pts = fd.v3_final_positions()
    near, far = fig4_final_positions(pts)
    numbers = {
        "fig1_stage_cumulative": cum,
        "fig2_close_offsets": [{k: r[k] for k in ("trial", "policy", "close_offset_deg", "success")} for r in rows],
        "fig3_durations": {"v2": s2, "v3": s3},
        "fig4_final_positions": {"points": pts, "at_start_mark": near, "displaced": far,
                                  "seat_px": fd.SEAT_PX, "start_px": fd.START_PX},
    }
    (OUT / "numbers.json").write_text(json.dumps(numbers, indent=2))
    print(json.dumps({"stage": cum, "durations": {"v2": s2, "v3": s3}, "final_positions": {"at_start": near, "displaced": far}}, indent=1))


if __name__ == "__main__":
    main()
