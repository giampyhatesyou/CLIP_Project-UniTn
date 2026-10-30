"""Draw the base-to-novel trade-off figure used in the README and in the paper.

The numbers are the final test accuracies reported in dl_project.ipynb (section 6.1);
nothing is recomputed here. Run from the repository root:

    python assets/make_figures.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

# (name, base accuracy, novel accuracy, harmonic mean), Flowers102, 10 shots per base class,
# CLIP ViT-B/16. The HM is the one reported in the notebook (computed before rounding).
RESULTS = {
    "clip": ("CLIP zero-shot", 71.33, 78.24, 74.62),
    "cocoop": ("CoCoOp", 93.57, 75.19, 83.38),
    "kd": ("CoCoOp + KD", 87.18, 76.47, 81.47),
    "ours": ("ProtoCoCoOp (ours)", 93.85, 76.74, 84.44),
}

THEMES = {
    "light": dict(bg="#ffffff", ink="#0b0b0b", ink2="#52514e", muted="#898781",
                  contour="#dcdad3", axis="#c3c2b7", accent="#2a78d6", other="#898781"),
    "dark": dict(bg="#0d1117", ink="#f0efec", ink2="#c3c2b7", muted="#898781",
                 contour="#30302d", axis="#383835", accent="#3987e5", other="#898781"),
}

XLIM, YLIM = (73.6, 79.2), (68.0, 97.0)
HM_LEVELS = range(72, 88, 2)


def base_for_hm(hm, novel):
    """Base accuracy that gives harmonic mean `hm` at a given novel accuracy."""
    return hm * novel / (2 * novel - hm)


def pick_font():
    available = {f.name for f in font_manager.fontManager.ttflist}
    return "Inter" if "Inter" in available else "DejaVu Sans"


def draw(theme, out_path, size, fs, labels, title=True, dpi=200):
    t = THEMES[theme]
    plt.rcParams.update({"font.family": pick_font(), "pdf.fonttype": 3, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=size, dpi=dpi)
    fig.patch.set_facecolor(t["bg"])
    ax.set_facecolor(t["bg"])
    halo = [pe.withStroke(linewidth=fs * 0.45, foreground=t["bg"])]

    # Iso-HM curves: every point on a curve has the same harmonic mean.
    novel = np.linspace(*XLIM, 400)
    for hm in HM_LEVELS:
        base = base_for_hm(hm, novel)
        ax.plot(novel, base, color=t["contour"], lw=0.9, zorder=1, solid_capstyle="round")
        if hm % 4 == 2:
            # Label where the curve leaves the plot on the right, or on the top edge.
            x_edge, y_edge = XLIM[1], base_for_hm(hm, XLIM[1])
            if y_edge > YLIM[1] - 0.6:
                x_edge, y_edge = hm * YLIM[1] / (2 * YLIM[1] - hm), YLIM[1]
                ax.text(x_edge + 0.06, y_edge - 0.35, f"HM {hm}", fontsize=fs * 0.82,
                        color=t["muted"], ha="left", va="top", zorder=2, path_effects=halo)
            else:
                ax.text(x_edge - 0.05, y_edge + 0.3, f"HM {hm}", fontsize=fs * 0.82,
                        color=t["muted"], ha="right", va="bottom", zorder=2, path_effects=halo)

    # The two steps that make ProtoCoCoOp: distillation, then prototypes.
    for start, end, text, spec in [("cocoop", "kd", "+ distillation", labels["step_kd"]),
                                   ("kd", "ours", "+ prototypes", labels["step_proto"])]:
        (_, b0, n0, _), (_, b1, n1, _) = RESULTS[start], RESULTS[end]
        ax.annotate("", xy=(n1, b1), xytext=(n0, b0), zorder=3,
                    arrowprops=dict(arrowstyle="-|>,head_length=0.55,head_width=0.28",
                                    color=t["ink2"], lw=1.1, shrinkA=fs * 0.75, shrinkB=fs * 0.75))
        ax.text(*spec["xy"], text, fontsize=fs * 0.9, color=t["ink2"], style="italic",
                ha=spec["ha"], va="center", zorder=4, path_effects=halo)

    for key, (name, base, novel_acc, hm) in RESULTS.items():
        ours = key == "ours"
        ax.scatter([novel_acc], [base], s=(fs * 1.25) ** 2 if ours else (fs * 0.95) ** 2,
                   color=t["accent"] if ours else t["other"], edgecolors=t["bg"],
                   linewidths=1.6, zorder=5)
        spec = labels[key]
        text = spec.get("text", name)
        if spec.get("hm"):
            text += f"  ·  HM {hm:.2f}"
        ax.text(*spec["xy"], text, fontsize=fs, color=t["ink"], ha=spec["ha"],
                va=spec.get("va", "center"), fontweight="semibold" if ours else "normal",
                zorder=6, path_effects=halo)

    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_xticks(range(74, 80))
    ax.set_yticks(range(70, 100, 5))
    ax.set_xlabel("Novel-class accuracy (%)", fontsize=fs * 0.9, color=t["ink2"], labelpad=fs * 0.5)
    ax.set_ylabel("Base-class accuracy (%)", fontsize=fs * 0.9, color=t["ink2"], labelpad=fs * 0.5)
    ax.tick_params(colors=t["muted"], labelsize=fs * 0.82, length=0, pad=fs * 0.4)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["axis"])
        ax.spines[side].set_linewidth(0.8)

    if title:
        fig.text(0.02, 0.965, "Base-to-novel trade-off on Flowers102", fontsize=fs * 1.4,
                 fontweight="semibold", color=t["ink"], ha="left", va="top")
        fig.text(0.02, 0.905, "CLIP ViT-B/16, 10 shots per base class. Each grey curve joins points with the "
                 "same harmonic mean (HM)\nof base and novel accuracy, so up and to the right is better.",
                 fontsize=fs * 0.9, color=t["ink2"], ha="left", va="top", linespacing=1.45)
        fig.subplots_adjust(left=0.085, right=0.975, top=0.80, bottom=0.115)
    else:
        fig.tight_layout(pad=0.3)

    fig.savefig(out_path, facecolor=t["bg"], dpi=dpi)
    plt.close(fig)
    print("wrote", out_path)


README_LABELS = {
    "clip": dict(xy=(78.1, 71.33), ha="right", hm=True),
    "cocoop": dict(xy=(75.05, 93.57), ha="right", hm=True),
    "kd": dict(xy=(76.6, 85.75), ha="left", hm=True),
    "ours": dict(xy=(76.9, 93.85), ha="left", hm=True),
    "step_kd": dict(xy=(75.68, 90.4), ha="right"),
    "step_proto": dict(xy=(76.75, 90.6), ha="left"),
}

PAPER_LABELS = {
    "clip": dict(xy=(78.05, 71.33), ha="right"),
    "cocoop": dict(xy=(75.0, 93.57), ha="right"),
    "kd": dict(xy=(76.62, 86.0), ha="left"),
    "ours": dict(xy=(76.92, 93.85), ha="left", text="ProtoCoCoOp"),
    "step_kd": dict(xy=(75.66, 90.3), ha="right"),
    "step_proto": dict(xy=(76.76, 90.7), ha="left"),
}

if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for theme in ("light", "dark"):
        draw(theme, os.path.join(root, "assets", f"tradeoff_{theme}.png"),
             size=(8.0, 5.0), fs=10.5, labels=README_LABELS)
    draw("light", os.path.join(root, "paper", "figures", "tradeoff.pdf"),
         size=(3.35, 2.35), fs=7.2, labels=PAPER_LABELS, title=False)
