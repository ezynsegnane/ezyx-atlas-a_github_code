"""Render the publication-style EZNX-ATLAS-A architecture figure."""

from __future__ import annotations

from pathlib import Path
from shutil import copyfile

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = ROOT / "mdpi_mathematics_submission_package" / "MDPI_template_ACS"
FIG_DIR = TEMPLATE_DIR / "figures"
MIRROR_DIR = TEMPLATE_DIR


C = {
    "ink": "#1f2937",
    "muted": "#64748b",
    "line": "#334155",
    "ecg": "#edf7e8",
    "ecg_edge": "#6f9e57",
    "meta": "#eaf3ff",
    "meta_edge": "#5f8ec1",
    "fusion": "#fff2df",
    "fusion_edge": "#c0843d",
    "gray": "#f6f7f9",
    "gray_edge": "#9ca3af",
    "purple": "#f5edff",
    "purple_edge": "#8b5cf6",
}


def text(ax, x, y, s, size=9.0, weight="normal", color=None, ha="center", va="center"):
    ax.text(
        x,
        y,
        s,
        fontsize=size,
        fontweight=weight,
        color=color or C["ink"],
        ha=ha,
        va=va,
        linespacing=1.18,
    )


def box(ax, x, y, w, h, title, body, face, edge, title_size=8.6, body_size=7.6, lw=1.15):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.010,rounding_size=0.010",
        linewidth=lw,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(patch)
    text(ax, x + w / 2, y + h - 0.028, title, size=title_size, weight="bold")
    text(ax, x + w / 2, y + h / 2 - 0.013, body, size=body_size)
    return (x, y, w, h)


def lft(b):
    x, y, _, h = b
    return x, y + h / 2


def rgt(b):
    x, y, w, h = b
    return x + w, y + h / 2


def top(b):
    x, y, w, h = b
    return x + w / 2, y + h


def bot(b):
    x, y, w, _ = b
    return x + w / 2, y


def arrow(ax, start, end, color=None, rad=0.0, dashed=False, lw=1.25):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=lw,
            color=color or C["line"],
            connectionstyle=f"arc3,rad={rad}",
            linestyle=(0, (3, 3)) if dashed else "solid",
            shrinkA=2,
            shrinkB=2,
        )
    )


def render(output_dir: Path = FIG_DIR, mirror_dir: Path = MIRROR_DIR) -> list[Path]:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "dejavusans",
            "axes.unicode_minus": False,
        }
    )

    fig, ax = plt.subplots(figsize=(14.0, 6.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    text(ax, 0.5, 0.965, "EZNX-ATLAS-A architecture", size=15.5, weight="bold", color="#111827")
    text(
        ax,
        0.5,
        0.928,
        "A quality-gated ECG model with demographic and anthropometric metadata",
        size=9.7,
        color=C["muted"],
    )

    # Branch labels.
    text(ax, 0.070, 0.875, "ECG branch", size=9.8, weight="bold", color=C["ecg_edge"], ha="left")
    text(ax, 0.070, 0.640, "Metadata branch", size=9.8, weight="bold", color=C["meta_edge"], ha="left")
    text(ax, 0.070, 0.385, "Cross-modal fusion", size=9.8, weight="bold", color=C["fusion_edge"], ha="left")

    # ECG pathway.
    ecg_in = box(
        ax,
        0.070,
        0.720,
        0.145,
        0.115,
        "Input ECG",
        r"$\mathbf{X}\in\mathbb{R}^{12\times1000}$" "\n100 Hz; scale by 1/5",
        C["ecg"],
        C["ecg_edge"],
    )
    ecg_encoder = box(
        ax,
        0.265,
        0.695,
        0.215,
        0.165,
        "TSBackbone1D_v5",
        "Conv7/s2, maxpool\nthree residual stages\n"
        r"$64\to64,\ 64\to128,\ 128\to256$",
        C["ecg"],
        C["ecg_edge"],
        body_size=7.4,
    )
    h_ts1 = box(
        ax,
        0.540,
        0.720,
        0.145,
        0.115,
        "ECG embedding",
        r"$[\mu,\sigma,\max,\min]$" "\n" r"$\mathbf{h}_{ts1}\in\mathbb{R}^{1024}$",
        C["ecg"],
        C["ecg_edge"],
    )

    arrow(ax, rgt(ecg_in), lft(ecg_encoder))
    arrow(ax, rgt(ecg_encoder), lft(h_ts1))

    # Metadata pathway.
    meta_in = box(
        ax,
        0.070,
        0.480,
        0.145,
        0.115,
        "Tabular input",
        r"$\mathbf{x}_{meta}\in\mathbb{R}^{8}$" "\n" r"$\mathbf{m}\in\{0,1\}^{8}$",
        C["meta"],
        C["meta_edge"],
    )
    meta_construct = box(
        ax,
        0.265,
        0.455,
        0.215,
        0.165,
        "Split and quality",
        r"demo: $\mathbb{R}^{4}$; anthro: $\mathbb{R}^{12}$" "\n"
        r"$q_d=\frac{1}{2}\sum_{0}^{1}m_i,\quad q_a=\frac{1}{6}\sum_{2}^{7}m_i$" "\n"
        r"$q_{meta}=\min(1,q_d+0.5q_a)$",
        C["meta"],
        C["meta_edge"],
        body_size=7.0,
    )
    h_meta = box(
        ax,
        0.540,
        0.440,
        0.235,
        0.195,
        "Metadata embedding",
        r"$\mathbf{h}_{demo}=MLP_{demo}(\cdot)$" "\n"
        r"$\widetilde{\mathbf{h}}_{anthro}=q_a\,MLP_{anthro}(\cdot)$" "\n"
        r"$\bar{\mathbf{h}}_m=MLP_m([\mathbf{h}_{demo}\Vert\widetilde{\mathbf{h}}_{anthro}\Vert q_d\Vert q_a])$" "\n"
        r"$p_{drop}=0.10,\quad \mathbf{h}_m=q_{meta}\bar{\mathbf{h}}_m\in\mathbb{R}^{128}$",
        C["meta"],
        C["meta_edge"],
        title_size=8.5,
        body_size=6.5,
    )

    arrow(ax, rgt(meta_in), lft(meta_construct))
    arrow(ax, rgt(meta_construct), lft(h_meta))

    # Fusion module.
    fusion = box(
        ax,
        0.325,
        0.190,
        0.410,
        0.165,
        "Quality-gated fusion module",
        r"$\mathbf{h}_{ts}=\mathbf{h}_{ts1}+0.10\,q_{meta}W_{res}\mathbf{h}_m,\quad W_{res}:128\to1024$ (zero init.)"
        "\n"
        r"$\mathbf{h}=[\mathbf{h}_{ts}\Vert\mathbf{h}_m]\in\mathbb{R}^{1152},\quad"
        r"\mathbf{z}=\mathbf{h}\odot\sigma(W_2\,ReLU(W_1\mathbf{h}))$"
        "\n"
        r"$\ell_f=W_f\mathbf{z}+b_f+0.05\,q_{meta}\ell_m$",
        C["fusion"],
        C["fusion_edge"],
        title_size=8.7,
        body_size=6.6,
    )

    arrow(ax, bot(h_ts1), (0.500, 0.355), rad=0.05)
    arrow(ax, bot(h_meta), (0.600, 0.355), rad=-0.04)

    # Heads.
    ecg_head = box(
        ax,
        0.820,
        0.730,
        0.130,
        0.090,
        "ECG-only head",
        r"$\ell_{ecg}=W_e\mathbf{h}_{ts1}+b_e$",
        "#ffffff",
        C["ecg_edge"],
        title_size=8.1,
        body_size=6.9,
    )
    meta_head = box(
        ax,
        0.820,
        0.485,
        0.130,
        0.090,
        "Meta-only head",
        r"$\ell_m=W_m\mathbf{h}_m+b_m$",
        "#ffffff",
        C["meta_edge"],
        title_size=8.1,
        body_size=6.9,
    )
    fused_head = box(
        ax,
        0.820,
        0.225,
        0.130,
        0.090,
        "Fused head",
        r"$\ell_f$",
        "#ffffff",
        C["fusion_edge"],
        title_size=8.2,
        body_size=8.0,
    )

    arrow(ax, rgt(h_ts1), lft(ecg_head))
    arrow(ax, rgt(h_meta), lft(meta_head))
    arrow(ax, rgt(fusion), lft(fused_head))

    # Objective and inference are shown as readout boxes to avoid cluttering the model graph.
    objective = box(
        ax,
        0.070,
        0.045,
        0.455,
        0.095,
        "Training objective",
        r"$0.52\,BCE_{w^+}(\ell_f,y)+0.30\,BCE_{w^+}(\ell_{ecg},y)$" "\n"
        r"$+\,0.10\,q_{meta}BCE(\ell_m,y)+0.08\,\mathcal{L}_{AUC}(\sigma(\ell_f),y)$",
        C["purple"],
        C["purple_edge"],
        title_size=8.2,
        body_size=6.8,
    )
    inference = box(
        ax,
        0.575,
        0.045,
        0.375,
        0.095,
        "Validation-tuned inference",
        r"$\mathbf{p}=w^\ast\sigma(\ell_f)+(1-w^\ast)\sigma(\ell_{ecg})$" "\n"
        r"$w^\ast\in\{0,.50,.65,.75,.85,.95,1\}$; class thresholds tuned on validation",
        C["purple"],
        C["purple_edge"],
        title_size=8.2,
        body_size=6.7,
    )

    # Subtle lane separators.
    for y in (0.670, 0.405, 0.165):
        ax.plot([0.055, 0.955], [y, y], color="#e5e7eb", lw=0.75, zorder=0)

    output_dir.mkdir(parents=True, exist_ok=True)
    mirror_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    for fmt in ("png", "pdf", "svg"):
        path = output_dir / f"fig1_architecture.{fmt}"
        fig.savefig(path, dpi=400, bbox_inches="tight", pad_inches=0.05)
        paths.append(path)

    for path in list(paths):
        mirror_path = mirror_dir / path.name
        copyfile(path, mirror_path)
        paths.append(mirror_path)

    plt.close(fig)
    return paths


if __name__ == "__main__":
    for generated in render():
        print(generated)
