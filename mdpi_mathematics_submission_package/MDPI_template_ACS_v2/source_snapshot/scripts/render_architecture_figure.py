"""Render the publication-style EZNX-ATLAS-A architecture figure.

The figure is generated from the implementation contract used in
``eznx_model_v5.py`` and ``atlas_a_v5_multiseed_v2.py``:

* Temporal statistics pooling is [mean, std, max, min].
* Final prediction uses the fused branch only, with w*=1.0 fixed a priori.
* Class thresholds are tuned on validation, but the fused/ECG blend is not.
"""

from __future__ import annotations

from pathlib import Path
from shutil import copyfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


def default_output_dirs() -> list[Path]:
    """Resolve output folders from either the repo root or the MDPI source snapshot."""

    here = Path(__file__).resolve()
    for parent in here.parents:
        if parent.name == "MDPI_template_ACS_v2" and (parent / "figures").exists():
            dirs = [parent / "figures"]
            repo_root = parent.parents[1] if len(parent.parents) > 1 else None
            if repo_root and (repo_root / "figures").exists():
                dirs.insert(0, repo_root / "figures")
            return list(dict.fromkeys(dirs))

        template_dir = (
            parent
            / "mdpi_mathematics_submission_package"
            / "MDPI_template_ACS_v2"
            / "figures"
        )
        if (parent / "figures").exists() and template_dir.exists():
            return [parent / "figures", template_dir]

    fallback = here.parents[1] / "figures"
    return [fallback]


C = {
    "ink": "#111827",
    "muted": "#4b5563",
    "line": "#111827",
    "ecg_bg": "#eef7e8",
    "ecg_box": "#f8fff3",
    "ecg_edge": "#5c844a",
    "meta_bg": "#eaf2ff",
    "meta_box": "#f7fbff",
    "meta_edge": "#416b9f",
    "fusion_bg": "#fff0cf",
    "fusion_box": "#fff7df",
    "fusion_edge": "#a96f00",
    "readout_bg": "#f4e7fb",
    "readout_box": "#fbf5ff",
    "readout_edge": "#7a3f8f",
}


def text(
    ax,
    x: float,
    y: float,
    s: str,
    size: float = 9.0,
    weight: str = "normal",
    color: str | None = None,
    ha: str = "center",
    va: str = "center",
) -> None:
    ax.text(
        x,
        y,
        s,
        fontsize=size,
        fontweight=weight,
        color=color or C["ink"],
        ha=ha,
        va=va,
        linespacing=1.16,
    )


def box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    body: str,
    face: str,
    edge: str,
    title_size: float = 9.2,
    body_size: float = 8.0,
    lw: float = 1.05,
):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.009,rounding_size=0.012",
        linewidth=lw,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(patch)
    text(ax, x + w / 2, y + h - 0.025, title, size=title_size, weight="bold")
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


def arrow(ax, start, end, color: str | None = None, rad: float = 0.0, lw: float = 1.2):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=lw,
            color=color or C["line"],
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=3,
            shrinkB=3,
        )
    )


def lane(ax, y: float, h: float, color: str) -> None:
    ax.add_patch(Rectangle((0.0, y), 1.0, h, facecolor=color, edgecolor="none", zorder=-10))


def render(output_dirs: list[Path] | None = None) -> list[Path]:
    """Render Fig. 1 into the repository and the MDPI v2 template."""

    if output_dirs is None:
        output_dirs = default_output_dirs()

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "dejavusans",
            "axes.unicode_minus": False,
        }
    )

    fig, ax = plt.subplots(figsize=(7.6, 9.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    lane(ax, 0.700, 0.285, C["ecg_bg"])
    lane(ax, 0.385, 0.315, C["meta_bg"])
    lane(ax, 0.165, 0.220, C["fusion_bg"])
    lane(ax, 0.010, 0.155, C["readout_bg"])

    text(ax, 0.02, 0.962, "ECG branch", size=12.0, weight="bold", ha="left", color=C["ecg_edge"])
    text(ax, 0.02, 0.675, "Metadata input", size=12.0, weight="bold", ha="left", color=C["meta_edge"])
    text(
        ax,
        0.02,
        0.360,
        "Cross-modal fusion",
        size=12.0,
        weight="bold",
        ha="left",
        color=C["fusion_edge"],
    )

    # ECG pathway.
    ecg_in = box(
        ax,
        0.37,
        0.895,
        0.26,
        0.075,
        "Input ECG",
        r"$\mathbf{X}\in\mathbb{R}^{12\times1000}$" "\n100 Hz; scale by 1/5",
        C["ecg_box"],
        C["ecg_edge"],
        title_size=10.0,
        body_size=8.4,
    )
    backbone = box(
        ax,
        0.315,
        0.790,
        0.37,
        0.085,
        "TSBackbone1D_v5",
        "Conv7/s2, maxpool; residual stages\n"
        r"$64{\to}64,\ 64{\to}128,\ 128{\to}256$",
        C["ecg_box"],
        C["ecg_edge"],
        title_size=10.0,
        body_size=8.0,
    )
    h_ts1 = box(
        ax,
        0.37,
        0.710,
        0.26,
        0.065,
        "ECG embedding",
        r"$[\mu,\sigma,\max,\min]$" "\n" r"$\mathbf{h}_{ts1}\in\mathbb{R}^{1024}$",
        C["ecg_box"],
        C["ecg_edge"],
        title_size=9.5,
        body_size=8.1,
    )
    ecg_head = box(
        ax,
        0.715,
        0.716,
        0.25,
        0.060,
        "ECG-only head",
        r"$\ell_{ecg}=W_{ecg}\mathbf{h}_{ts1}+b_{ecg}$",
        "#ffffff",
        C["ecg_edge"],
        title_size=8.8,
        body_size=7.7,
    )
    arrow(ax, bot(ecg_in), top(backbone))
    arrow(ax, bot(backbone), top(h_ts1))
    arrow(ax, rgt(h_ts1), lft(ecg_head))

    # Metadata pathway.
    meta_in = box(
        ax,
        0.035,
        0.555,
        0.25,
        0.090,
        "Tabular input",
        r"$[\mathbf{x}_{meta}\Vert\mathbf{m}]\in\mathbb{R}^{16}$" "\n"
        r"$\mathbf{x}_{meta}\in\mathbb{R}^{8},\ \mathbf{m}\in\{0,1\}^{8}$",
        C["meta_box"],
        C["meta_edge"],
        title_size=9.4,
        body_size=7.8,
    )
    split = box(
        ax,
        0.090,
        0.420,
        0.38,
        0.110,
        "Split and quality",
        r"$\mathbf{x}_{demo}\in\mathbb{R}^{4},\ \mathbf{x}_{anthro}\in\mathbb{R}^{12}$" "\n"
        r"$q_d=\frac{1}{2}\sum_{i=0}^{1}m_i,\quad q_a=\frac{1}{6}\sum_{i=2}^{7}m_i$" "\n"
        r"$q_{meta}=\min(1,q_d+0.5q_a)$",
        C["meta_box"],
        C["meta_edge"],
        title_size=9.2,
        body_size=7.5,
    )
    h_meta = box(
        ax,
        0.570,
        0.500,
        0.385,
        0.135,
        "Metadata embedding",
        r"$\mathbf{h}_{demo}=MLP_{demo}(\cdot)$" "\n"
        r"$\widetilde{\mathbf{h}}_{anthro}=q_a\,MLP_{anthro}(\cdot)$" "\n"
        r"$\mathbf{h}_m=q_{meta}MLP_m([\mathbf{h}_{demo}\Vert\widetilde{\mathbf{h}}_{anthro}\Vert q_d\Vert q_a])$" "\n"
        r"$\mathbf{h}_m\in\mathbb{R}^{128}$",
        C["meta_box"],
        C["meta_edge"],
        title_size=9.2,
        body_size=7.4,
    )
    meta_head = box(
        ax,
        0.665,
        0.410,
        0.25,
        0.060,
        "Meta-only head",
        r"$\ell_m=W_m\mathbf{h}_m+b_m$",
        "#ffffff",
        C["meta_edge"],
        title_size=8.8,
        body_size=7.8,
    )
    arrow(ax, bot(meta_in), top(split), rad=-0.06)
    arrow(ax, rgt(split), lft(h_meta), rad=0.04)
    arrow(ax, bot(h_meta), top(meta_head))

    # Fusion and readouts.
    fusion = box(
        ax,
        0.145,
        0.220,
        0.710,
        0.120,
        "Quality-gated fusion module",
        r"$\mathbf{h}_{ts}=\mathbf{h}_{ts1}+0.10\,q_{meta}W_{res}\mathbf{h}_m$" "\n"
        r"$\mathbf{h}=[\mathbf{h}_{ts}\Vert\mathbf{h}_m],\quad"
        r"\mathbf{z}=\mathbf{h}\odot\sigma(g(\mathbf{h}))$" "\n"
        r"$\ell_f=W_f\mathbf{z}+b_f+0.05\,q_{meta}\ell_m$",
        C["fusion_box"],
        C["fusion_edge"],
        title_size=9.4,
        body_size=7.6,
    )
    fused_head = box(
        ax,
        0.400,
        0.168,
        0.20,
        0.045,
        "Fused head",
        r"$\ell_f$",
        "#ffffff",
        C["fusion_edge"],
        title_size=8.8,
        body_size=8.0,
    )
    arrow(ax, bot(h_ts1), (0.38, 0.342), rad=0.10)
    arrow(ax, bot(split), (0.30, 0.342))
    arrow(ax, bot(h_meta), (0.60, 0.342), rad=-0.05)
    arrow(ax, bot(meta_head), (0.72, 0.342), rad=-0.03)
    arrow(ax, bot(fusion), top(fused_head))

    objective = box(
        ax,
        0.035,
        0.090,
        0.930,
        0.065,
        "Training objective",
        r"$0.52\,\mathrm{BCE}_{w^+}(\ell_f,y)+0.30\,\mathrm{BCE}_{w^+}(\ell_{ecg},y)"
        r"+0.10\,q_{meta}\mathrm{BCE}(\ell_m,y)+0.08\,\mathcal{L}_{AUC}(\sigma(\ell_f),y)$",
        C["readout_box"],
        C["readout_edge"],
        title_size=8.5,
        body_size=7.0,
    )
    inference = box(
        ax,
        0.035,
        0.025,
        0.930,
        0.055,
        "Fixed-a-priori inference",
        r"$\mathbf{p}=\sigma(\ell_f)$"
        r"$\quad (w^\ast=1.0\ \mathrm{fixed\ a\ priori});\quad$"
        "class thresholds tuned on validation",
        C["readout_box"],
        C["readout_edge"],
        title_size=8.5,
        body_size=7.0,
    )

    # Fine separators.
    for y in (0.700, 0.385, 0.165):
        ax.plot([0.0, 1.0], [y, y], color="#ffffff", lw=1.2, zorder=-1)

    generated: list[Path] = []
    primary_dir = output_dirs[0]
    primary_dir.mkdir(parents=True, exist_ok=True)
    primary_pdf = primary_dir / "fig1_architecture.pdf"
    fig.savefig(primary_pdf, dpi=400, bbox_inches="tight", pad_inches=0.02)
    generated.append(primary_pdf)

    for output_dir in output_dirs[1:]:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "fig1_architecture.pdf"
        copyfile(primary_pdf, path)
        generated.append(path)

    plt.close(fig)
    return generated


if __name__ == "__main__":
    for generated_path in render():
        print(generated_path)
