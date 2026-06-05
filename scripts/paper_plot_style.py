"""Shared typography and axis styling for ECBIT manuscript figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def apply_paper_style(font_size: float = 9.0) -> None:
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "font.size": font_size,
            "axes.labelsize": font_size,
            "axes.titlesize": font_size,
            "axes.titleweight": "normal",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#222222",
            "axes.linewidth": 0.8,
            "xtick.labelsize": font_size - 0.2,
            "ytick.labelsize": font_size - 0.2,
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "legend.fontsize": max(font_size - 0.4, 7.0),
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
        }
    )


def style_axis(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.grid(axis=grid_axis, color="#DDDDDD", linewidth=0.6, alpha=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def save_figure(fig: plt.Figure, out_pdf: Path, out_png: Path) -> None:
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
