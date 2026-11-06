"""
WSJ-style chart helpers for matplotlib.

Usage:
    from wsj_style import apply_style, finish, ACCENT, ACCENT_2, GRAY
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ...plot...
    finish(fig, ax, title="...", subtitle="...", source="...")
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Palette: one accent does the talking, gray is context.
# Blue + orange validated colorblind-safe as a pair; aqua only with direct labels.
ACCENT = "#2a78d6"      # the story
ACCENT_2 = "#eb6834"    # a second series / contrast
ACCENT_3 = "#1baf7a"    # only if direct-labeled
GRAY = "#b8b6b0"        # context marks
INK = "#111111"         # titles, values
INK_2 = "#55534f"       # subtitles, axis labels
INK_3 = "#8a8883"       # source line, tick labels
GRID = "#e4e2dd"
BG = "#ffffff"


def apply_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 11,
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "axes.edgecolor": INK_3,
        "axes.labelcolor": INK_2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "xtick.color": INK_3,
        "ytick.color": INK_3,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "ytick.left": False,          # no tick marks, just labels
        "lines.linewidth": 2.2,
        "lines.solid_capstyle": "round",
        "legend.frameon": False,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.facecolor": BG,
    })


def finish(fig, ax, title, subtitle=None, source=None):
    """Left-aligned bold title, gray subtitle, source note bottom-left."""
    left = ax.get_position().x0
    fig.text(left, 1.0, title, fontsize=16, fontweight="bold", color=INK,
             ha="left", va="bottom", transform=fig.transFigure)
    if subtitle:
        fig.text(left, 0.985, subtitle, fontsize=11.5, color=INK_2,
                 ha="left", va="top", transform=fig.transFigure)
    if source:
        fig.text(left, -0.02, source, fontsize=8.5, color=INK_3,
                 ha="left", va="top", transform=fig.transFigure)
    fig.subplots_adjust(top=0.86)


def pct_axis(ax, axis="y", decimals=0):
    fmt = mticker.PercentFormatter(xmax=1, decimals=decimals)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def comma_axis(ax, axis="x"):
    fmt = mticker.FuncFormatter(lambda v, _: f"{v:,.0f}")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)