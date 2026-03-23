import os
# Add texlive path to environment BEFORE importing matplotlib
os.environ["PATH"] += os.pathsep + '/home/ET/yjzhou/texlive/bin/x86_64-linux'

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.style.use("seaborn-v0_8-paper")

# Nature suggests that fontsizes should be between 5 ~ 7pt.
LARGE_FONT_SIZE = 12
DEFAULT_FONT_SIZE = 10
SMALL_FONT_SIZE = 8
# plt.rc("text", usetex=True)
# plt.rc("font", size=DEFAULT_FONT_SIZE)
# plt.rc("axes", titlesize=DEFAULT_FONT_SIZE)
# plt.rc("axes", labelsize=DEFAULT_FONT_SIZE)
# plt.rc("xtick", labelsize=SMALL_FONT_SIZE)
# plt.rc("ytick", labelsize=SMALL_FONT_SIZE)
# plt.rc("legend", fontsize=SMALL_FONT_SIZE)
# plt.rc("figure", titlesize=DEFAULT_FONT_SIZE)
import matplotlib as mpl


mpl.rcParams["font.size"] = DEFAULT_FONT_SIZE
mpl.rcParams["axes.titlesize"] = DEFAULT_FONT_SIZE
mpl.rcParams["axes.labelsize"] = DEFAULT_FONT_SIZE
mpl.rcParams["xtick.labelsize"] = SMALL_FONT_SIZE
mpl.rcParams["ytick.labelsize"] = SMALL_FONT_SIZE
mpl.rcParams["legend.fontsize"] = SMALL_FONT_SIZE
mpl.rcParams["figure.titlesize"] = LARGE_FONT_SIZE
mpl.rcParams["pgf.preamble"] = "\n".join(
    [
        r"\usepackage{bm}%",
        r"\usepackage{mismath}",
        r"\newcommand{\SSSText}[1]{{\scriptscriptstyle \mathup{#1}}}%",
        r"\renewcommand{\mathdefault}[1][]{}%",
    ]
)

# Costumize pgf settings.
mpl.rcParams["pgf.texsystem"] = "pdflatex"
mpl.rcParams["pgf.rcfonts"] = False
mpl.rcParams["text.usetex"] = True
mpl.rcParams["font.family"] = "serif"
# mpl.rcParams['font.serif'] = ['Times New Roman']


A4_WIDTH = 6.5
# Nature suggests the width should be 180mm.
NATURE_WIDTH = 7.0866142

FIGS_ROOT_PATH = "resources"

# Some pdfs are too big! Maybe it is better to use pngs.
DPI = 1000


def plot_elem_dat(dat: np.ndarray, ax, ran=None):
    # In our setting, dat[i, j] means the block arround (x, y) = (i*h, j*h)
    if ran == None:
        posi = ax.imshow(
            dat.T,
            aspect="equal",
            interpolation="none",
            origin="lower",
            extent=(0.0, 1.0, 0.0, 1.0),
        )
    else:
        posi = ax.imshow(
            dat.T,
            aspect="equal",
            interpolation="none",
            origin="lower",
            extent=(0.0, 1.0, 0.0, 1.0),
            vmin=ran[0],
            vmax=ran[1],
        )
    return posi


def plot_node_dat(dat: np.ndarray, ax, ran=None):
    # In our setting, dat[i, j] means the node at (x, y) = (j*h, i*h)
    xx = np.linspace(0.0, 1.0, dat.shape[1])
    yy = np.linspace(0.0, 1.0, dat.shape[0])
    if ran == None:
        posi = ax.pcolormesh(xx, yy, dat, shading="gouraud")
    else:
        posi = ax.pcolormesh(xx, yy, dat, shading="gouraud", vmin=ran[0], vmax=ran[1])
    ax.set_aspect("equal", "box")
    return posi


def append_colorbar(fig, ax, posi):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="5%", pad="2%")
    cbar = fig.colorbar(posi, cax=cax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=SMALL_FONT_SIZE, rotation=15)
    cax.xaxis.set_ticks_position("top")
    return cbar
