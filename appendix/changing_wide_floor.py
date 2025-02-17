from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pypesh.analytic as analytic

parent_dir = Path(__file__).parent

floor_path = parent_dir / "different_floor" / "output"
wide_path = parent_dir / "different_wide" / "output"


floor5 = []
floor10 = []
floor20 = []
for file_path in floor_path.rglob("*.txt"):
    if file_path.is_file():
        with file_path.open("r") as f:
            read = f.read()
            read = read.split("\n")[1]
            read = read.split("\t")
            floor5 += [
                np.array(
                    [
                        float(read[0]),
                        float(read[1]),
                        float(read[2]),
                    ]
                )
            ]
            floor10 += [
                np.array(
                    [
                        float(read[0]),
                        float(read[1]),
                        float(read[3]),
                    ]
                )
            ]
            floor20 += [
                np.array(
                    [
                        float(read[0]),
                        float(read[1]),
                        float(read[4]),
                    ]
                )
            ]
floor5 = np.array(floor5)
floor5 = floor5[np.lexsort((floor5[:, 0], floor5[:, 1]))]

floor10 = np.array(floor10)
floor10 = floor10[np.lexsort((floor10[:, 0], floor10[:, 1]))]

floor20 = np.array(floor20)
floor20 = floor20[np.lexsort((floor20[:, 0], floor20[:, 1]))]


wide5 = []
wide10 = []
wide20 = []
for file_path in wide_path.rglob("*.txt"):
    if file_path.is_file():
        with file_path.open("r") as f:
            read = f.read()
            read = read.split("\n")[1]
            read = read.split("\t")
            wide5 += [
                np.array(
                    [
                        float(read[0]),
                        float(read[1]),
                        float(read[2]),
                    ]
                )
            ]
            wide10 += [
                np.array(
                    [
                        float(read[0]),
                        float(read[1]),
                        float(read[3]),
                    ]
                )
            ]
            wide20 += [
                np.array(
                    [
                        float(read[0]),
                        float(read[1]),
                        float(read[4]),
                    ]
                )
            ]
wide5 = np.array(wide5)
wide5 = wide5[np.lexsort((wide5[:, 0], wide5[:, 1]))]

wide10 = np.array(wide10)
wide10 = wide10[np.lexsort((wide10[:, 0], wide10[:, 1]))]

wide20 = np.array(wide20)
wide20 = wide20[np.lexsort((wide20[:, 0], wide20[:, 1]))]

ball_list = [
    0.8,
    # 0.9,
    0.95,
    # 0.98,
    0.99,
    # 0.995,
    0.998,
    # 0.999,
    # 1,
]

fontsize = 15 * 15 / 14 * (23.8 / 20.9) * (11.34 / 12.23)
plt.rcParams.update({"text.usetex": True, "font.family": "Times"})
fig, axes = plt.subplots(
    1,
    2,
    figsize=(10, 6),
    sharey=True,
    # sharex=True,
    gridspec_kw={
        "width_ratios": [1, 1],
        "wspace": 0,
        "hspace": 0.25,
    },
)

for n, ball in enumerate(ball_list):
    to_plot10 = floor10[floor10[:, 1] == ball]
    to_plot20 = floor20[floor20[:, 1] == ball]

    yargs10 = [
        (sh - analytic.our_approximation(pe, ball)) / sh for pe, ball, sh in to_plot10
    ]
    yargs20 = [
        (sh - analytic.our_approximation(pe, ball)) / sh for pe, ball, sh in to_plot20
    ]

    axes[1].scatter(
        to_plot10[:, 0],
        yargs10,
        color=f"C{n}",
        # label=f"$\\beta={ball}$",
        zorder=0,
    )
    axes[1].scatter(
        to_plot20[:, 0],
        yargs20,
        color=f"C{n}",
        facecolors="w",
        s=10,
        zorder=1,
    )

to_legend_color = []
for n, ball in enumerate(ball_list):
    to_plot10 = wide10[wide10[:, 1] == ball]
    to_plot20 = wide20[wide20[:, 1] == ball]

    yargs10 = [
        (sh - analytic.our_approximation(pe, ball)) / sh for pe, ball, sh in to_plot10
    ]
    yargs20 = [
        (sh - analytic.our_approximation(pe, ball)) / sh for pe, ball, sh in to_plot20
    ]

    to_legend_color += [
        axes[0].scatter(
            to_plot10[:, 0],
            yargs10,
            color=f"C{n}",
            label=f"$\\beta={ball}$",
            zorder=0,
        )
    ]
    axes[0].scatter(
        to_plot20[:, 0],
        yargs20,
        color=f"C{n}",
        facecolors="w",
        s=10,
        zorder=1,
    )

for ax in axes:
    ax.set_xscale("log")
    # ax.set_yscale("log")
    ax.set_xlim(1, 5 * 10**4)
    ax.set_ylim(-0.05, 0.19)
    ax.tick_params(which="both", labelsize=fontsize)

    ax.set_xlabel(r"Peclet number $\textrm{Pe}$", fontsize=fontsize)
axes[0].set_ylabel(
    r"$(\textrm{Sh} - \textrm{Sh}_{\mathrm{f}}) / \textrm{Sh}$", fontsize=fontsize
)

legend_wide10 = axes[0].scatter([0], [0], label=r"$\texttt{wide} = 10$", color="k")
legend_wide20 = axes[0].scatter(
    [0], [0], label=r"$\texttt{wide} = 20$", color="k", facecolors="none", s=10
)

axes[1].scatter([0], [0], label=r"$\texttt{floor} = 10$", color="k")
axes[1].scatter(
    [0], [0], label=r"$\texttt{floor} = 20$", color="k", facecolors="none", s=10
)

second_legend = [legend_wide10, legend_wide20]

legend1 = axes[0].legend(
    fontsize=fontsize, handles=to_legend_color, frameon=False, loc=2
)
legend2 = axes[0].legend(fontsize=fontsize, handles=second_legend, frameon=False, loc=4)

axes[0].add_artist(legend1)
axes[0].add_artist(legend2)
axes[1].legend(fontsize=fontsize, frameon=False, loc=2)

tosave = parent_dir / "graphics/changing_floor_width.pdf"
plt.savefig(
    tosave,
    bbox_inches="tight",
    pad_inches=0.02,
)

plt.show()
