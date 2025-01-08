from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

floor5 = []
floor10 = []
floor20 = []
for file_path in Path("./output/").rglob("*"):
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

ball_list = [
    0.8,
    # 0.9,
    0.95,
    # 0.98,
    0.99,
    # 0.995,
    0.998,
    # 0.999,
    1,
]

floor5 = np.array(floor5)
floor5 = floor5[np.lexsort((floor5[:, 0], floor5[:, 1]))]

floor10 = np.array(floor10)
floor10 = floor10[np.lexsort((floor10[:, 0], floor10[:, 1]))]

floor20 = np.array(floor20)
floor20 = floor20[np.lexsort((floor20[:, 0], floor20[:, 1]))]


fontsize = 15
plt.figure(figsize=(10, 6))
plt.rcParams.update({"text.usetex": True, "font.family": "Times"})

for n, ball in enumerate(ball_list):
    to_plot10 = floor10[floor10[:, 1] == ball]
    to_plot20 = floor20[floor20[:, 1] == ball]
    plt.scatter(
        to_plot10[:, 0],
        to_plot10[:, 2],
        color=f"C{n}",
        label=f"$\\beta={ball}$",
        zorder=0,
    )
    plt.scatter(
        to_plot20[:, 0],
        to_plot20[:, 2],
        color=f"C{n}",
        facecolors="w",
        s=10,
        zorder=1,
    )

plt.xscale("log")
plt.yscale("log")
plt.xlim(0.5, 10**6)
plt.ylim(0.9, 2 * 10**4)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)

# Labels and Title
plt.xlabel(r"Peclet number $\left(Pe\right)$", fontsize=fontsize)
plt.ylabel(
    r"Modified Sherwood number $\left(\widetilde{Sh}\right)$",
    fontsize=fontsize,
)

plt.scatter([0], [0], label=r"$\it{floor} = 10$", color="k")

plt.scatter([0], [0], label=r"$\it{floor} = 20$", color="k", facecolors="none")

plt.legend(fontsize=fontsize, frameon=False, loc=0)

plt.show()
