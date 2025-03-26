import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from pypesh.mpl.curved_text import CurvedText
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors
import matplotlib.image as mpimg

tableau = list(mcolors.TABLEAU_COLORS)

parent_dir = Path(__file__).parent


image_path = parent_dir / "images" / "marine_aggregates.jpg"
marine_aggregates = mpimg.imread(image_path)
marine_aggregates = marine_aggregates[::-1]

image_path = parent_dir / "images" / "snowflake_small.jpg"
single_snowflake = mpimg.imread(image_path)
single_snowflake = single_snowflake[::-1]

single_dimensions = single_snowflake.shape
aggregates_dimensions = marine_aggregates.shape

fontsize = 11.5 * 4.41 / 2.48 * 7.3 / 4.5 * (47.5 / 52.9)
marker_size = 80
plt.rcParams.update({"text.usetex": True, "font.family": "Times", "savefig.dpi": 700})

figure_aspect_ratio = 300 / 200


fig = plt.figure(
    # constrained_layout=True,
    figsize=(11.7 * figure_aspect_ratio, 11.7),
)
gs = fig.add_gridspec(
    1,
    2,
    wspace=0.05,
    height_ratios=[1],
    width_ratios=[
        (single_dimensions[0] / aggregates_dimensions[0])
        * (aggregates_dimensions[1] / single_dimensions[1]),
        1,
    ],
)

axes_l = fig.add_subplot(gs[0, 0])
axes_r = fig.add_subplot(gs[0, 1])

for image_ax, size in zip([axes_l, axes_r], [aggregates_dimensions, single_dimensions]):
    image_ax.set_xlim(0, size[1])
    image_ax.set_ylim(0, size[0])
    image_ax.set_axis_off()

for image_ax, image in zip(
    [axes_l, axes_r],
    [marine_aggregates, single_snowflake],
):
    image_ax.imshow(
        image,
        aspect="equal",
        # extent=(0, 1, 0, 1),
    )

x_pos = 0.065
y_pos = 0.965

axes_l.text(
    x_pos,
    y_pos,
    r" (a) ",
    ha="center",
    va="center",
    color="w",
    fontsize=fontsize,
    backgroundcolor="#000a",
    transform=axes_l.transAxes,
)

axes_r.text(
    x_pos
    * (
        aggregates_dimensions[1]
        / (single_dimensions[1] * (aggregates_dimensions[0] / single_dimensions[0]))
    ),
    y_pos,
    r" (b) ",
    ha="center",
    va="center",
    color="w",
    fontsize=fontsize,
    backgroundcolor="#000a",
    transform=axes_r.transAxes,
)

tosave = parent_dir / "graphics/marine_aggregates.pdf"
plt.savefig(
    tosave,
    bbox_inches="tight",
    pad_inches=0,
)

# plt.show()
