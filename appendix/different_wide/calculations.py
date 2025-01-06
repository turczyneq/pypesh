import numpy as np
import pypesh.fem as fem
from pathlib import Path
import time
import jax
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--peclet", type=float, required=True)


parser.add_argument("--ball_radius", type=float, required=True)

args = parser.parse_args()

parent_dir = Path(__file__).parent
ball_radius = args.ball_radius
peclet = args.peclet
start = time.time()
sol = fem._sherwood_fem_custom_mesh(peclet, ball_radius, width=5), fem._sherwood_fem_custom_mesh(peclet, ball_radius, width=10), fem._sherwood_fem_custom_mesh(peclet, ball_radius, width=20)
end = time.time()

pe_string = f"{int(peclet)}_{int((peclet - int(peclet))*10)}"
ball_string = "0_" + f"{ball_radius}"[2:]
file_to_export = "peclet_" + pe_string + "__ball_" + ball_string + ".txt"
output_file = parent_dir / "output" / file_to_export

with open(output_file, "w") as f:
    f.write(f"\n{peclet}\t{ball_radius}\t{sol[0]}\t{sol[1]}\t{sol[2]}")