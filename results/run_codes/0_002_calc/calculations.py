import numpy as np
import pypesh.pesh as psh
from pathlib import Path
import time
import jax
import argparse

parser = argparse.ArgumentParser()

parent_dir = Path(__file__).parent

output_file = parent_dir / "test_result.txt"

parser.add_argument("--peclet", type=float, required=True)


parser.add_argument("--ball_radius", type=float, required=True)

args = parser.parse_args()

ball_radius = args.ball_radius
peclet = args.peclet
start = time.time()
sol = psh.all_sherwood(
    peclet,
    ball_radius,
    trials=10000,
    mesh_out=15,
    mesh_jump=20,
    spread=10,
    t_max=800,
    partition=10,
)
end = time.time()


pe_string = f"{int(peclet)}_{int((peclet - int(peclet))*10)}"
ball_string = "0_" + f"{ball_radius}"[2:]
file_to_export = "peclet_" + pe_string + "__ball_" + ball_string + ".txt"
output_file = parent_dir / "output" / file_to_export

with open(output_file, "w") as f:
    f.write(f"\n{peclet}\t{ball_radius}\t{sol[0]}\t{sol[1]}\t{sol[2]}\t{sol[3]}\t{sol[4]}\t{end-start}")