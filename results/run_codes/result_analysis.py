from pathlib import Path
import numpy as np

parent_dir = Path(__file__).parent
calculations_dir = parent_dir / "calculations" / "output"
dir_0_002 = parent_dir / "0_002_calc" / "output"
dir_0_001 = parent_dir / "0_001_high_pe_calc" / "output"

fem_result = []
trajectory_result = []
for file_path in calculations_dir.rglob("*"):
    if file_path.is_file():
        with file_path.open("r") as f:
            read = f.read()
            read = read.split("\n")[1]
            read = read.split("\t")
            fem_result += [
                np.array(
                    [
                        float(read[0]),
                        float(read[1]),
                        float(read[4]),
                    ]
                )
            ]
            trajectory_result += [
                np.array(
                    [
                        float(read[0]),
                        float(read[1]),
                        float(read[6]),
                    ]
                )
            ]

for file_path in dir_0_002.rglob("*"):
    if file_path.is_file():
        with file_path.open("r") as f:
            read = f.read()
            read = read.split("\n")[1]
            read = read.split("\t")
            fem_result += [
                np.array(
                    [
                        float(read[0]),
                        float(read[1]),
                        float(read[4]),
                    ]
                )
            ]
            trajectory_result += [
                np.array(
                    [
                        float(read[0]),
                        float(read[1]),
                        float(read[6]),
                    ]
                )
            ]

for file_path in dir_0_001.rglob("*"):
    if file_path.is_file():
        with file_path.open("r") as f:
            read = f.read()
            read = read.split("\n")[1]
            read = read.split("\t")
            fem_result += [
                np.array(
                    [
                        float(read[0]),
                        float(read[1]),
                        float(read[4]),
                    ]
                )
            ]
            trajectory_result += [
                np.array(
                    [
                        float(read[0]),
                        float(read[1]),
                        float(read[6]),
                    ]
                )
            ]

fem_result = np.array(fem_result)
fem_result = fem_result[
    np.lexsort(
        (
            fem_result[:, 0],
            fem_result[:, 1],
        )
    )
]

trajectory_result = np.array(trajectory_result)
trajectory_result = trajectory_result[
    np.lexsort(
        (
            trajectory_result[:, 0],
            trajectory_result[:, 1],
        )
    )
]

parent_parent_dir = parent_dir.parent

np.savetxt(parent_parent_dir / "data" / "fem_pe_vs_sh.csv", fem_result, delimiter=",")

np.savetxt(parent_parent_dir / "data" / "py_pe_vs_sh.csv", trajectory_result, delimiter=",")
