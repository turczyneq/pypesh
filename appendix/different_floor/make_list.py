from pathlib import Path
parent_dir = Path(__file__).parent

output_file = parent_dir / "list_to_calculate.txt"


table = [10, 13, 17, 22, 28, 36, 46, 60, 77]

pelist = [0.1, 0.2, 0.5]

for pow in range(-1,5):
    pelist += [el * 10**(pow) for el in table]

to_export = {ball: pelist for ball in [0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.998, 0.999, 1]}

with open(output_file, "w") as f:
    for ball, peclet_list in to_export.items():
        for pe in peclet_list:
            f.write(f"{float(pe):,.2}\t{ball}\n")