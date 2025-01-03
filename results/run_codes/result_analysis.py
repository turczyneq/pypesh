import numpy as np
from pathlib import Path
from itertools import groupby

parent_dir = Path(__file__).parent

calculation_result_path = parent_dir / "calculations" / "test_result.txt"
calculation_result = np.loadtxt(calculation_result_path, delimiter="\t")

appendix_1_path = parent_dir / "0_002_calc" / "test_result.txt"
appendix_1 = np.loadtxt(appendix_1_path, delimiter="\t")

appendix_2_path = parent_dir / "0_001_high_pe_calc" / "test_result.txt"
appendix_2 = np.loadtxt(appendix_2_path, delimiter="\t")

calculation_result = np.vstack((calculation_result, appendix_1))
calculation_result = np.vstack((calculation_result, appendix_2))

calculation_result = calculation_result[calculation_result[:,1].argsort()]

# as there is no numpy groupby, had to use the loop
sorted_data = {}
for radius in set(calculation_result[:,1]):
    working = calculation_result[calculation_result[:,1] == radius]
    sorted_data[radius] = working[working[:,0].argsort()]
    for 