#!/bin/bash

# Array of peclet values
peclet_values=(0.1  0.125  0.2  0.3  0.5  0.7  1  1.25  2  3  5  7  10  12.5  20  30  50  70  100  125  200  300  500  700  1000  1250  2000  3000  5000  7000  10000 12500  20000  30000  50000  70000  100000  125000  200000  300000  500000  700000  1000000  1250000  2000000  3000000  5000000  7000000)

# Loop over each peclet value and run the command
for peclet in "${peclet_values[@]}"; do
    echo "doing peclet${peclet}"
    python3 stokes_flow_cylindric_metric.py --peclet $peclet --quiet >> "./numerical_results/sh(pe)_rsyf0"
done
