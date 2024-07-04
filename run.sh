#!/bin/bash

# Array of peclet values
peclet_values=(1 2 5 10 20 50 100 200 500 1000 2000 5000 10000)

# Loop over each peclet value and run the command
for peclet in "${peclet_values[@]}"; do
    python3 stokes_flow_cylindric_metric.py --peclet $peclet --quiet
done
