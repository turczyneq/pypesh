#!/bin/bash

# Array of peclet values
ball_rad=(1 0.99 0.98 0.95 0.9 0.8 0.5)

# Loop over each peclet value and run the command
for ball in "${ball_rad[@]}"; do
    echo "doing ball ${ball}"
    python3 sh_vs_pe.py --ball $ball --mesh cylinder_stokes_fine_turbo.msh --quiet
done
