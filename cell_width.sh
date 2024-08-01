width_size=(1.1 2 5 10 20 50)

# Loop over each mesh value and run the command
for width in "${width_size[@]}"; do
    filename="./changing_floor_width/width_size$(echo $width | sed 's/\.//g').msh"
    python3 generate_mesh.py --width $width --quiet --filename $filename
    echo "width ${width}:" >> "./numerical_results/changing_width"
    python3 stokes_flow_cylindric_metric.py --quiet --peclet 10000 --mesh $filename >> "./numerical_results/changing_width"
    printf "\n \n" >> "./numerical_results/changing_width"
done