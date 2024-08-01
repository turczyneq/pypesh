depth_size=(2 5 10 20 50 100 200)

# Loop over each mesh value and run the command
for depth in "${depth_size[@]}"; do
    filename="./changing_floor_depth/depth_size$(echo $depth | sed 's/\.//g').msh"
    python3 generate_mesh.py --ceiling $depth --quiet --filename $filename
    echo "depth ${depth}:" >> "./numerical_results/changing_depth"
    python3 stokes_flow_cylindric_metric.py --quiet --peclet 10000 --mesh $filename >> "./numerical_results/changing_depth"
    printf "\n \n" >> "./numerical_results/changing_depth"
done