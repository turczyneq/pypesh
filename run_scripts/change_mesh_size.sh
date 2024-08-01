mesh_size=(1 0.5 0.4 0.3 0.2 0.1 0.07 0.05 0.02 0.01 0.005 0.002 0.001)

# Loop over each mesh value and run the command
for mesh in "${mesh_size[@]}"; do
    filename="./changing_mesh_size/mesh_size$(echo $mesh | sed 's/\.//g').msh"
    python3 generate_mesh.py --mesh $mesh --quiet --filename $filename
    echo "mesh size ${mesh}:" >> "./numerical_results/changing_mesh_size"
    python3 stokes_flow_cylindric_metric.py --quiet --peclet 10000 --mesh $filename >> "./numerical_results/changing_mesh_size"
    printf "\n \n" >> "./numerical_results/changing_mesh_size"
done