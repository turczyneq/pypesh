depth_size=(11 20 50 80 100 200 500)

# Loop over each mesh value and run the command
for depth in "${depth_size[@]}"; do
    filename="./tests/depth_size$(echo $depth | sed 's/\.//g').msh"
    output="depth_size$(echo $depth | sed 's/\.//g').txt"
    python3 generate_mesh.py --ceiling $depth --quiet --filename $filename
    echo "depth ${depth}:"
    python3 debug.py --quiet --peclet 10000 --mesh $filename --dist2d $output
done