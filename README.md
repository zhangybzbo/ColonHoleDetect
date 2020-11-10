# Hole finding pipeline
The pipeline program to detect holes on chunk surface.
Also includes the program to sample virtual points from the holes.

To run the program:
python centerline.py --chuck_dir='dir_to_mesh/' --chuck_ls='name_of_chunks' --center=extremal --closing --opening

The parameters:
`--chuck_dir` the path to where all the mesh located, each mesh should be in a seperate folder
`--chuck_ls` the folder name of each mesh, presented as a list
`--save_dir` the path to saving folder
`--center` the method to anchor centerline, `centroid` if at mesh centroid, `extremal` if at the average of min and max extremal coordinates
`--oversample` whether to oversample theta or not when generating 2D image
`--closing` applying closing in mathematical morphology
`--opening` applying opening in mathematical morphology
`--disc_size` the disc size for mathematical morphology
`--ending` the percentage of end of the chunk to be cut off when computing missing ratio

Some tools to visualize the chunk meshes are in `utils/`.

