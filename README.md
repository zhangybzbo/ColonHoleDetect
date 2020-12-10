# Hole finding pipeline
The pipeline program to detect holes on chunk surfaces. Giving a list of chunks, the program will give the relative areas of surface and every missing region, and the missing ratio of each chunk.

Also includes the program to sample virtual points from the holes. The coordinates of virtual points in different missing regions will be saved separately in the generated `.pkl` file.

To run the program:
```
python centerline.py --chunk_dir='dir_to_mesh/' --center=extremal --closing --opening --chunk_ls 'chunk1' 'chunk2' 'chunk3'
```

The parameters:
* `--chunk_dir` the path to where all the meshes locate, each mesh should be in a seperate folder 
* `--chunk_ls` the folder name list of all meshes
* `--save_dir` the path to the saving folder
* `--center` the method to anchor centerline, `centroid` if at mesh centroid, `extremal` if at the average of min and max extremal coordinates
* `--oversample` whether to oversample theta or not when generating 2D image
* `--closing` applying closing in mathematical morphology
* `--opening` applying opening in mathematical morphology
* `--disc_size` the default disc size for mathematical morphology
* `--ending` the percentage of the end of the chunk to be cut off when computing missing ratio

Some tools to visualize the chunk meshes are in `utils/`.

