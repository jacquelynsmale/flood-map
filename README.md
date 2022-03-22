# flood-map
Space to work on flood depth estimation maps. 
`flood_map.py` is the main script to estimate flood depths. It imports functions from `util.py`, which is an updated version of `from_OSL/convience.py`, and `new_functions.py`, which are new functions I pulled from `from_OSL/main_OSL.py`. 

The goal is to get everything in `main.py`, which will parse arguments handed through the command line. 

Data are to be uploaded to a S3 bucket (too big for GitHub). Need a water extent tif and a HAND raster for calculation. 
