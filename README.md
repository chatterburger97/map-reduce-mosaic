# map-reduce-mosaic
Target image construction from very large image dataset

### known issues
* Dataset needs to be large enough, otherwise some tiles will have no images that map to it and those areas will be blank
*  Broadcasting all the tile colour- tile coordinate pairs (even grouped by tile colour) could be a potential bottleneck
* Reading from HDFS is slow. Can only offer a potential advantage over a simple for-loop read-and-write-to-local-disk solution when the dataset is really large

### future work
* patch thought of for the blank area error (another map-reduce to get null values and do something with them)
* use dataframe joins instead of collecting everything as a map
