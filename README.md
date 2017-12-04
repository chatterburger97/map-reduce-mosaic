# map-reduce-mosaic
Target image construction from very large image dataset

### known issues
* Dataset needs to be large enough, otherwise some tiles will have no images that map to it and those areas will be blank
* Reading from HDFS is slow. Can only offer a potential advantage over a simple for-loop read-and-write-to-local-disk solution when the dataset is really large

### future work
* use dataframe joins instead of collecting everything as a map
* read from local disk, write to local disk, let spark handle only key value pairs
