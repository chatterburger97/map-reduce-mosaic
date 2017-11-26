
# map-reduce-mosaic
Target image construction from very large image dataset

Steps to make match_tiles.py run : 
1) first chmod +x start.sh 
2) then ./start.sh . this will download the target image for you and
   put it onto the hdfs
3) manually download a few small images and put them into the directory 
   /user/bitnami/small_imgs
4) download numpy `pip install numpy --user --no-cache-dir`, to be safe
5) spark-submit match_tiles.py

## note : only dependent on features.py
