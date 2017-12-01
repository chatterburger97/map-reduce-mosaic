from __future__ import print_function
from functools import reduce
import math
import cv2
import numpy as np
import logging
import features
from pyspark import SparkContext


def calcEuclideanColourDistance(rgblist1, rgblist2):
    sumSqrts = 0
    sumSqrts += math.pow((rgblist1[0] - rgblist2[0]), 2)
    sumSqrts += math.pow((rgblist1[1] - rgblist2[1]), 2)
    sumSqrts += math.pow((rgblist1[2] - rgblist2[2]), 2)
    return math.sqrt(sumSqrts)


def compute_tile_avgs(cvimg, tile_height, tile_width):
    numrows = int(cvimg.shape[0]/tile_height)
    numcols = int(cvimg.shape[1]/tile_width)

    tile_avgs = []

    for row in range(0, numrows):
        for col in range(0, numcols):
            y0 = row * tile_height
            y1 = y0 + tile_height
            x0 = col * tile_width
            x1 = x0 + tile_width
            tile_avg = features.extractFeature(cvimg[y0:y1, x0:x1])
            tile_coords = [y0, y1, x0, x1]
            tile_stats = [tile_coords, tile_avg]
            tile_avgs.append(tile_stats)

    return tile_avgs

flatten = lambda l: [item for sublist in l for item in sublist]


# for turning values that shouldn't be turned into keys, into exactly that. hashtag rebel
def stringify(vec):
    stuff = map(lambda value_in_vec: str(value_in_vec), vec)
    more_stuff = reduce(lambda x, y: x + "," + y, stuff)
    return more_stuff


def extract_opencv_tiles(imgfile_imgbytes, tile_height, tile_width):
    try:
        imgfilename, imgbytes = imgfile_imgbytes
        nparr = np.fromstring(buffer(imgbytes), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # print(type(img), "is the type of the ndarray representation of the image")
        height, width, channels = img.shape
        # print(height, width, channels, " is the shape of the image ndarray ")
        return compute_tile_avgs(img, tile_height, tile_width)

    except e:
        logging.exception(e)
        return []


def get_possible_buckets(imgfile_imgbytes, bucketBroadcast):
    # decode small img and get features
    img_name, img_bytes = imgfile_imgbytes
    file_bytes = np.asarray(bytearray(img_bytes), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    smallimg_avg = features.extractFeature(img)

    # finds the optimum tile
    buckets = bucketBroadcast.value
    min_distance = 277
    bucket_distance_dict = {}

    for bucket in buckets:
        bucket_coords, bucket_colour = bucket
        distance_from_smallimg_avg = calcEuclideanColourDistance(smallimg_avg, bucket_colour)
        bucket_colour_key = stringify(bucket_colour)
        bucket_distance_dict[bucket_colour_key] = [distance_from_smallimg_avg, bucket_coords]
        if distance_from_smallimg_avg < min_distance:
            min_distance = distance_from_smallimg_avg

    # finds a few more tiles that are closer so that each small image can map to a range of colours of tiles instead of just 1
    possible_buckets = []
    for bucket_colour_key, distance_coords in bucket_distance_dict.iteritems():
        if(abs(distance_coords[0] - min_distance) <= 20):
            possible_buckets.append((bucket_colour_key, list([img_name, distance_coords[0], distance_coords[1]])))
    # possible_buckets.append((current_bucket_colour_key, list([img_name,distance, current_bucket_coords])))

    return possible_buckets


def closest_avg_in_bucket():
    def closest_avg_in_bucket_nested(a, b):
        if(a[1] < b[1]):
            return a
        else:
            return b
    return closest_avg_in_bucket_nested


def return_bytearray(imgfile_imgbytes):
    file_bytes = np.asarray(bytearray(imgfile_imgbytes[1]), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return img

def return_final_pairs(key_valuelist):
    smallimgpath_distance_tilecoords = key_valuelist[1]
    smallimgpath, distance, tilecoords = smallimgpath_distance_tilecoords
    return (stringify(tilecoords), smallimgpath)

def find_smallimg_path(tilecoords_imgpath_dict, coordList):
    coordkey = stringify(coordList)
    if coordkey in tilecoords_imgpath_dict:
        return tilecoords_imgpath_dict[coordkey]
    return


def main():
    sc = SparkContext(appName="tileMapper")
    print("I do all the input output jazz")

    ###############big image tile averages computed in parallel################
    big_image = sc.binaryFiles("/user/bitnami/project_input/calvin.jpg")
    tile_avgs = big_image.flatMap(lambda x: extract_opencv_tiles(x, 256, 256))
    buckets = tile_avgs.collect()

    ##########################################################################
    broadcastBucket = sc.broadcast(buckets)  # this isn't too big
    smallimgs = sc.binaryFiles(
        "/user/bitnami/small_dataset/", minPartitions=None)
    final_tiles_mapped = smallimgs.flatMap(lambda x: get_possible_buckets(x, broadcastBucket)) # expected (bucket_colour_key, list([img_name, bucket_distance_dict[bucket_colour_key], current_bucket_coords]))
    final_tiles_reduced = final_tiles_mapped.reduceByKey(closest_avg_in_bucket())

    tilecoords_imgpath_rdd = final_tiles_reduced.map(lambda k: return_final_pairs(k))

    tilecoords_imgpath_dict = tilecoords_imgpath_rdd.collectAsMap()

    # reconstruct final image
    flatbuckets = tile_avgs.map(flatten).collect()

    current_row = None
    no_of_row = 0
    no_of_col = 0
    tile_size = flatbuckets[0][1]

    for tile in flatbuckets:
        if tile[0] == current_row:
            no_of_col += 1
        else:
            current_row = tile[0]
            no_of_col = 1
            no_of_row += 1
            current_row = tile[0]
    canvas = np.zeros((no_of_row * tile_size, no_of_col * tile_size, 3), np.uint8)


    ## Put small images into the big image canvas
    for coord_key, smallimg_path in tilecoords_imgpath_dict.iteritems():
        coords = coord_key.split(",")
        y0 = int(coords[0])
        x0 = int(coords[2])

        read_smallimg_rdd = sc.binaryFiles(smallimg_path, minPartitions=None)
        read_smallimg = read_smallimg_rdd.take(1)
        file_bytes = np.asarray(bytearray(read_smallimg[0][1]), dtype=np.uint8)

        read_smallimg_rdd.unpersist()

        little_tile = cv2.imdecode(file_bytes, 1)
        canvas[y0:y0 + tile_size, x0:x0 + tile_size] = cv2.resize(little_tile, (256, 256))

    cv2.imwrite('mosaic.jpg', canvas)
    sc.stop()

if __name__ == '__main__':
    main()
