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

def stringify(vec):
    # print(vec, "THIS MUST BE ITERABLE, IT SEEMS")
    stuff =  map(lambda value_in_vec: str(value_in_vec), vec)
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

    except Exception, e:
        logging.exception(e)
        return []


def return_avgs(imgfile_imgbytes, bucketBroadcast):
    img_name, img_bytes = imgfile_imgbytes
    file_bytes = np.asarray(bytearray(img_bytes), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    min_distance = 277
    small_img_avg = features.extractFeature(img)
    buckets = bucketBroadcast.value
    best_colour_bucket = buckets[0]

    bucket_distance_dict = {}

    # finds the optimum tile
    for bucket in buckets:
        current_distance = calcEuclideanColourDistance(
            bucket[1], small_img_avg)
        if (current_distance < min_distance):
            min_distance = current_distance
            best_colour_bucket = bucket
            bucket_colour_vec = map(lambda vec: vec[1], best_colour_bucket)
            bucket_colour_key = stringify(bucket_colour_vec)
            bucket_distance_dict[bucket_colour_key] = current_distance

    # finds a few more tiles that are closer so that each small image can map to a range of colours of tiles instead of just 1
    possible_buckets = []
    for bucket in buckets:
        current_bucket_colour = bucket[1]
        current_bucket_colour_key = stringify(current_bucket_colour)
        if (current_bucket_colour_key in bucket_distance_dict and abs(bucket_distance_dict[current_bucket_colour_key] - min_distance) <= 20):
            possible_buckets.append((bucket_colour_key, 
                                                list([img_name,\
                                                bucket_distance_dict[bucket_colour_key],\
                                                bucket[0]])))

    return possible_buckets


def return_closest_avg():
    def return_closest_avg_nested(a, b):
        if(a[1] < b[1]):
            return a
        else:
            return b
    return return_closest_avg_nested


def return_final_pairs(k):
    val = k[1]
    tile_coordinates_vec = map(lambda vec: vec[2], k)
    tile_coordinates_key = stringify(tile_coordinates_key)
    return (tile_coordinates_key, val[0])


def return_bytearray(imgfile_imgbytes):
    file_bytes = np.asarray(bytearray(imgfile_imgbytes[1]), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return img


def find_smallimg_path(img_tile_pairs, coordList):
    key = stringify(coordList[:4])
    if key in img_tile_pairs:
        return img_tile_pairs[key]
    return


def main():
    sc = SparkContext(appName="tileMapper")
    print("I do all the input output jazz")

    ###############big image tile averages computed in parallel################
    big_image = sc.binaryFiles("/user/bitnami/project_input/calvin.jpg")
    tile_avgs = big_image.flatMap(lambda x: extract_opencv_tiles(x, 16, 16))
    tile_avgs.persist()
    buckets = tile_avgs.collect()

    ##########################################################################
    broadcastBucket = sc.broadcast(buckets)
    small_imgs = sc.binaryFiles("/user/bitnami/small_imgs", minPartitions=None)
    final_tiles = small_imgs.flatMap(lambda x: return_avgs(x, broadcastBucket)).reduceByKey(return_closest_avg())

    img_tile_pairs = final_tiles.map(lambda k: return_final_pairs(k))
    img_tile_pairs_dict = img_tile_pairs.collectAsMap()
    # print(img_tile_pairs_dict)

    # reconstruct final image
    flatbuckets = tile_avgs.map(flatten).collect()
    readyToCombine = {}
    currentRow = None
    noOfRow = 0
    noOfCol = 0
    tile_size = flatbuckets[0][1]

    for tile in flatbuckets:
        if tile[0]==currentRow:
            noOfCol = noOfCol + 1
        else:
            currentRow = tile[0]
            noOfCol = 1
            noOfRow = noOfRow + 1
            currentRow = tile[0]

        correct_img = find_smallimg_path(img_tile_pairs_dict, [tile[0], tile[1], tile[2], tile[3]])

        if correct_img is not None:  # will return the path of the image
            stringified_tile_coordinates = str(tile[0]) + "," + str(tile[1]) + "," + str(tile[2]) + "," + str(tile[3])
            readyToCombine[stringified_tile_coordinates] = correct_img

    # # Put small images into the big image canvas
    canvas = np.zeros((noOfRow*tile_size, noOfCol*tile_size, 3), np.uint8)
    for key, value in readyToCombine.iteritems():
        coords = key.split(",")
        y0 = int(coords[0])
        x0 = int(coords[2])
        read_small_img_rdd = sc.binaryFiles(value, minPartitions=None)
        read_small_img = read_small_img_rdd.take(1)
        file_bytes = np.asarray(bytearray(read_small_img[0][1]), dtype=np.uint8)
        read_small_img_rdd.unpersist()
        R = cv2.imdecode(file_bytes, 1)
        # print(y0, y1, x0, x1)
        # print((canvas[y0:y0 + tile_size, x0:x0 + tile_size]).shape)
        canvas[y0:y0 + tile_size, x0:x0 + tile_size] = cv2.resize(R, (16, 16))

    cv2.imwrite('mosaic.jpg', canvas)
    sc.stop()

if __name__ == '__main__':
    main()
