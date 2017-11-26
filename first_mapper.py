from __future__ import print_function

from fractions import gcd

import math

import cv2

import numpy as np

import logging

import features

from pyspark import SparkContext

from PIL import Image


global buckets

def calcEuclideanColourDistance(rgblist1, rgblist2):
    sumSqrts = 0
    sumSqrts+=math.pow((rgblist1[0] - rgblist2[0]), 2)
    sumSqrts+=math.pow((rgblist1[1] - rgblist2[1]), 2)
    sumSqrts+=math.pow((rgblist1[2] - rgblist2[2]), 2)
    return math.sqrt(sumSqrts)

def compute_tile_avgs(cvimg):
    block_width = gcd(cvimg.shape[0], cvimg.shape[1])
    block_height = block_width

    numrows = int(cvimg.shape[0]/block_height)
    numcols = int(cvimg.shape[1]/block_width)

    tile_avgs = []

    for row in range(0, numrows):
        for col in range(0, numcols):
            y0 = row * block_height
            y1 = y0 + block_height
            x0 = col * block_width
            x1 = x0 + block_width
            tile_avg = features.extractFeature(cvimg[y0:y1, x0:x1])
            tile_coords = [y0, y1, x0, x1]
            tile_stats = [tile_coords, tile_avg]
            tile_avgs.append(tile_stats)

    # print(buckets)
    return tile_avgs


def extract_opencv_tiles():
    def extract_opencv_tiles_nested(imgfile_imgbytes):
        try:
            imgfilename, imgbytes = imgfile_imgbytes
            nparr = np.fromstring(buffer(imgbytes), np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            print(type(img), "is the type of the ndarray representation of the image")
            
            height, width, channels = img.shape
            print (height, width, channels , " is the shape of the image ndarray ")
            square_size = gcd(height, width)

            img = cv2.resize(img, (width / square_size * square_size, height / square_size * square_size))
            print (img.shape , " is the shape of the resized image ndarray ")
            return compute_tile_avgs(img)

        except Exception, e:
            logging.exception(e)
            return []

    return extract_opencv_tiles_nested


def return_avgs(imgfile_imgbytes, bucketBroadcast):
    file_bytes = np.asarray(bytearray(imgfile_imgbytes[1]), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    min_distance = 277
    small_img_avg = features.extractFeature(img)
    buckets = bucketBroadcast.value
    # print(buckets)
    dominant_colour_bucket = buckets[0]
    for bucket in buckets:
        current_distance = calcEuclideanColourDistance(bucket[1], small_img_avg)
        print(current_distance, " I AM HERE")
        if (current_distance < min_distance):
            min_distance = current_distance
            dominant_colour_bucket = bucket # has the coordinates of the tile of the dominant colour bucket
    # # return tile/bucket's average color, (name of current small img, distance to bucket,
    # #.. average of current small image, coordinates of tile (for the final stitch step))
    return (dominant_colour_bucket[1], (imgfile_imgbytes[0], min_distance, features.extractFeature(img), dominant_colour_bucket[0]))
    # return buckets.value[0]
    # return small_img_avg


def return_closest_avg():
    def return_closest_avg_nested(a, b):
        print(a)
        if(a[1][1] < b[1][1]) :
            return a
        else :
            return b

    return return_closest_avg_nested


def main():
    sc = SparkContext(appName="tileMapper")
    print("I do all the input output jazz")

    ###########################################################################
    big_image = sc.binaryFiles("/user/bitnami/project_input/calvin.jpg")
    tile_avgs = big_image.flatMap(extract_opencv_tiles())
    buckets = tile_avgs.collect()
    tile_avgs.map(lambda l: [item for sublist in l for item in sublist]).saveAsTextFile("/user/bitnami/project_output/buckets.txt")
    ############################################################################

    broadcastBucket = sc.broadcast(buckets)
    small_imgs = sc.binaryFiles("/user/bitnami/small_imgs", minPartitions=None)
    imgname_imgavg_arrays = small_imgs.map(lambda x: return_avgs(x, broadcastBucket))
    # print(imgname_imgavg_arrays.take(1), " HIIIIIIIIIIIIIIIIIIIIIIELLOOOO")
    imgname_imgavg_arrays = imgname_imgavg_arrays.reduce(return_closest_avg())
    print(imgname_imgavg_arrays)
    # imgname_imgavg_arrays.saveAsTextFile("/user/bitnami/project_output/to_put_in_buckets.txt")

    sc.stop()


if __name__ == '__main__':
    main()
