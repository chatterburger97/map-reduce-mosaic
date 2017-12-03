from __future__ import print_function
from functools import reduce
import math
import cv2
import numpy as np
import logging
import features
from pyspark import SparkContext
from pyspark.sql import SQLContext, Row


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
            tile_avg_key = stringify(tile_avg)
            tile_coords = [y0, y1, x0, x1]
            tile_avgs.append((tile_avg_key, tile_coords))

    return tile_avgs

# thank you, random dude on SO for teaching me how to use combineByKey
def Combiner(a):    #Turns value a (a tuple) into a list of a single tuple.
    return [a]

def MergeValue(a, b): #a is the new type [(,), (,), ..., (,)] and b is the old type (,)
    a.extend([b])
    return a

def MergeCombiners(a, b): #a is the new type [(,),...,(,)] and so is b, combine them
    a.extend(b)
    return a

flatten = lambda l: [item for sublist in l for item in sublist]


# for turning values that shouldn't be turned into keys, into exactly that
def stringify(vec):
    stuff = map(lambda value_in_vec: str(value_in_vec), vec)
    more_stuff = reduce(lambda x, y: x + "," + y, stuff)
    return more_stuff


def unstringify(strung):
    unstrung_list = strung.split(",")
    result = list(map(int, unstrung_list))
    return result


def extract_opencv_tiles(imgfile_imgbytes, tile_height, tile_width):
    try:
        imgfilename, imgbytes = imgfile_imgbytes
        nparr = np.fromstring(buffer(imgbytes), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        height, width, channels = img.shape
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

    # finds the optimum colour bucket
    bucketcolour_alltilesinbucket_dict = bucketBroadcast.value

    min_distance = 277
    bestbucket = bucketcolour_alltilesinbucket_dict.iterkeys().next() # starts with a random bucket
    bucket_distance_dict  = {}

    for bucketcolour in bucketcolour_alltilesinbucket_dict:
        current_distance = calcEuclideanColourDistance(unstringify(bucketcolour), smallimg_avg)
        bucket_distance_dict[bucketcolour] = current_distance
        if (current_distance < min_distance):
            bestbucket = bucketcolour
            min_distance = current_distance

    # finds a few more buckets
    possible_buckets = []
    for bucketcolour in bucketcolour_alltilesinbucket_dict:
        current_distance = bucket_distance_dict[bucketcolour]
        if (abs(current_distance - min_distance) <= 20 ):
            possible_buckets.append((bucketcolour, list([img_name, current_distance])))
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
    key, valueList = key_valuelist
    return (key, valueList[0])


def main():
    sc = SparkContext(appName="tileMapper")
    print("I do all the input output jazz")
    tile_size = 16

    ###############big image tile averages computed in parallel################
    big_image = sc.binaryFiles("/user/bitnami/project_input/calvin.jpg")
    tile_avgs = big_image.flatMap(lambda x: extract_opencv_tiles(x, tile_size, tile_size)).combineByKey(Combiner, MergeValue, MergeCombiners)
    buckets = tile_avgs.collectAsMap()

    ##########################################################################
    # now this broadcast thing is a problem when the tile size is too small
    broadcastBucket = sc.broadcast(buckets)

    smallimgs = sc.binaryFiles("/user/bitnami/big_dataset_2/", minPartitions=None)
    final_tiles = smallimgs.flatMap(lambda x: get_possible_buckets(x, broadcastBucket)).reduceByKey(closest_avg_in_bucket())
    bucketcolour_smallimgpath_rdd = final_tiles.map(lambda k: return_final_pairs(k))
    bucketcolour_smallimgpath_dict = bucketcolour_smallimgpath_rdd.collectAsMap()

    # future work (use dataframes instead of collecting both as map)  
    # future work : another map-reduce/ way to pass the size of the target image dynamically
    canvas = np.zeros((768, 1024, 3), np.uint8)
    # # assumes buckets is a dictionary (used collectAsMap, got values correctly as RGB, [list of coords])
    for bucketcolour_alltilesinbucket in buckets.iteritems():
        bucket_colour, alltilesinbucket = bucketcolour_alltilesinbucket
        try:
            read_smallimg_rdd = sc.binaryFiles(bucketcolour_smallimgpath_dict[bucket_colour], minPartitions=None)
            read_smallimg = read_smallimg_rdd.take(1)
            file_bytes = np.asarray(bytearray(read_smallimg[0][1]), dtype=np.uint8)
            read_smallimg_rdd.unpersist()

            little_tile = cv2.imdecode(file_bytes, 1)
            for coords in alltilesinbucket:
                y0 = int(coords[0])
                x0 = int(coords[2])
                canvas[y0:y0 + tile_size, x0:x0 + tile_size] = cv2.resize(little_tile, (tile_size, tile_size))
        except KeyError:
            # print("Key not found : " ,bucket_colour)
            continue

    cv2.imwrite('mosaic.jpg', canvas)
    sc.stop()

if __name__ == '__main__':
    main()
