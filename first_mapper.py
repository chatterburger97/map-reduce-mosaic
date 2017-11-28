from __future__ import print_function

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


def extract_opencv_tiles(imgfile_imgbytes, tile_height, tile_width):
    try:
        imgfilename, imgbytes = imgfile_imgbytes
        nparr = np.fromstring(buffer(imgbytes), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        print(type(img), "is the type of the ndarray representation of the image")

        height, width, channels = img.shape
        print(height, width, channels, " is the shape of the image ndarray ")
        return compute_tile_avgs(img, tile_height, tile_width)

    except Exception, e:
        logging.exception(e)
        return []


def return_avgs(imgfile_imgbytes, bucketBroadcast):
    file_bytes = np.asarray(bytearray(imgfile_imgbytes[1]), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    min_distance = 277
    small_img_avg = features.extractFeature(img)
    buckets = bucketBroadcast.value
    dominant_colour_bucket = buckets[0]

    for bucket in buckets:
        current_distance = calcEuclideanColourDistance(
            bucket[1], small_img_avg)
        if (current_distance < min_distance):
            min_distance = current_distance
            dominant_colour_bucket = bucket
    # # return tile/bucket's average color, (name of current small img, distance to bucket,
    # #.. average of current small image, coordinates of tile (for the final stitch step))
    stringified_bucket_colour = str(dominant_colour_bucket[1][0]) + "," + str(
        dominant_colour_bucket[1][1]) + "," + str(dominant_colour_bucket[1][2])
    return (stringified_bucket_colour, list([imgfile_imgbytes[0], min_distance, dominant_colour_bucket[0]]))


def return_closest_avg():
    def return_closest_avg_nested(a, b):
        if(a[1] < b[1]):
            return a
        else:
            return b
    return return_closest_avg_nested


def return_final_pairs(k):
    val = k[1]
    stringified_tile_coordinates = str(
        val[2][0]) + "," + str(val[2][1]) + "," + str(val[2][2]) + "," + str(val[2][3])
    return (stringified_tile_coordinates, val[0])


def return_bytearray(imgfile_imgbytes):
    file_bytes = np.asarray(bytearray(imgfile_imgbytes[1]), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return img


def find_smallimg_path(img_tile_pairs, coordList):
    key = str(coordList[0]) + "," + str(coordList[1]) + \
        "," + str(coordList[2]) + "," + str(coordList[3])
    if key in img_tile_pairs:
        return img_tile_pairs[key]
    return


def main():
    sc = SparkContext(appName="tileMapper")
    print("I do all the input output jazz")

    ###############big image tile averages computed in parallel################
    big_image = sc.binaryFiles("/user/bitnami/project_input/calvin.jpg")
    tile_avgs = big_image.flatMap(lambda x: extract_opencv_tiles(x, 32, 32))
    tile_avgs.persist()
    buckets = tile_avgs.collect()

    ##########################################################################
    broadcastBucket = sc.broadcast(buckets)
    small_imgs = sc.binaryFiles("/user/bitnami/small_imgs", minPartitions=None)
    final_tiles = small_imgs.map(lambda x: return_avgs(
        x, broadcastBucket)).reduceByKey(return_closest_avg())

    img_tile_pairs = final_tiles.map(lambda k: return_final_pairs(k))
    img_tile_pairs_dict = img_tile_pairs.collectAsMap()
    img_tile_pairs.foreach(print)

    # reconstruct final image
    flatbuckets = tile_avgs.map(flatten).collect()
    readyToCombine = []
    currentRow = None
    noOfRow = 0
    noOfCol = 0
    firstTile = flatbuckets[0]
    tileSize = flatbuckets[0][1]

    for tile in flatbuckets:
        if tile[0] == currentRow:
            correct_img = find_smallimg_path(img_tile_pairs_dict, [tile[0], tile[1], tile[
                                             2], tile[3]])  # will return the path of the image
            if correct_img is not None:
                readyToCombine.append(correct_img)
            noOfCol = noOfCol + 1
        else:
            currentRow = tile[0]
            noOfCol = 1
            noOfRow = noOfRow + 1
            currentRow = tile[0]
            correct_img = find_smallimg_path(
                img_tile_pairs_dict, [tile[0], tile[1], tile[2], tile[3]])
            if correct_img is not None:
                readyToCombine.append(correct_img)

    # # Put small images into the big image canvas
    canvas = np.zeros((noOfRow*tileSize, noOfCol*tileSize, 3), np.uint8)
    count = 0
    for i in range(0, noOfRow):
        for j in range(0, noOfCol):
            try:
                read_small_img = sc.binaryFiles(
                    readyToCombine[count], minPartitions=None).take(1)
                file_bytes = np.asarray(
                    bytearray(read_small_img[0][1]), dtype=np.uint8)
                R = cv2.imdecode(file_bytes, 1)
                canvas[i * tileSize:(i + 1) * tileSize, j * tileSize:(j + 1)
                       * tileSize] = cv2.resize(R, (32, 32))
                count = count + 1
            except:
                break

    cv2.imwrite('mosaic.jpg', canvas)
    sc.stop()

if __name__ == '__main__':
    main()
