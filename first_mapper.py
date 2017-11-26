from __future__ import print_function

from fractions import gcd

import math

import cv2

import numpy as np

import logging

import features

from pyspark import SparkContext

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
            tile_stats = (tile_coords, tile_avg)
            tile_avgs.append(tile_stats)

    return tile_avgs


def extract_opencv_tiles():

    def extract_opencv_tiles_nested(imgfile_imgbytes):
        try:
            imgfilename, imgbytes = imgfile_imgbytes
            nparr = np.fromstring(buffer(imgbytes), np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            print(type(img), "is the type of the ndarray representation of the image")
            # height,width,channels = img.shape
            
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


def main():
    sc = SparkContext(appName="tileMapper")
    print("I do all the input output jazz")
    images = sc.binaryFiles("/user/bitnami/project_input/")
    tile_avgs = images.flatMap(extract_opencv_tiles())

    tile_avgs.collect()
    tile_avgs.foreach(print)

if __name__ == '__main__':
    main()
