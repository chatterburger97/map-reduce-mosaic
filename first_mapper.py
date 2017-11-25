from __future__ import print_function

from fractions import gcd

import scipy

import sklearn.feature_extraction

from sklearn.feature_extraction import image as skimage

import math

import cv2

import numpy as np

import logging

import features

from pyspark import SparkContext


def tile_mapper():

    def tile_mapper_nested(tile_rgb_ndarray):
        tile_as_img = cv2.imdecode(tile_rgb_ndarray, cv2.IMREAD_COLOR);
        return [features.extractFeature(tile_as_img)]

    return tile_mapper_nested

def extract_opencv_tiles():

    def extract_opencv_tiles_nested(imgfile_imgbytes):
        try:
            imgfilename, imgbytes = imgfile_imgbytes
            nparr = np.fromstring(buffer(imgbytes), np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            print(type(img), "is the type of the ndarray representaion of the image")
            # height,width,channels = img.shape
            
            height, width, channels = img.shape
            print (height, width, channels , " is the shape of the image ndarray ")
            square_size = gcd(height, width)
            
            num_patches = int(math.floor(height * width / (4 * 4)))
            
            tiles = skimage.extract_patches_2d(img, (4, 4), max_patches=num_patches, random_state=0)
            # tiles = []
            # return tiles
            print (type(tiles), "is the type of the object returned")

            avg_list = []
            for tile in tiles:
                print(tile, " is a tile")
                print(tile.shape, " is the shape of the tile")
                print(type(tile), " is the type of the tie")
                tile_img = cv2.imdecode(tile, cv2.IMREAD_COLOR)
                print (tile_img, "is what cv read from the tile")
                # print(tile_img)
                curr_avg = features.extractFeature(tile_img)
                # print(curr_avg, " is a feature")
                avg_list.extend([curr_avg])
            return features

        except Exception, e:
            logging.exception(e)
            return

    return extract_opencv_tiles_nested


def main():
    sc = SparkContext(appName="tileMapper")
    print("I do all the input output jazz")
    images = sc.binaryFiles("/user/bitnami/project_input/")
    tile_avgs = images.map(extract_opencv_tiles())

    # averages = tiles.map(tile_mapper())
    # print(averages.take(1))

    # output = tiles.flatMap(tile_mapper())
    # output.foreach(print)
    print(tile_avgs.take(1))

if __name__ == '__main__':
    main()
