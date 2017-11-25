from __future__ import print_function

from fractions import gcd

import scipy

import sklearn.feature_extraction

from sklearn.feature_extraction import image as skimage

import math

import cv2

import numpy as np

from pyspark import SparkContext

import logging

def tile_mapper():

    def tile_mapper_nested(tile_rgb_ndarray):
        return [tile_rgb_ndarray]

    return tile_mapper_nested

def extract_opencv_tiles():

    def extract_opencv_tiles_nested(imgfile_imgbytes):
        try:
            imgfilename, imgbytes = imgfile_imgbytes
            nparr = np.fromstring(buffer(imgbytes), np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            print(type(img), "is the type of the ndarray representaion of the image")
            # height,width = img.shape
            print(img.shape, " is the shape of the image ndarray")
            height, width, channels = img.shape
            print (height, width, channels , " is the shape of the image ndarray ")
            square_size = gcd(height, width)
            print(square_size, " is the square edge size")
            num_patches = int(math.floor(height * width / (4 * 4)))
            print(type(num_patches) , "is the type of the num_patches variable")
            tiles = skimage.extract_patches_2d(img, (4, 4), max_patches=num_patches, random_state=0)
            # tiles = []
            # return tiles
            print (type(tiles), "is the type of the object returned")
            return tiles
        except Exception, e:
            logging.exception(e)
            return []

    return extract_opencv_tiles_nested


def main():
    sc = SparkContext(appName="tileMapper")
    print("I do all the input output jazz")
    images = sc.binaryFiles("/user/bitnami/project_input/")
    tiles = images.map(extract_opencv_tiles())
    output = tiles.flatMap(tile_mapper())
    # output.foreach(print)
    print(output.take(1))

if __name__ == '__main__':
    main()
