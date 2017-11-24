from __future__ import print_function

import cv2

import numpy as np

from pyspark import SparkContext

import logging

def tile_mapper():
    print("I run on each tile")


def extract_opencv_tiles():

    def extract_opencv_tiles_nested(imgfile_imgbytes):
        try:
            imgfilename, imgbytes = imgfile_imgbytes
            nparr = np.fromstring(buffer(imgbytes), np.uint8)
            img = cv2.imdecode(nparr, 0)
            print(type(img), "is the type of the ndarray representaion of the image")
            height,width,_ = img.shape
            print(img.shape, " is the shape of the image ndarray")
            return [img]
        except Exception, e:
            logging.exception(e)
            return []

    return extract_opencv_tiles_nested



def main():
    sc = SparkContext(appName="tileMapper")
    print("I do all the input output shit")
    images = sc.binaryFiles("/user/bitnami/project_input/")
    tiles = images.map(extract_opencv_features())
    tiles.foreach(print)
    print(type(tiles))

if __name__ == '__main__':
    main()
