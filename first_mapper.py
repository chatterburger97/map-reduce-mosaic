from __future__ import print_function

import cv2

import numpy as np

from pyspark import SparkContext

import logging

def tile_mapper():
    print("I run on each tile")


def extract_opencv_features():

    def extract_opencv_features_nested(imgfile_imgbytes):
        try:
            imgfilename, imgbytes = imgfile_imgbytes
            nparr = np.fromstring(buffer(imgbytes), np.uint8)
            img = cv2.imdecode(nparr, 0)
            print(type(img))
            print(img.shape)
            return [img]
        except Exception, e:
            logging.exception(e)
            return []

    return extract_opencv_features_nested



def main():
    sc = SparkContext(appName="tileMapper")
    print("I do all the input output shit")
    images = sc.binaryFiles("/user/bitnami/project_input/")
    features = images.map(extract_opencv_features())
    features.foreach(print)
    print(type(features))

if __name__ == '__main__':
    main()