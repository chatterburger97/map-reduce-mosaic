from __future__ import print_function

from StringIO import StringIO

from PIL import Image

import cv2

import numpy as np

from pyspark import SparkContext

img_as_array = lambda rawdata: np.asarray(Image.open(StringIO(rawdata)))
get_type = lambda rdd_or_var: type(rdd_or_var)
rgb_nparray_to_cvimg = lambda nparr: cv2.imdecode(nparr, cv2.IMREAD_COLOR)
print_shape = lambda key, val: print(val.shape)


def shape_of_nparray(nparr):
    print(nparr)


def main():
    sc = SparkContext(appName="tileMapper")
    img_binary = sc.binaryFiles("/user/bitnami/allsmiles.jpg")

    img_rgb_matrix = img_binary.mapValues(img_as_array)

    print("XCXCXCXCXCXCXCXCXCXCXCXCXCXCXCXCXCXCXXCXCXCXCX")
    img_rgb_matrix.foreach(print_shape)
    print("XCXCXCXCXCXCXCXCXCXCXCXCXCXCXCXCXCXCXXCXCXCXC")
    sc.stop()

if __name__ == '__main__':
    main()
