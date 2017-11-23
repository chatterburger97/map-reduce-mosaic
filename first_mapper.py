from __future__ import print_function
import sys
from StringIO import StringIO

from pyspark import SparkContext
import numpy as np
import cv2
from PIL import Image

img_as_array = lambda rawdata: np.asarray(Image.open(StringIO(rawdata)))
get_type = lambda rdd_or_var: type(rdd_or_var)
rgb_nparray_to_cvimg = lambda nparr: cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def main():
    sc = SparkContext(appName="tileMapper")
    imgBinary = sc.binaryFiles("/user/bitnami/allsmiles.jpg")
    img_rgb_matrix = imgBinary.mapValues(img_as_array)
    img_rgb_matrix.collect()

    print("XCXCXCXCXCXCXCXCXCXCXCXCXCXCXCXCXCXCXXCXCXCXCXXCXCXCXCXCXCXCXCXCXCXCXXCXCXCXCXCXCXCXCXCXCXCX")
    img_rgb_matrix.foreach(print)

    # cv_img = img_rgb_matrix.mapValues(rgb_nparray_to_cvimg) # cannot get shape
    print("XCXCXCXCXCXCXCXCXCXCXCXCXCXCXCXCXCXCXXCXCXCXCXXCXCXCXCXCXCXCXCXCXCXCXXCXCXCXCXCXCXCXCXCXCXCX",
    # img_shape(cv_img))
    # img_rgb_matrix.foreach(print)
    # img_rgb_matrix.values().foreach(print)

    sc.stop()

if __name__ == '__main__':
    main()
