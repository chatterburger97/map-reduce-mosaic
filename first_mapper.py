from pyspark import SparkContext
from PIL import Image
import numpy as np


def mapper:
    print "this is the mapper"


def cropper(pilImage):
    print "cropping the image to return an array of bytestrings"


if __name__ == '__main__':
    sc = SparkContext(appName="largeImageTile")

    imageBinary = sc.binaryFiles("/user/bitnami/calvin-being-funny.jpg")
    #

    image_to_array = lambda rawdata: np.asarray(Image.open(StringIO(rawdata)))
    imageBinary.values().map(image_to_array)
    imageBinary.cache()

    # the imageBinary should now be a PIL image
    # chop it up into cropped squares here

    # convert the array of bytestrings into an RDD list
