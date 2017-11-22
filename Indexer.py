import glob
import json
import cv2
import ntpath
import os

import features


from numpy import array
from math import sqrt
from pyspark.mllib.clustering import KMeans
from pyspark import SparkContext

DB_PATH = "./images"
INDEX_PATH = "./index/"
LENGTH = 50

sc = SparkContext()

def convertImage(path):
    image = cv2.imread(path)
    height, width, depth = image.shape
    ratio = float(height)/float(width)
    print (height, width, ratio)
    if height > width:
        image = cv2.resize(image, (LENGTH, int(LENGTH*ratio))) # width, height
        # image is now LENGTHpx width, now we crop it to the correct height
        h = int(LENGTH*ratio); # the final height
        margin = int(float(h-LENGTH)/float(2)) # margin to crop at top and bottom
        image = image[margin:(LENGTH + margin), 0:LENGTH] # crop image now
    else:
        image = cv2.resize(image, (int(LENGTH/ratio), LENGTH))
        # image is now LENGTHpx height, now we crop it to the correct width
        w = int(LENGTH/ratio); # final width
        margin = int(float(w-LENGTH)/float(2)) # calculate margin
        image = image[0:LENGTH, margin:(LENGTH+margin)] # crop image
    # finally store image
    cv2.imwrite(INDEX_PATH + ntpath.basename(path), image)
    return image

def cluster_img(index):
    clusters = KMeans.train(sc.parallelize(index),3,maxIterations=10,initializationMode="random",seed=50)
    
    def error(point):
        center = clusters.centers[clusters.predict(point)]
        return sqrt(sum([x**2 for x in (point - center)]))
    
    return clusters

def getFileList():
    included_extenstions = ['jpg','bmp','png','gif' ] ;
    return [fn for fn in os.listdir(DB_PATH) if any([fn.endswith(ext) for ext in included_extenstions])];

def main():
    index = []
    if not os.path.exists(INDEX_PATH):
        os.makedirs(INDEX_PATH)
    files = glob.glob(DB_PATH + "/" + "*.jpg")
    
    entry = []
    fileName = []
    
    for file in files:
        print "Processing file: " + file
        image = convertImage(file)
        entry = features.extractFeature(image)
        print(entry)
        #entry["file"] = ntpath.basename(file)
        index.append(entry)
        fileName.append(ntpath.basename(file))
        
    trained_model = cluster_img(index)
    
    print ("Train Model COMPLETED! Showing the result..")
    for mCenter in trained_model.centers:
        print(mCenter)
    print ("Now Saving the K Measns Model")
    trained_model.save(sc, "myModelPath")

    #with open(INDEX_PATH + "histogram.index", 'w') as outfile:
    #json.dump(index, outfile, indent=4)
    
    #rdd = sc.parallelize(index)
    #print rdd.collect()
    #rdd.saveAsTextFile("historgam")
    
    i = 0
    

    text_file = open("images_cluster.txt", "w")
    for image in index:
        result = trained_model.predict(image)
        text_file = open("images_cluster.txt", "a")
        text_file.write(fileName[i]+ ":" +str(result)+"\n")
        text_file.close()
        print(fileName[i]+ ":" +str(result))
        i+=1
    
        
    print ("Index written to: " + INDEX_PATH + "histogram.index")

if __name__ == "__main__":
    main()
