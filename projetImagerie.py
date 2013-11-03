"""
Author : GORMEZ David

Imagery Project: Pattern recognition

"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
from skimage.color  import rgb2hsv,rgb2lab,hsv2rgb
from mpl_toolkits.mplot3d import Axes3D

from scipy.cluster.vq import kmeans,kmeans2,vq

def loadImages(formatChange): 
    if formatChange:
        return changeFormat(plt.imread("./images/AB05.png"))
    else:
        return plt.imread("./images/AB05.png")

def changeFormat(img):
    return (255*img).astype(np.uint8)
    
def convertHSV(img):
    if img.shape[2]==4:
        return rgb2hsv(img[:,:,0:3])
    else:
        if img.shape[2]==3:
            return rgb2hsv(img)
        else:
            print ("Image format not supported")
 
def convertHSVtoRGB(img):
    return hsv2rgb(img)

def scatter3D(centroids):
    # visualizing the centroids into the RGB space
    fig = plt.figure(3)
    ax = Axes3D(fig)
    ax.scatter(centroids[:,0],centroids[:,1],centroids[:,2],c=centroids/255.,s=100)

def convertLAB(img):
    if img.shape[2]==4:
        return rgb2lab(img[:,:,0:3])
    else:
        if img.shape[2]==3:
            return rgb2lab(img)
        else:
            print ("Image format not supported")

def showOnScreen(img):
    plt.Figure()
    plt.imshow(img,interpolation='nearest')

def clustering(img,clusters):
    #Reshaping image in list of pixels to allow kmean Algorithm
    #From 1792x1792x3 to 1792^2x3
    pixels = np.reshape(img,(img.shape[0]*img.shape[1],3))
    print ("pixels in Clustering : ",pixels.dtype,pixels.shape,type(pixels))
    #performing the clustering
    centroids,_ = kmeans(pixels,clusters,iter=3)
    print ("Centroids : ",centroids.dtype,centroids.shape,type(centroids))
    print centroids
    # quantization
    #Assigns a code from a code book to each observation
    #code : A length N array holding the code book index for each observation.
    #dist : The distortion (distance) between the observation and its nearest code.
    code,_ = vq(pixels,centroids)
    print ("Code : ",code.dtype,code.shape,type(code))
    print code

    # reshaping the result of the quantization
    reshaped = np.reshape(code,(img.shape[0],img.shape[1]))
    print ("reshaped : ",reshaped.dtype,reshaped.shape,type(reshaped))

    clustered = centroids[reshaped]
    print ("clustered : ",clustered.dtype,clustered.shape,type(clustered))
    
    #scatter3D(centroids)
    return clustered

def clustering2(img,clusters):
    #Reshaping image in list of pixels to allow kmean Algorithm
    #From 1792x1792x3 to 1792^2x3
    pixels = np.reshape(img,(img.shape[0]*img.shape[1],3))
    centroids,_ = kmeans2(pixels,3,iter=3,minit= 'random')
    print ("Centroids : ",centroids.dtype,centroids.shape,type(centroids))
    print centroids
    # quantization
    #Assigns a code from a code book to each observation
    #code : A length N array holding the code book index for each observation.
    #dist : The distortion (distance) between the observation and its nearest code.
    code,_ = vq(pixels,centroids)
    print ("Code : ",code.dtype,code.shape,type(code))
    print code

    # reshaping the result of the quantization
    reshaped = np.reshape(code,(img.shape[0],img.shape[1]))
    print ("reshaped : ",reshaped.dtype,reshaped.shape,type(reshaped))

    clustered = centroids[reshaped]
    print ("clustered : ",clustered.dtype,clustered.shape,type(clustered))
    
    #scatter3D(centroids)
    return clustered

img = loadImages('false')
print ("Original Image",img.dtype, type(img),img.shape)
print ("pixel test Original = ", img[img.shape[0]/2,img.shape[1]/2,:])

#img = changeFormat(img)

imgHSV = convertHSV(img)
print ("imgHSV : ", imgHSV.dtype, type(imgHSV),imgHSV.shape)
print ("pixel test HSV = ", imgHSV[imgHSV.shape[0]/2,imgHSV.shape[1]/2,:])

clusters = 6
imgClus = convertHSVtoRGB(clustering(imgHSV,clusters))
imgClus2 = convertHSVtoRGB(clustering2(imgHSV,clusters))

"""
kmeanHSV1 = kmeansAlgo(imgHSV)

kmean2 = kmeansAlgo2(img)
kmeanHSV2 = kmeansAlgo2(imgHSV)
"""

#imgLAB = convertLAB(img)    
    
window1 = plt.figure(1)
window1.add_subplot(1,2,1)
plt.title('Original')
plt.imshow(img)
window1.add_subplot(1,2,2)
plt.imshow(imgClus)
plt.title("After Clustering1")

window2= plt.figure(2)
plt.imshow(imgClus2)
plt.title("After Clustering2")
plt.show()
