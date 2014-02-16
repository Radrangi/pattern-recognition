"""
Author : GORMEZ David

Imagery Project: Pattern recognition

"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
from skimage.color  import rgb2hsv,rgb2lab,hsv2rgb
from skimage.morphology  import closing,opening,square
from scipy.ndimage.morphology import distance_transform_edt

from mpl_toolkits.mplot3d import Axes3D
from scipy.cluster.vq import kmeans,kmeans2,vq
import math

class Circle:
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius
    
    def __init__(self, center, radius,color):
        self.center = center
        self.radius = radius
        self.color = color
        
    def area(self):
        return math.pi * self.radius**2
    
    def contains(self, point):
        dx = point[0] - self.center[0]
        dy = point[1] - self.center[1]
        #print ("x courant =",math.sqrt(dx*dx + dy*dy),"Rayon=",self.radius)
        return (math.sqrt(dx*dx + dy*dy) <= self.radius) 
    
    def getradius(self):
        return self.radius
    def getcenter(self):
        return self.center
    def getcolor(self):
        return self.color
    def setradius(self, r):
        self.radius = r
    def setcenter(self, c):
        self.center = c

def loadImages(formatChange): 
    if formatChange:
        return changeFormat(plt.imread("./images/AB05.png"))
    else:
        return plt.imread("./images/AB05.png")

def changeFormat(img):
    return (255*img).astype(np.uint8)
    
def convertHSV(img):
    "convert into HSV color Space"
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
    "convert into LAB color space"
    if img.shape[2]==4:
        return rgb2lab(img[:,:,0:3])
    else:
        if img.shape[2]==3:
            return rgb2lab(img)
        else:
            print ("Image format not supported")
            
def nbrDiff(img):
    "count the number of different element in an array"
    diff = [img[0,0],]
    for i in range(0,img.shape[0]):
        for j in range(0,img.shape[1]):
            if (img[i,j] in diff) == False:
                diff.append(img[i,j])
    return diff

def showOnScreen(img):
    plt.Figure()
    plt.imshow(img,interpolation='nearest')

def clustering(img,clusters):
    "use the kmean algo to cluster img colors"
    #Reshaping image in list of pixels to allow kmean Algorithm
    #From 1792x1792x3 to 1792^2x3
    pixels = np.reshape(img,(img.shape[0]*img.shape[1],3))
    #print ("pixels in Clustering : ",pixels.dtype,pixels.shape,type(pixels))
    #performing the clustering
    centroids,_ = kmeans(pixels,clusters,iter=3)
    #print ("Centroids : ",centroids.dtype,centroids.shape,type(centroids))
    print centroids
    # quantization
    #Assigns a code from a code book to each observation
    #code : A length N array holding the code book index for each observation.
    #dist : The distortion (distance) between the observation and its nearest code.
    code,_ = vq(pixels,centroids)
    print ("Code : ",code.dtype,code.shape,type(code))

    # reshaping the result of the quantization
    reshaped = np.reshape(code,(img.shape[0],img.shape[1]))
    print ("reshaped : ",reshaped.dtype,reshaped.shape,type(reshaped))
    #print reshaped
    #print nbrDiff(reshaped)
  
    clustered = centroids[reshaped]
    #print ("Centroids : ",centroids.dtype,centroids.shape,type(centroids))
    print ("Clustered : ",clustered.dtype,clustered.shape,type(clustered))
    
    return clustered,reshaped

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

def drawCircle(circle,img,color):
    
    borneInfX= (circle.getcenter()[0]-circle.getradius()).astype(np.int16)
    borneSupX= (circle.getcenter()[0]+circle.getradius()).astype(np.int16)
    borneInfY= (circle.getcenter()[1]-circle.getradius()).astype(np.int16)
    borneSupY=(circle.getcenter()[1]+circle.getradius()).astype(np.int16)
    #print ("Dans drawCircle", circle.getcenter()[0],circle.getcenter()[1],circle.getradius())
    #print (borneInfX,borneSupX,borneInfY,borneSupY)
    if borneInfX < 0:
        borneInfX =0
    if borneSupX > img.shape[0]:
        borneSupX =img.shape[0]
    if borneInfY < 0:
        borneInfY =0
    if borneSupY > img.shape[1]:
        borneSupY =img.shape[1]
        
    for i in range (borneInfX,borneSupX):
        for j in range (borneInfY,borneSupY):
            #print ("point dans cercle? ",circle.contains ([i,j]))
            if circle.contains ([i,j]):
                img[i,j] = color
                #print("coloration")
    return img
          
""" 
def circlePlacementOK(circle,img,color):
    img2 = img
    print circle.getcenter()
    print circle.getradius()
    center = circle.getcenter()
    radius = circle.getradius()

    
    for i in range (center[0]-radius ,center[0]+radius):
            for j in range (center[1]-radius,center[1]+radius):
                if (circle.contains ([i,j])):
                    img2 [i,j] = color
    
    count = np.count_nonzero(img -img2)
    
    return count == 0;
"""

def circularizationDistMap(img,clusters):
    go = True
    listCircle = []
    threshold = 5
    for i in range (0,clusters):
        BinaryImg = (img == i)*1
        print("nbrDiff(img) = ", nbrDiff(BinaryImg))

        window = plt.figure(2)
        plt.imshow(BinaryImg,interpolation = 'nearest')
        plt.title("BinaryImg Avant While")
        plt.show()
        go = True
        while (go):
            distMap= distance_transform_edt(BinaryImg)
            """    
            window = plt.figure(1)
            plt.imshow(distMap,interpolation = 'nearest')
            plt.title("distMap dans le While")
            plt.show()
            """
            
            maxIndex = np.unravel_index(distMap.argmax(), distMap.shape) #recherche de l'indice du maximum de distMap
            maxTest2 = distMap[maxIndex[0],maxIndex[1]]
            """
            maxTest = distMap.max()
            if maxTest == maxTest2 :
                print ("Le Max est le bon")
            """
            # verification distmap(maxIndex) == distmap.max()
            circle = Circle ([maxIndex[0],maxIndex[1]],distMap[maxIndex[0],maxIndex[1]],i)
            BinaryImg = drawCircle(circle,BinaryImg,0)#On place le cercle sur l'img. Les points du cercle appartiennent mnt au fond

            #print ("diff originale - draw",nbrDiff(BinaryImg - tmp))
            """
            window = plt.figure(1)
            plt.imshow(BinaryImg,interpolation = 'nearest')
            plt.title("BinaryImg Dans While")
            plt.show()
            """
            listCircle.append(circle)#memorize les centres et rayon des cercles
            #print (listCircle)
            
            if distMap[maxIndex[0],maxIndex[1]] < threshold:
                go = False
                print("Break du while.")
                
            print ('Iteration suivante ',"maxTest = ",maxTest2," maxIndex = ",maxIndex)
    
    return listCircle
    
def circularization(img):
    "Places the circles"
    diff = nbrDiff(img)
    go = True
    minRadius = 5
    print ("img = ", img.dtype, type(img),img.shape )

    imgTmp = np.zeros_like(img)
    print ("imgTmp1 = ", imgTmp.dtype, type(imgTmp),imgTmp.shape )

    imgTmp = (img == diff[0])
    print ("imgTmp2 = ", imgTmp.dtype, type(imgTmp),imgTmp.shape )

    #imgTmp = img*imgTmp
    """
    imgTmp[0]= img[diff == 0]
    imgTmp[1]= img[diff==1]
    imgTmp[2]= img[diff==2]
    
    for c in range(0,len(diff)):
    
        for i in range(0,imgTmp[c].shape[0]):
            for j in range(0,imgTmp[c].shape[1]):
            """
    for i in range(0,imgTmp.shape[0]):
            for j in range(0,imgTmp.shape[1]):
                minRadius = 5
                circle = Circle ([i,j],minRadius)
                color = imgTmp[i,j]
                while (circlePlacementOK(circle,img,color)):
                    circle = Circle ([i,j],minRadius)
                    if circlePlacementOK(circle,img,color) == False:
                        placement.append([[i,j],radius,color])
                    minRadius =+ 1 #minimal radius of interest
    
    return placement



def morphoNoiseRemoval(img):
    imgB = img
    for i in range(0,5):
        imgB = opening (imgB, square(3))
        imgB = closing (imgB, square(3))
    return imgB

def formationImgCercle (listCircle,img):
    imgCircle = np.zeros_like(img)
    print ("nbr cercles = ", len(listCircle))
    for i in range (0,len(listCircle)):
        print listCircle[i].getcenter(),listCircle[i].getradius(),listCircle[i].getcolor()
        imgCircle = drawCircle(listCircle[i],imgCircle,listCircle[i].getcolor()+1)
    
    return imgCircle

img = loadImages('false')
print ("Original Image",img.dtype, type(img),img.shape)
print ("pixel test Original = ", img[img.shape[0]/2,img.shape[1]/2,:])

#img = changeFormat(img)

imgHSV = convertHSV(img)
print ("imgHSV : ", imgHSV.dtype, type(imgHSV),imgHSV.shape)
print ("pixel test HSV = ", imgHSV[imgHSV.shape[0]/2,imgHSV.shape[1]/2,:])

clusters = 4
imgClus,code = clustering(imgHSV,clusters)
imgClus = convertHSVtoRGB(imgClus)
code2 = code.astype(np.uint8)
morpho = morphoNoiseRemoval(code2)
print ("Code = ", code.dtype, type(code),code.shape )
lcircle = circularizationDistMap(code2,clusters)
imgCirc = formationImgCercle (lcircle,code2)


"""
#circularization(code)
#imgClus2 = convertHSVtoRGB(clustering2(imgHSV,clusters))
kmeanHSV1 = kmeansAlgo(imgHSV)
kmean2 = kmeansAlgo2(img)
kmeanHSV2 = kmeansAlgo2(imgHSV)
#imgLAB = convertLAB(img) 
"""

window1 = plt.figure(1)
plt.title('IMG Cercle')
plt.imshow(imgCirc,interpolation = 'nearest')
plt.colorbar()
plt.show()
"""
window1 = plt.figure(1)
plt.title('IMG Originale')
plt.imshow(img,interpolation = 'nearest')
window = plt.figure(2)
plt.imshow(imgClus,interpolation = 'nearest')
plt.title("Apres Clustering")
window = plt.figure(3)
plt.imshow(morpho,interpolation = 'nearest')
plt.title("Apres operateurs Morpho")
window = plt.figure(4)
plt.imshow(code2,interpolation = 'nearest')
plt.title("Avant operateurs Morpho")
plt.show()
window2= plt.figure(2)
plt.imshow(imgClus2)
plt.title("After Clustering2")
plt.show()
"""