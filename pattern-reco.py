"""
Author : GORMEZ David

Imagery Project: Pattern recognition

"""

import numpy as np

from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.collections import LineCollection

#from matplotlib import delaunay as triang
from matplotlib import colors
#from matplotlib import tri as tria

from pymorph import open as opening2
from pymorph import close as closing2


from skimage.color  import rgb2hsv,rgb2lab,hsv2rgb
from skimage.morphology  import closing,opening,square #abandonned because return a untin8 array
from skimage.filter.rank import median
from skimage.morphology import disk

from scipy.ndimage.morphology import distance_transform_edt
from scipy.cluster.vq import kmeans,kmeans2,vq
from scipy.spatial import Delaunay
from scipy.ndimage.filters import median_filter as median2

import itertools
import time
import math

import Tkinter
import tkFileDialog
import os
#import pylab

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

def loadImages(imgPath,custom):
    "charge l'image test ou l'image choisie par l'utilisateur dans un numpy array"
    if custom == True :
        img =  plt.imread(imgPath)
    
    else:
        img =  plt.imread("./images/AB05.png")
        
    img = (255*img).astype(np.uint8)
    
    return img

    
def convertHSV(img):
    "convert RGBA into HSV color Space"
    if img.shape[2]==4:
        return rgb2hsv(img[:,:,0:3])
    else:
        if img.shape[2]==3:
            return rgb2hsv(img)
        else:
            print ("Image format not supported")
 
def convertHSVtoRGB(img):
    "convert colors from HSV to RGB "
    return hsv2rgb(img)

def convertLAB(img):
    "convert an RGB img into LAB color space"
    if img.shape[2]==4:
        return rgb2lab(img[:,:,0:3])
    else:
        if img.shape[2]==3:
            return rgb2lab(img)
        else:
            print ("Image format not supported")
            
def nbrDiff(img):
    "count the number of different element in an array and their occurence"
    diff = []
    cmpt = []
    for i in range(0,img.shape[0]):
        for j in range(0,img.shape[1]):
            if (img[i,j] in diff) == False:
                diff.append(img[i,j])
                cmpt.append(0)
            else :
                cmpt[diff.index(img[i,j])] +=1
                
    return diff,cmpt

def clustering(img,clusters):
    "use the kmean algo to cluster img colors"
    #Reshaping image in list of pixels to allow kmean Algorithm
    #From 1792x1792x3 to (1792^2)x3
    pixels = np.reshape(img,(img.shape[0]*img.shape[1],3))
    #clustering is done on hue value of a pixel color
    #performing the clustering
    centroids,_ = kmeans(pixels,clusters,iter=3)
    # quantization
    #Assigns a code from a code book to each observation
    #code : A length N array holding the code book index for each observation.
    code,_ = vq(pixels,centroids)
    #print ("Code : ",code.dtype,code.shape,type(code))

    # reshaping the result of the quantization
    reshaped = np.reshape(code,(img.shape[0],img.shape[1]))
    #print ("reshaped : ",reshaped.dtype,reshaped.shape,type(reshaped))
    #print reshaped
    #print nbrDiff(reshaped)
  
    clustered = centroids[reshaped]
    #print ("Centroids : ",centroids.dtype,centroids.shape,type(centroids))
    #print ("Clustered : ",clustered.dtype,clustered.shape,type(clustered))
    #print ("nbrDiff de Clustered 0 = " , nbrDiff(clustered[:,:,0]))
    #print ("nbrDiff de Clustered 1 = " ,nbrDiff(clustered[:,:,1]))
    #print ("nbrDiff de Clustered 2 = " ,nbrDiff(clustered[:,:,2]))

    #print nbrDiff(reshaped)
    return clustered,reshaped,standardCode

def clustering2(img,clusters):
    "another clustering method - no major differences"
    #Reshaping image in list of pixels to allow kmean Algorithm
    #From 1792x1792x3 to 1792^2x3
    pixels = np.reshape(img,(img.shape[0]*img.shape[1],3))
    centroids,_ = kmeans2(pixels,3,iter=3,minit= 'random')
    #print ("Centroids : ",centroids.dtype,centroids.shape,type(centroids))
    #print centroids
    # quantization
    #Assigns a code from a code book to each observation
    #code : A length N array holding the code book index for each observation.
    #dist : The distortion (distance) between the observation and its nearest code.
    code,_ = vq(pixels,centroids)
    #print ("Code : ",code.dtype,code.shape,type(code))
    #print code

    # reshaping the result of the quantization
    reshaped = np.reshape(code,(img.shape[0],img.shape[1]))
    #print ("reshaped : ",reshaped.dtype,reshaped.shape,type(reshaped))

    clustered = centroids[reshaped]
    #print ("clustered : ",clustered.dtype,clustered.shape,type(clustered))
    
    #scatter3D(centroids)
    return clustered

def drawCircle(circle,img,color):
    "Draw a circle of a given color on a map"
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
          
def circularizationDistMap(img,standard,mon_fichier,threshold):
    "Uses a partially updated distance map to place a minimum of circles of maximum radius on a clusterised color image"
    go = True
    listCircle = []

    #print("nbrDiff(img) = ", nbrDiff(img))

    print "Go circularization"
    
    for i in range (len(standard)):
        #iterate on the different colors
        BinaryImg = (img == standard[i])*1 #A distance map needs a binary image
        #print("nbrDiff(img) = ", nbrDiff(BinaryImg))
        go = True
        cmpt = 0
        distMap= distance_transform_edt(BinaryImg)#calculate the first distMap
        
        while (go):
            maxIndex = np.unravel_index(distMap.argmax(), distMap.shape) #recherche de l'indice du maximum de distMap
            maxTest2 = distMap[maxIndex[0],maxIndex[1]] #no point of doing that except verification purpose
            # verification distmap(maxIndex) == distmap.max()
            circle = Circle ([maxIndex[0],maxIndex[1]],distMap[maxIndex[0],maxIndex[1]],standard[i])
            BinaryImg = drawCircle(circle,BinaryImg,0)#On place le cercle sur l'img. Les points du cercle appartiennent mnt au fond
            listCircle.append(circle)#memorise les centres et rayon des cercles
            
            if distMap[maxIndex[0],maxIndex[1]] < threshold:
                #If circle size smaller threshold
                go = False
                mon_fichier.write("Break du while.")
            
            str_fichier = "Iteration suivante" + "maxTest = " + str(maxTest2) + " maxIndex = " + str(maxIndex)
            mon_fichier.write(str_fichier+ '\n')
            
            distMap = distMapUpdate(distMap,BinaryImg, circle)
            #update a part of the distance map

    return listCircle

def distMapUpdate(distMap, binaryImg, placedCircle):
    "Update a part of the distance map"

    radiusCirc = (placedCircle.radius).astype(np.int16)
    circCenter = placedCircle.center
    lowerLeftCoordX = circCenter[0] - 2* radiusCirc
    lowerLeftCoordY = circCenter[1] - 2* radiusCirc
    mapToUpdate = np.ones((radiusCirc*4,radiusCirc*4))
    #print lowerLeftCoordX, lowerLeftCoordY
    #print radiusCirc
    #print mapToUpdate.shape
    
    lowerLeftCoordX,lowerLeftCoordY,limitX,limitY = ConsiderLimitsFrame(lowerLeftCoordX,lowerLeftCoordY,radiusCirc,binaryImg)
    
    mapToUpdate = np.ones((limitX,limitY))

    for i in range(0,limitX):
        for j in range(0,limitY):
            mapToUpdate[i,j] = binaryImg [lowerLeftCoordX+i,lowerLeftCoordY+j]
    
    distMapUpdate = distance_transform_edt(mapToUpdate)
    
    for i in range(limitX):
        for j in range(limitY):
            distMap[lowerLeftCoordX+i,lowerLeftCoordY+j] = distMapUpdate [i,j] 
    
    return distMap
    
def ConsiderLimitsFrame(lowerLeftCoordX,lowerLeftCoordY,radius,binaryImg):
    "Returns the limits of the dist map to update considering the limits of the image"
    shapeX = binaryImg.shape[0]
    shapeY = binaryImg.shape[1]
    lowerX = lowerLeftCoordX
    lowerY = lowerLeftCoordY
    limitX = 4*radius
    limitY = 4*radius
    
    if (lowerLeftCoordX < 0):
        limitX = 4*radius +  lowerLeftCoordX
        lowerX = 0

    if (lowerLeftCoordY < 0):
        limitY = 4*radius +  lowerLeftCoordY
        lowerY = 0
        
    if ((lowerLeftCoordX + 4*radius) > shapeX):
        limitX =  shapeX - lowerLeftCoordX 
        
    if ((lowerLeftCoordY + 4*radius) > shapeY):
        limitY =  shapeY - lowerLeftCoordY
        
    return lowerX,lowerY,limitX,limitY
    
def morphoNoiseRemoval(img):
    "Removes noise by succession of 5 opening/closing morphological operators"
    for i in range(0,5):
        img = opening2(img, square(3))
        img = closing2(img, square(3))
        
    return img

def formationImgCercle (listCircle,img,mon_fichier):
    "Use the list of circle found with circularizationDistMap to draw the image composed exclusively of circle and representing the original image"
    imgCircle = np.zeros_like(img)
    mon_fichier.write("nbr cercles = ")
    mon_fichier.write(str(len(listCircle)))
                      
    for i in range (0,len(listCircle)):
        imgCircle = drawCircle(listCircle[i],imgCircle,listCircle[i].getcolor())
    
    return imgCircle

def delaunayTriangulation3(lcircle,mon_fichier3):
    "Apply a Delaunauy triangulation on the centers of the circles given by circularizationDistMap"
    points = []
    coordCentersX = []
    coordCentersY = []
    
    for i in range(len(lcircle)):
        points.append(lcircle[i].getcenter())
        coordCentersX.append(lcircle[i].getcenter()[0])#could be avoided 
        coordCentersY.append(lcircle[i].getcenter()[1])#could be avoided 
    
    #print points
    #y = [i[0] for i in x] #From list of tuples to list of integrers
    
    tri = Delaunay(points)
    mon_fichier3.write("points :"+ str(tri.points)+ str(tri.points.shape) +"\n")
    mon_fichier3.write("Sommets des triangles:"+ str(tri.vertices)+str (tri.vertices.shape)+ "\n")

    return tri,coordCentersX,coordCentersY,points

def colorizedDelaunay3(tri,coordCentersX,coordCentersY,lcircle,imgCirc,points,mon_fichier4):
    "Make a list of all the segments composing the triangulation, remove the redundant ones and determine the color that the segment should have based on the node propreties"
    # Make a list of line segments: 
    # edge_points = [ ((x1_1, y1_1), (x2_1, y2_1)),
    #                 ((x1_2, y1_2), (x2_2, y2_2)),
    #                 ... ]
    pointsTest = np.array(points)
    edge_points = []
    edges = set()
    colorsEdge = []
    #print pointsTest
    #print edges
    
    # loop over triangles: 
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.vertices:
        edge_points,edges,colorsEdge = add_edge(ia, ib,edge_points,edges,pointsTest,imgCirc,colorsEdge)
        edge_points,edges,colorsEdge = add_edge(ib, ic,edge_points,edges,pointsTest,imgCirc,colorsEdge)
        edge_points,edges,colorsEdge = add_edge(ic, ia,edge_points,edges,pointsTest,imgCirc,colorsEdge)
    
    mon_fichier4.write("edges :"+ str(edges)+"\n")
    mon_fichier4.write("edge_points :"+ str(edges)+"\n")

    lines = LineCollection(edge_points,colors = colorsEdge,linewidths=1)#All the lines with their associated color

    return lines,pointsTest,colorsEdge

def showColorisedDelaunayGraph(lines,points):
    "Draw the colorised graph considering the node propreties. "
    plt.figure()
    plt.title('Delaunay triangulation Colorized')
    plt.gca().add_collection(lines)
    plt.plot(points[:,0], points[:,1], 'o', hold=1,markersize=1,markerfacecolor='green', markeredgecolor='red')

    plt.show()
    
    return

def add_edge(i, j,edge_points,edges,pointsTest,imgCirc,colors):
    """Add a line between the i-th and j-th points, if not in the list already"""
    
    if (i, j) in edges or (j, i) in edges:
        # already added
        colors.append(colorChoice(imgCirc[pointsTest[i,0],pointsTest[i,1]],
                                  imgCirc[pointsTest[j,0],pointsTest[j,1]]))
        return edge_points,edges,colors
    
    edges.add( (i, j) )
    edge_points.append(pointsTest[[i, j]])
    colors.append(colorChoice(imgCirc[pointsTest[i,0],pointsTest[i,1]],
                              imgCirc[pointsTest[j,0],pointsTest[j,1]]))

    return edge_points,edges,colors

def colorChoice(a,b):
    """
    Chooses right color depending on the texture of the vertices of the edge
    There are ' propreties : Lumina(a), cancer(b),nuclear(c) and Stroma(d) 
    So there are 10 possible colors to attribute to the edge.
    a-a a-b a-c a-d b-b b-c b-d c-c c-d d-d
    """
    #different standard tested for optimal visualization
    standardColors = ['black',
                      'LightYellow', #a-a
                      'orange', #a-b
                      'cyan',#a-c
                      'yellow',#a-d
                      'brown',#b-b
                      'DarkViolet',#b-c
                      'DarkSlateGray',#b-d
                      'blue',#c-c
                      'green',#c-d
                      'Silver',#d-d
                      ]
    
    standardColors2 = ['black',
                      'yellow', #a-a
                      'Gold', #a-b
                      'Khaki',#a-c
                      'Orange',#a-d
                      'Chocolate',#b-b
                      'Brown',#b-c
                      'SaddleBrown',#b-d
                      'Blue',#c-c
                      'SteelBlue',#c-d
                      'Silver',#d-d
                      ]
    
    standardColors3 = ['black',
                      'yellow', #a-a
                      'Gold', #a-b
                      'white',#a-c
                      'white',#a-d
                      'Chocolate',#b-b
                      'Brown',#b-c
                      'Chocolate',#b-d
                      'Blue',#c-c
                      'SteelBlue',#c-d
                      'black',#d-d
                      ]
    standardColors4 = ['black',
                      'White', #a-a
                      'brown', #a-b
                      'blue',#a-c
                      'White',#a-d
                      'brown',#b-b
                      'brown',#b-c
                      'Chocolate',#b-d
                      'blue',#c-c
                      'green',#c-d
                      'Silver',#d-d
                      ]
    standardSum = [0,20,110,1010,10010,200,1100,10100,2000,11000,20000] 
    #because of the standard used, the sum of the node propreties is unique for each combination
    #print "a+b = ", a+b
    tempIndex = standardSum.index(a+b)
    #print "index = ", tempIndex
    color = standardColors4[tempIndex]
    
    return color

def colorizedDelaunay(tri,coordCentersX,coordCentersY,lcircle):
    "Basic coloration function for simple vue of the Triangulation. No consideration of node propreties"
    plt.triplot(coordCentersX, coordCentersY, tri.vertices.copy())#manage triangle edges color here
    plt.plot(coordCentersX, coordCentersY, 'o', markersize=1,markerfacecolor='green', markeredgecolor='red')
    plt.show()
    
    return

def mouseClick(imgClick,nbrClick):
    "Allows the user to tell the program what color is lumina,cancer,nuclear,stroma"
    
    plt.imshow(imgClick,origin='lower')
    print("Please click" + str(nbrClick) +" times. ")
    print ("In order : Lumina, brown(cancer), nuclear (blue) , stoma (beige),fond")
    coord = plt.ginput(nbrClick, timeout=0, show_clicks=True)
    print("clicked",coord)
    plt.close()
    x = []
    y = []
    listDiff=[]
    for i in range (len(coord)):
        xTemp,yTemp = coord[i]
        y.append(xTemp.astype(np.uint16)) #Because Fuck you! That's why!
        x.append(yTemp.astype(np.uint16))#ginput reverse x and y when clicking on an image
        #i hate you ginput
        #listDiff.append(imgClick[x[i],y[i]])
    
    #print imgClick[x[0],y[0]],imgClick[x[1],y[1]],imgClick[x[2],y[2]],imgClick[x[3],y[3]]#,imgClick[x[4],y[4]]
    
    return x,y

def etablishCodeLink(code,x,y):
    "establish the link between the standard used and the code used by the program"
    listDiff= []
    for i in range (len(x)):        
        listDiff.append(code[x[i],y[i]])
    
    return listDiff

def standardCode(img,listDiff):
     "use a standard code to identify lumina,cancer,nuclear,stromal parts. After this function, lumina = 10,cancer = 100, nuclear = 1000, stroma = 10000"
     #[lumina, cancer, noyau,stroma]
     #standard = [10,15,20,25]
     standard = [10,100,1000,10000] #the standard used to represent lumina,cancer,nuclear,stromal parts
     imgStandard = np.zeros_like(img, dtype=np.uint16)

     for i in range (img.shape[0]):
        for j in range (img.shape[1]):
            imgStandard[i,j] = standard[listDiff.index(img[i,j])]
            
     #print listDiff
     return imgStandard,standard

def customColorMap():
    "Make a standard color map to standardise lumina,cancer,nuclear, stroma colors"
    # make a color map of fixed colors
    cmap = colors.ListedColormap(['white','orange','brown','blue','Silver'])
    #bounds=[-1,9,13,18,23,28] #standard
    bounds=[-1,9,13,110,1010,10010] #standard

    norm = colors.BoundaryNorm(bounds, cmap.N)
    return cmap,norm

def initFiles():
    "create some files for output results"
    mon_fichier = open ("Sortie_projet_feedback.txt", "w")#argh, everything is curshed
    mon_fichier2 = open("DelaiExecution.txt","w")
    mon_fichier3 = open("Triangles delaunay","w")
    mon_fichier4 = open("colorization Triangulation 3","w")
    return mon_fichier,mon_fichier2,mon_fichier3,mon_fichier4

def closeFiles(mon_fichier,mon_fichier2,mon_fichier3,mon_fichier4):
    "close the opened output files by initFiles"
    mon_fichier.close()
    mon_fichier2.close()
    mon_fichier3.close()
    mon_fichier4.close()
    return

def fileChooserTinker():
    "Window to allow selection of image to be procesed "
    custom = False

    root = Tkinter.Tk()
    file = tkFileDialog.askopenfile(parent=root,mode='rb',title='Choose a Image file to process')
    root.withdraw()
    
    if file != None:
        custom = True
    
    return file,custom

def main():
    tmps1 = time.clock()
    mon_fichier,mon_fichier2,mon_fichier3,mon_fichier4 = initFiles() #initialize outputfiles
    clusters = 4 
    threshold = 5 #Minimal radius for circles
    cmap_Custom,Norm = customColorMap()#create custom color map
    mon_fichier2.write("Temps Initialisation = " + str (time.clock()-tmps1))
    
    "Allow a choice of the image to be processed"
    imgPath,custom = fileChooserTinker()
    tmps1 = time.clock()
    img = loadImages(imgPath,custom) #load img
    mon_fichier2.write("Temps Chargement Img = " + str (time.clock()-tmps1))
    print "Image loaded + Init complete"
    #mon_fichier.write("Original Image :"+str(img.dtype) + str(type(img))+ str(img.shape)+ "\n")
    #mon_fichier.write("imgHSV : "+ str(imgHSV.dtype) +str(type(imgHSV)) + str(imgHSV.shape)+"\n")
    #mon_fichier.write("pixel test HSV = "+ str(imgHSV[imgHSV.shape[0]/2,imgHSV.shape[1]/2,:])+"\n")
    
    "Convert image to HSV"
    tmps1 = time.clock()
    imgHSV = convertHSV(img)
    mon_fichier2.write("Temps ConversionHSV = " + str (time.clock()-tmps1))
    print "Convert RGB to HSV complete"
    
    "Color Clustering"
    tmps1=time.clock()
    imgClus,code,standardCode = clustering(imgHSV,clusters)
    mon_fichier2.write("Clustering = " + str (time.clock()-tmps1) + "\n")
    print "Clustering Complete"
    
    "Denoise a first time"
    tmps1 = time.clock()
    code = median2(code, size = 3)
    mon_fichier2.write("Median denoising = " + str (time.clock()-tmps1) + "\n")

    print "Median denoisig complete"
    
    "Color Standardisation for lumina,cancer,Nuclear and stromal parts"
    #print code.shape ,  nbrDiff(code)
    imgClus = convertHSVtoRGB(imgClus)
    x,y = mouseClick(imgClus,clusters)
    listDiff = etablishCodeLink(code,x,y)
    clusStandard,standard = standardCode(code,listDiff)
    #print "ClusStandard = " , nbrDiff(clusStandard)
    print "Color standisation complete"

    "Noise Removal"
    tmps1 = time.clock()
    morpho = morphoNoiseRemoval(clusStandard)
    #print "morpho = " , nbrDiff(morpho)
    #mon_fichier.write("Code = " + str(code.dtype)+str( type(code))+str(code.shape))
    mon_fichier2.write("Noise Removal open/close = " + str(time.clock()-tmps1)+ "\n")
    print "Noise removal (opening/closing) complete"
    
    "Calculate Circle placement position"
    tmps1=time.clock()
    lcircle = circularizationDistMap(morpho,standard,mon_fichier,threshold)
    mon_fichier2.write("Circularization = " + str(time.clock()-tmps1) + "\n")
    print "Circularisation complete"
    
    "Create img composed of circles"
    tmps1 = time.clock()
    imgCirc = formationImgCercle (lcircle,morpho,mon_fichier)
    mon_fichier2.write("formation Img Cercle = " + str(time.clock()-tmps1) + "\n")
    print "Image Circle formed"
    
    #x,y = mouseClick(imgCirc,5)
    #cens,edg,tri,neig = delaunayTriangulation(lcircle)
    #delaunayTriangulation2(lcircle)
    #ImgCircStandard = standardisationCouleur(imgCirc,x,y)
    
    "Triangulation de Delaunay"
    tmps1 = time.clock()
    tri,coordCentersX,coordCentersY,points = delaunayTriangulation3(lcircle,mon_fichier3)
    mon_fichier2.write("Triangulation Delaunay = " + str(time.clock()-tmps1) + "\n")
    print "Delaunay Triangulation complete"
    
    "Colorisation de la triangulation"
    #colorizedDelaunay(tri,coordCentersX,coordCentersY,lcircle)
    #colorizedDelaunay2(tri,coordCentersX,coordCentersY,lcircle,imgCirc)
    graph,pointsTemp,colorEdges = colorizedDelaunay3(tri,coordCentersX,coordCentersY,lcircle,imgCirc,points,mon_fichier4)
    mon_fichier2.write("Triangulation Delaunay Colorised = " + str(time.clock()-tmps1) + "\n")
    print "Colorised Delaunay Triangulation complete"

    "Closes the files openend for output stream"
    closeFiles(mon_fichier,mon_fichier2,mon_fichier3,mon_fichier4)
    print "After Closing output files"
    
    window1 = plt.figure(1)
    plt.title('IMG Cercle')
    plt.imshow(imgCirc,origin='lower',interpolation = 'nearest',cmap= cmap_Custom,norm = Norm)
    plt.colorbar()

    window1 = plt.figure(3)
    plt.title('IMG Originale')
    plt.imshow(img,origin='lower', interpolation = 'nearest')
    
    window1 = plt.figure(4)
    plt.title('IMG Clusterised')
    plt.imshow(imgClus,interpolation = 'nearest')
    
    window = plt.figure(5)
    plt.imshow(morpho,interpolation = 'nearest',cmap= cmap_Custom,norm = Norm)
    plt.title('After opening/closing')
    
    window = plt.figure(6)
    plt.triplot(coordCentersX, coordCentersY, tri.vertices.copy())#manage triangle edges color here
    plt.plot(coordCentersX, coordCentersY, 'o', markersize=1,markerfacecolor='green', markeredgecolor='red')
    plt.title('Delaunay Triangulation Basic')
    
    showColorisedDelaunayGraph(graph,pointsTemp)

    return 0

if __name__ == "__main__":
    main()
    sys.exit()


    
    
    """
    window1 = plt.figure(1)
    plt.title('IMG Originale')
    plt.imshow(morphoStandard,interpolation = 'nearest')
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