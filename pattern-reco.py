"""
Author : GORMEZ David

Imagery Project: Pattern recognition

"""

import numpy as np

from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.collections import LineCollection
import matplotlib.delaunay as triang
from matplotlib import colors
import matplotlib.tri as tria

from skimage.color  import rgb2hsv,rgb2lab,hsv2rgb
from skimage.morphology  import closing,opening,square

from scipy.ndimage.morphology import distance_transform_edt
from scipy.cluster.vq import kmeans,kmeans2,vq
from scipy.spatial import Delaunay

from mpl_toolkits.mplot3d import Axes3D

import itertools
import time
import math

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
    #From 1792x1792x3 to 1792^2x3
    pixels = np.reshape(img,(img.shape[0]*img.shape[1],3))
    #print ("pixels in Clustering : ",pixels.dtype,pixels.shape,type(pixels))
    #performing the clustering
    centroids,_ = kmeans(pixels,clusters,iter=3)
    #print ("Centroids : ",centroids.dtype,centroids.shape,type(centroids))
    #print centroids
    # quantization
    #Assigns a code from a code book to each observation
    #code : A length N array holding the code book index for each observation.
    #dist : The distortion (distance) between the observation and its nearest code.
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
    #standardCode = standardisationCode(reshaped,clustered)
    return clustered,reshaped,standardCode

def standardisationCode(code,imgClus):
    "To always have the same code for the different colors - Does not work"
    StandardCode = np.zeros_like(imgClus)
    
    for i in range (imgClus.shape[0]):
       for j in range (imgClus.shape[1]):
           if (abs(imgClus[i,j,1] - 0.08426584056974025) < 0.01):
               StandardCode[i,j] = 1
           else: 
               if (abs(imgClus[i,j,1] - 0.097370640760267024) < 0.01):
                   StandardCode[i,j] = 3
               else:  
                   if (abs(imgClus[i,j,1] - 0.12959040866125687) < 0.01):
                       StandardCode[i,j] = 2
                   else:
                       if (abs(imgClus[i,j,1] - 0.33117686721545525) < 0.01):
                           StandardCode[i,j] = 4
                           
    lumina = imgClus[100,100]
    brown = imgClus [1270,989]
    blue = imgClus [510,108]
    stroma = imgClus [1580,792]
    print (lumina,brown,blue,stroma)
    """
    print (lumina,brown,blue,stroma)
    # palette must be given in sorted order
    palette = [lumina,brown,blue,stroma]
    # key gives the new values you wish palette to be mapped to.
    key = np.array([1,2,3,4])
    index = np.digitize(code.reshape(-1,), palette)-1
    
    print(key[index])#.reshape(a.shape))
    """
    return StandardCode

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
          
def circularizationDistMap(img,standard):
    go = True
    listCircle = []
    threshold = 8
    
    print "Go circularization"
    
    for i in range (len(standard)):
        BinaryImg = (img == standard[i])*1
        print("nbrDiff(img) = ", nbrDiff(BinaryImg))

        go = True
        cmpt = 0
        distMap= distance_transform_edt(BinaryImg)
        while (go):
            maxIndex = np.unravel_index(distMap.argmax(), distMap.shape) #recherche de l'indice du maximum de distMap
            maxTest2 = distMap[maxIndex[0],maxIndex[1]]
            # verification distmap(maxIndex) == distmap.max()
            circle = Circle ([maxIndex[0],maxIndex[1]],distMap[maxIndex[0],maxIndex[1]],standard[i])
            BinaryImg = drawCircle(circle,BinaryImg,0)#On place le cercle sur l'img. Les points du cercle appartiennent mnt au fond
            listCircle.append(circle)#memorize les centres et rayon des cercles
            #print (listCircle)
            
            if distMap[maxIndex[0],maxIndex[1]] < threshold:
                go = False
                mon_fichier.write("Break du while.")
            
            str_fichier = "Iteration suivante" + "maxTest = " + str(maxTest2) + " maxIndex = " + str(maxIndex)
            mon_fichier.write(str_fichier+ '\n')
            
            distMap = distMapUpdate(distMap,BinaryImg, circle)

    return listCircle

def distMapUpdate(distMap, binaryImg, placedCircle):
    #doc pavement 2D tiling pour cercle?
    #mettre a jour une partie de la dist map
    #afficher la distrib des rayons
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
    imgB = img
    for i in range(0,5):
        imgB = opening (imgB, square(3))
        imgB = closing (imgB, square(3))
    return imgB

def formationImgCercle (listCircle,img):
    imgCircle = np.zeros_like(img)
    mon_fichier.write("nbr cercles = ")
    mon_fichier.write(str(len(listCircle)))
                      
    for i in range (0,len(listCircle)):
        #print listCircle[i].getcenter(),listCircle[i].getradius(),listCircle[i].getcolor()
        imgCircle = drawCircle(listCircle[i],imgCircle,listCircle[i].getcolor())
    
    return imgCircle

def delaunayTriangulation3(lcircle):
    points = []
    coordCentersX = []
    coordCentersY = []
    
    for i in range(len(lcircle)):
        points.append(lcircle[i].getcenter())
        coordCentersX.append(lcircle[i].getcenter()[0])
        coordCentersY.append(lcircle[i].getcenter()[1])
    print points
    #y = [i[0] for i in x] #From list of tuples to list of integrers
    
    tri = Delaunay(points)
    mon_fichier3.write("points :"+ str(tri.points)+ str(tri.points.shape) +"\n")
    mon_fichier3.write("Sommets des triangles:"+ str(tri.vertices)+str (tri.vertices.shape)+ "\n")

    return tri,coordCentersX,coordCentersY,points

def colorizedDelaunay3(tri,coordCentersX,coordCentersY,lcircle,imgCirc,points):
    ""
    # Make a list of line segments: 
    # edge_points = [ ((x1_1, y1_1), (x2_1, y2_1)),
    #                 ((x1_2, y1_2), (x2_2, y2_2)),
    #                 ... ]
    pointsTest = np.array(points)
    edge_points = []
    edges = set()
    colorsEdges = []
    print pointsTest
    print edges
    
    # loop over triangles: 
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.vertices:
        print ia,ib,ic
        edge_points,edges,colors = add_edge(ia, ib,edge_points,edges,pointsTest,imgCirc,colorsEdges)
        edge_points,edges,colors = add_edge(ib, ic,edge_points,edges,pointsTest,imgCirc,colorsEdges)
        edge_points,edges,colors = add_edge(ic, ia,edge_points,edges,pointsTest,imgCirc,colorsEdges)
    
    mon_fichier4.write("edges :"+ str(edges)+"\n")
    mon_fichier4.write("edge_points :"+ str(edges)+"\n")

    lines = LineCollection(edge_points,colors = colorsEdges,linewidths=1)
    """
    fig, ax = plt.subplots()
    ax.add_collection(lines)
    ax.autoscale()
    ax.margins(0.1)
    """
    return lines,pointsTest

def showColorisedDelaunayGraph(lines,points):
    
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
    standardSum = [0,20,110,1010,10010,200,1100,10100,2000,11000,20000]
    #print "a+b = ", a+b
    tempIndex = standardSum.index(a+b)
    #print "index = ", tempIndex
    color = standardColors[tempIndex]
    
    return color

def colorizedDelaunay(tri,coordCentersX,coordCentersY,lcircle):
    
    plt.triplot(coordCentersX, coordCentersY, tri.vertices.copy())#manage triangle edges color here
    plt.plot(coordCentersX, coordCentersY, 'o', markersize=1,markerfacecolor='green', markeredgecolor='red')
    plt.show()
    
    return

def mouseClick(imgClick,nbrClick):
    #print  "shape ImgClick = ",  imgClick.shape
    window2 = plt.figure(2)
    plt.imshow(imgClick,origin='lower')
    print("Please click" + str(nbrClick) +" times. ")
    print ("In order : Lumina, brown(cancer), nuclear (blue) , stoma (beige),fond")
    coord = plt.ginput(nbrClick, timeout=0, show_clicks=True)
    print("clicked",coord)
    plt.show()
    x = []
    y = []
    listDiff=[]
    for i in range (len(coord)):
        xTemp,yTemp = coord[i]
        y.append(xTemp.astype(np.uint16)) #Because Fuck you! That's why!
        x.append(yTemp.astype(np.uint16))#ginput reverse x and y when clicking on an image
        #i hate you ginput
        listDiff.append(imgClick[x[i],y[i]])
    
    print imgClick[x[0],y[0]],imgClick[x[1],y[1]],imgClick[x[2],y[2]],imgClick[x[3],y[3]]#,imgClick[x[4],y[4]]
    
    return x,y,listDiff

def standardCode(img,listDiff):
     "using a standard code to identify lumina,cancer,nuclear,stromal parts"
     #[lumina, cancer, noyau,stroma]
     #standard = [10,15,20,25]
     standard = [10,100,1000,10000]
     imgStandard = np.zeros_like(img, dtype=np.uint16)

     for i in range (img.shape[0]):
        for j in range (img.shape[1]):
            imgStandard[i,j] = standard[listDiff.index(img[i,j])]
            
     print listDiff
     return imgStandard,standard

def customColorMap():
    # make a color map of fixed colors
    cmap = colors.ListedColormap(['black','LightYellow','brown','blue','Silver'])#in order = 8,10,15,20,25
    #bounds=[-1,9,13,18,23,28] #standard
    bounds=[-1,9,13,110,1010,10010] #standard

    norm = colors.BoundaryNorm(bounds, cmap.N)
    return cmap,norm

mon_fichier = open ("Sortie_projet_feedback.txt", "w")#argh, everything is curshed
mon_fichier2 = open("DelaiExecution.txt","w")
mon_fichier3 = open("Triangles delaunay","w")
mon_fichier4 = open("colorization Triangulation 3","w")

tmps1 = time.clock()
img = loadImages('false')
mon_fichier.write("Original Image :"+str(img.dtype) + str(type(img))+ str(img.shape)+ "\n")
#print ("pixel test Original = ", img[img.shape[0]/2,img.shape[1]/2,:])
imgHSV = convertHSV(img)
#print ("imgHSV : ", imgHSV.dtype, type(imgHSV),imgHSV.shape)
#print ("pixel test HSV = ", imgHSV[imgHSV.shape[0]/2,imgHSV.shape[1]/2,:])
tmps2 = time.clock()
mon_fichier2.write("DelaiOuverture + ConversionHSV = " + str (tmps2-tmps1))

clusters = 4
tmps1=time.clock()
imgClus,code,standardCode = clustering(imgHSV,clusters)
tmps2 = time.clock()
mon_fichier2.write("Clustering = " + str (tmps2-tmps1) + "\n")
tmps1 = time.clock()
imgClus = convertHSVtoRGB(imgClus)
code2 = code.astype(np.uint8)
morpho = morphoNoiseRemoval(code2)

cmap_Custom,Norm = customColorMap()

_,_,listDiff = mouseClick(morpho,4)
morphoStandard,standard = standardCode(morpho,listDiff)


print nbrDiff(morphoStandard)

"""
window1 = plt.figure(1)
plt.title('IMG Cercle Standard')
plt.imshow(morphoStandard,origin = 'lower',interpolation = 'nearest',cmap= cmap_Custom,norm = Norm)
plt.colorbar()
plt.show()
"""

mon_fichier2.write("Code = " + str(code.dtype)+str( type(code))+str(code.shape))
tmps2 = time.clock()
mon_fichier2.write("Noise Removal = " + str(tmps2-tmps1)+ "\n")
tmps1 = time.clock()
lcircle = circularizationDistMap(morphoStandard,standard)
tmps2=time.clock()
mon_fichier2.write("Circularization = " + str(tmps2-tmps1) + "\n")
tmps1 = time.clock()

imgCirc = formationImgCercle (lcircle,morphoStandard)
mon_fichier2.write("formation Img Cercle = " + str(time.clock()-tmps1) + "\n")

#x,y = mouseClick(imgCirc,5)
#cens,edg,tri,neig = delaunayTriangulation(lcircle)
#delaunayTriangulation2(lcircle)
#ImgCircStandard = standardisationCouleur(imgCirc,x,y)

tmps1 = time.clock()

tri,coordCentersX,coordCentersY,points = delaunayTriangulation3(lcircle)
mon_fichier2.write("Triangulation Delaunay = " + str(time.clock()-tmps1) + "\n")

colorizedDelaunay(tri,coordCentersX,coordCentersY,lcircle)
#colorizedDelaunay2(tri,coordCentersX,coordCentersY,lcircle,imgCirc)
graph,pointsTemp = colorizedDelaunay3(tri,coordCentersX,coordCentersY,lcircle,imgCirc,points)


mon_fichier.close()
mon_fichier2.close()
mon_fichier4.close()
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
plt.imshow(imgCirc,origin='lower',interpolation = 'nearest',cmap= cmap_Custom,norm = Norm)
plt.colorbar()

window1 = plt.figure(3)
plt.title('IMG Originale')
plt.imshow(img,origin='lower', interpolation = 'nearest')

window1 = plt.figure(2)
plt.title('morphoStandard')
plt.imshow(morphoStandard,interpolation = 'nearest',cmap= cmap_Custom,norm = Norm)
plt.colorbar()
#window = plt.figure(5)
#plt.imshow(standardCode,interpolation = 'nearest')
#plt.title("Apres Clustering Standard")
showColorisedDelaunayGraph(graph,pointsTemp)

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