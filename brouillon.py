"""
Brouillon
"""

"""
def rgb3Dscatter(img):
    fig = plt.figure()
    ax = Axes3D(fig)
    if img.shape[2]==4:
        img = img[:,:,0:3]
    RGBlist = [(img[:,:,0],img[:,:,1],img[:,:,2])]
    ax.scatter(img[:,:,0],img[:,:,1],img[:,:,2],c=[(r[0], r[1],r[2]) for r in RGBlist])
    ax.grid(False)
    ax.set_title('grid on')
    plt.savefig('blah.png')


rgb3Dscatter(img)
"""

"""
window2 = plt.figure(2)
window2.add_subplot(1,2,1)
plt.title('Hue hist')
hueHist = plt.hist(imgHSV[0].flatten(),256,range=(0.0,1), fc='k', ec='k')
#plt.imshow(hueHist)
"""