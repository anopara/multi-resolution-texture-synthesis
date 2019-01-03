#imports
from skimage import io
from skimage.transform import pyramid_gaussian, pyramid_laplacian
from skimage.transform.pyramids import resize
from sklearn.neighbors import KDTree

import numpy as np
import matplotlib.pyplot as plt

import cv2
from math import floor, ceil

from IPython.display import clear_output
from PIL import Image
import os
import random

import json

def multiResolution_textureSynthesis(parms, userExample = None):
    
    #check if save dir exists
    checkIfDirectoryExists(parms["saveImgsPath"])
    #write params
    saveParms(parms)
    
    #1. load example image and build pyramids
    if parms["pyramidType"] == "gaussian":
        exampleMap = gaussianPyramid(loadExampleMap(parms["exampleMapPath"]), levels = parms["pyramidLevels"])
        canvas = gaussianPyramid(initCanvas(parms["outputSize"]), levels = exampleMap.levels)
    elif parms["pyramidType"] == "laplacian":
        exampleMap = laplacianPyramid(loadExampleMap(parms["exampleMapPath"]), levels = parms["pyramidLevels"])
        canvas = laplacianPyramid(initCanvas(parms["outputSize"]), levels = exampleMap.levels)  
    else:
        raise Exception('Please, use either "gaussian" or "laplacian" for pyramidType')

    #track what has been filed
    filledMap = gaussianPyramid(initCanvas(parms["outputSize"]), levels = exampleMap.levels)
    
    #check if we have user example
    if userExample is not None:
        #resize user user example to be same size level 0
        userExampleImg = loadExampleMap(userExample["userExamplePath"])
        userExampleImg = resize(userExampleImg, np.shape(canvas.pyramid[0]))
        #copy to level 0
        canvas.pyramid[0], filledMap.pyramid[0] = copyMap2Map(exampleMap.pyramid[0], canvas.pyramid[0], filledMap.pyramid[0])
        canvas.pyramid[0] = userExampleImg
    else:
        #random init level 0
        randomRow2Map(exampleMap.pyramid[0], canvas.pyramid[0], filledMap.pyramid[0], 2) 

    #2. main resolve loop
    index = 0
    Cs = []
    for pLvl in range(0, canvas.levels+1):
        rows, cols, _ = np.shape(canvas.pyramid[pLvl])
        #build kD-tree for this level
        kD, samples = kDtree(exampleMap.pyramid, pLvl, parms)
        for r in range(rows):   
            for c in range(cols):
                #check if not already resolved
                if filledMap.pyramid[pLvl][r, c][0] == 0:
                    C = findBestMatchCoord(canvas, exampleMap, kD, pLvl, [r,c], parms, samples, k=min(samples, 1))
                    canvas.pyramid[pLvl][r, c] = exampleMap.pyramid[pLvl][C[0], C[1]]
                    filledMap.pyramid[pLvl][r, c] = np.ones((3,))
            
            print(pLvl, " > ", canvas.levels)
            print(r, " > ", rows)
            #SAVE IMAGE EVERY ROW
            showLiveUpdate(canvas, exampleMap, parms["pyramidType"])
            saveImg(canvas, parms["pyramidType"], pLvl, parms["saveImgsPath"], index)

            index += 1

        #copy for visualization purposes (only if gaussian)
        if parms["pyramidType"] == "gaussian": 
            if pLvl+1<=canvas.levels:
                canvas.pyramid[pLvl+1] = visualizeNextLevel(canvas.pyramid[pLvl], canvas.pyramid[pLvl+1], filledMap.pyramid[pLvl+1])

    
def visualizeNextLevel(prevLevel, nextLevel, filledMapNextLevel):
    rows, cols, _ = np.shape(nextLevel)
    img = Image.fromarray(np.uint8(prevLevel*255))
    img = img.resize((rows, cols), resample=0, box=None)
    img = np.asarray(img)/255.0
    return img * (1-filledMapNextLevel) + nextLevel * filledMapNextLevel

def saveParms(parms):
    path = parms["saveImgsPath"] + 'parms.txt'
    with open(path, 'w') as file:
        file.write(json.dumps(parms)) 
    

def resize(in_img, size):
    img = Image.fromarray(np.uint8(in_img*255))
    img = img.resize((size[0], size[1]), resample=0, box=None)
    return np.asarray(img)/255.0

def randomRow2Map(copyFrom, copyTo, filledMap, rowsToCopy):
    rowsEx, colsEx, _ = np.shape(copyFrom)
    rows, cols, _ = np.shape(copyTo)
    rand_r = random.randint(ceil((rowsEx - rowsToCopy)/4), int(rowsEx - rowsToCopy))
    rand_c = random.randint(0, int(colsEx/2))
    copyTo[0:rowsToCopy,0:int(colsEx/2)] = copyFrom[rand_r:rand_r+rowsToCopy, rand_c:rand_c+int(colsEx/2)]
    filledMap[0:rowsToCopy,0:int(colsEx/2)] = 1
    return copyTo, filledMap

def copyMap2Map(copyFrom, copyTo, filledMap):
    rows, cols, _ = np.shape(copyTo)
    img = Image.fromarray(np.uint8(copyFrom*255))
    img = img.resize((rows, cols), resample=0, box=None)
    copyTo = np.asarray(img)/255.0
    filledMap = np.ones(np.shape(filledMap))
    return copyTo, filledMap

def showLiveUpdate(canvas, exampleMap, pyramidType):
    #show live update
    if pyramidType=="gaussian":
        fig=plt.figure(figsize=(30, 30))
        fig_cols = len(canvas.pyramid)
        for c in range(1,fig_cols+1):
            fig.add_subplot(7, fig_cols, c)
            plt.imshow(canvas.pyramid[c-1])
            fig.add_subplot(7, fig_cols, c + fig_cols)
            plt.imshow(exampleMap.pyramid[c-1])
    else:
        fig=plt.figure(figsize=(30, 30))
        fig_cols = len(canvas.pyramid) + 1
        for c in range(1,fig_cols):
            fig.add_subplot(7, fig_cols, c)
            plt.imshow(canvas.pyramid[c-1])
            fig.add_subplot(7, fig_cols, c + fig_cols)
            plt.imshow(exampleMap.pyramid[c-1])
        fig.add_subplot(7, fig_cols, fig_cols)
        plt.imshow(canvas.reconstruct())
        fig.add_subplot(7, fig_cols, fig_cols*2)
        plt.imshow(exampleMap.reconstruct())        
    clear_output(wait=True)
    display(plt.show())

def saveImg(canvas, pyramidType, pyramidLevel, savePath, index):
    #save img
    if pyramidType == "gaussian":
        img = Image.fromarray(np.uint8(canvas.pyramid[pyramidLevel]*255))
    else:
        img = Image.fromarray(np.uint8(np.clip(canvas.reconstruct(), 0.0, 1.0)*255))
    img = img.resize((300, 300), resample=0, box=None)
    img.save(savePath+ str(index) + '.jpg')

def checkIfDirectoryExists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def getSingleLevelNeighbourhood(pyramid, level, coord, kernel, mode):
    
    if mode=='parent':
        coord[0] = floor(coord[0]/2)
        coord[1] = floor(coord[1]/2)

    half_kernel = floor(kernel / 2)
    #pad the image
    padded = padding(pyramid[level], half_kernel)
    #get neighbourhood
    shifted_row = coord[0] + half_kernel
    shifted_col = coord[1] + half_kernel
    row_start = shifted_row - half_kernel
    row_end = shifted_row + half_kernel + 1
    col_start = shifted_col - half_kernel
    col_end = shifted_col + half_kernel + 1

    padded = padded[row_start:row_end, col_start:col_end]

    if mode=='parent':
        return padded.reshape(np.shape(padded)[0]*np.shape(padded)[1], 3)
    if mode=='child': #then return only the half of the neighbourhood which was already resolved (we are going in scan-like order)
        return padded.reshape(np.shape(padded)[0]*np.shape(padded)[1], 3)[0:floor(kernel*kernel/2), :]

def getNeighbourhood(pyramid, pyramidLevel, coord, parms, mirror_hor=False, mirror_vert=False):
    Nchild = getSingleLevelNeighbourhood(pyramid, pyramidLevel, coord, parms["child_kernel_size"], mode='child')
    if pyramidLevel-1>=0:
        Nparent = getSingleLevelNeighbourhood(pyramid, pyramidLevel-1, coord, parms["parent_kernel_size"], mode='parent')
    else:
        Nparent = np.zeros((parms["parent_kernel_size"] * parms["parent_kernel_size"], 3))

    #combine into a single neighbourhood
    N = np.concatenate((Nchild, Nparent), axis=0)
    return N

def findBestMatchCoord(canvas, exampleMap, kD, pyramidLevel, coord, parms, samples, k = 4):
    N = getNeighbourhood(canvas.pyramid, pyramidLevel, coord, parms)
    #find best neighbours
    dist, ind = kD.query([N.reshape(-1)], k=k)
    dist = dist[0]
    ind = ind[0]
    #choose random valid patch
    #PARM_truncation = 0.8
    #PARM_attenuation = 2
    #probabilities = distances2probability(dist, PARM_truncation, PARM_attenuation)
    chosen = ind[0] #np.random.choice(ind, 1, p=probabilities)[0]
    #make flat coord id to 2d coord
    chosenCoord = id2coord(chosen, np.shape(exampleMap.pyramid[pyramidLevel]))
    return chosenCoord

def id2coord(coordFlat, imgSize):
    row = floor(coordFlat / imgSize[0])
    col = coordFlat - row * imgSize[1]
    
    return [row, col]

def kDtree(pyramid, pyramidlevel, parms):
    #get all the possible neighbourhood "coordinates/samples":
    rows, cols, _ = np.shape(pyramid[pyramidlevel])
    samples = []
    for r in range(rows):
        for c in range(cols):
            N = getNeighbourhood(pyramid, pyramidlevel, [r, c], parms)
            samples.append(N.reshape(-1))
            #N = getNeighbourhood(pyramid, pyramidlevel, [r, c], parms, mirror_hor=True)
            #samples.append(N.reshape(-1))
            #N = getNeighbourhood(pyramid, pyramidlevel, [r, c], parms, mirror_vert=True)
            #samples.append(N.reshape(-1))
    return KDTree(samples), int(len(samples))          

def distances2probability(distances, PARM_truncation, PARM_attenuation):

    probabilities = 1 - distances / np.max(distances)  
    probabilities *= (probabilities > PARM_truncation)
    probabilities = pow(probabilities, PARM_attenuation) #attenuate the values
    #check if we didn't truncate everything!
    if np.sum(probabilities) == 0:
        #then just revert it
        probabilities = 1 - distances / np.max(distances) 
        probabilities *= (probabilities > PARM_truncation*np.max(probabilities)) # truncate the values (we want top truncate%)
        probabilities = pow(probabilities, PARM_attenuation)
    probabilities /= np.sum(probabilities) #normalize so they add up to one  

    return probabilities

def padding(img, pad):
    npad = ((pad, pad), (pad, pad), (0, 0))
    return np.lib.pad(img, npad, "constant", constant_values=0) #constant_values=127 'wrap') #,

def initCanvas(size):
    return np.zeros((size[0], size[1], 3), dtype="float64")

def loadExampleMap(exampleMapPath):
    exampleMap = io.imread(exampleMapPath) #returns an MxNx3 array
    exampleMap = exampleMap / 255.0 #normalize
    #make sure it is 3channel RGB
    if (np.shape(exampleMap)[-1] > 3): 
        exampleMap = exampleMap[:,:,:3] #remove Alpha Channel
    elif (len(np.shape(exampleMap)) == 2):
        exampleMap = np.repeat(exampleMap[np.newaxis, :, :], 3, axis=0) #convert from Grayscale to RGB
    return exampleMap

class gaussianPyramid:
    
    def __init__(self, in_img, levels = None):
        self.levels = levels
        self.pyramid = self.build(in_img)
    
    def build(self, img):
        G = img.copy()
        gP = [G]
        if self.levels is None:
            self.levels = 0
            #loop until we have only 1x1 map
            while np.shape(G)[0] > 1:
                G = cv2.pyrDown(G)
                gP.insert(0,G)
                self.levels += 1
        else:
            for i in range(self.levels):
                G = cv2.pyrDown(G)
                gP.insert(0,G)
        return gP
    
    def reconstruct(self):
        return self.pyramid[-1]

class laplacianPyramid:
    
    def __init__(self, in_img, levels = None):
        self.levels = levels
        self.pyramid = self.build(in_img)

    def build(self, img):
        gpB = gaussianPyramid(img, self.levels)
        if self.levels is None:
            self.levels = gpB.levels
        gpB = gpB.pyramid
        lpB = [gpB[0]]
        for i in range(0,len(gpB)-1):
            GE = cv2.pyrUp(gpB[i])
            L = gpB[i+1] - GE
            lpB.append(L)
        return lpB

    def reconstruct(self):
        ls_ = self.pyramid[0].copy()
        for i in range(1,len(self.pyramid)):
            ls_ = cv2.pyrUp(ls_.astype('float64'))
            ls_ = ls_ + self.pyramid[i]
        return ls_