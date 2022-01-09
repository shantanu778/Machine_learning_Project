#!/usr/bin/env python
# coding: utf-8



import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from skimage.feature import local_binary_pattern
import skimage.feature
from skimage.transform import resize



def distance(i,j,a,b):
    """Find the eucladian distance between two points A and B
    
    Parameters
    --------
    i,j = coordinates from point A
    a,b = coordinates from point B
    
    
    Returns
    ---------
    floating point eucladian distance
    """
    
    return np.sqrt((i-a)*(i-a)+(j-b)*(j-b))
    

def get_feature(i,j,image):
    
    """Given the coordinates of a hotspot get the distances from the nearest 
    white pixel in 8 chaincode directions apart from each other at 45 degrees
    
    Parameter
    ---------
    
    i,j = coordinates of a hotspot
    
    image = 16 X 15 matrix of digit image
    
    returns
    ---------
    
    dictionary
                nearest distance of a white pixel in 8 different directions.
                
    
    """

    chain = [(0,1),(0,-1),(1,0),(-1,0),(-1,1),(-1,-1),(1,-1),(1,1)]
    D = {v:22.00 for v in range(1,9)}
    for idx,c in enumerate(chain):

        a, b = i,j
        m,n = c
        flag = False
        
        while a+m>0 and a+m<16 and b+n>0 and b+n<15:
            
            a = a+m 
            b = b+n
            if image[a][b]==6:
                D[idx+1]= min(D[idx+1],distance(i,j,a,b))
                
                flag = True
                break
                
        
        if flag==False:
            D[idx+1]=min(D[idx+1],distance(i,j,a,b))
            
    return D
        
def hotspots(X):
    
    """We define 16 hotspots in our image and measures the distances from the nearest 
    white pixel in 8 chaincode directions.
    
    Parameter
    ---------
    
    X = dataset of digit images (16 X 15 matrix)
    
    returns
    -------
    numpy array of features. For each image it finds 128 features
    
    """
    
    features = list()
    hotspots = list()

    for i in range(5,12,2):
        for j in range(5,12,2):
            hotspots.append((i,j))
            
    for r in range(X.shape[0]):
        f = list()
        for h in hotspots:
            i,j = h
            f.extend(get_feature(i,j,X[r]).values())

        features.append(f)    
    features = np.array(features)
    return features



def get_intensity(image,i,j):
    
    """calculate the change of the intensity of pixel
    
    Parameter
    ---------
    
    image = 16 X 15 matrix of digit image
    i,j = coordinates of pixel
    
    returns
    -------
    floating point
        magnitude of intensity change
    
    """
    soble_x = np.asarray([[-1,0,1],[-2,0,2],[-1,0,1]])
    soble_y = np.asarray([[-1,-2,-1],[0,0,0],[1,2,1]])
    x = 0
    y = 0
    chain = [(0,1),(0,-1),(1,0),(-1,0),(-1,1),(-1,-1),(1,-1),(1,1)]

    for c in chain:
        m,n = c
        x = x+image[i+m][j+n]*soble_x[1+m][1+n]
        y = y+image[i+m][j+n]*soble_y[1+m][1+n]

    
    magnitude = np.sqrt(x*x+y*y)

    return magnitude
    

def gradient(X):
    
    """replace the pixel values of images with the magnitude of gradient or intensity change.
    https://en.wikipedia.org/wiki/Image_gradient
    
    Parameter
    ---------
    
    X = dataset of digit images ((number of samples) 16 X 15 matrix)
    n_pca = number of pca components to return. defualt = 100
    
    
    returns
    -------
    
    features with n_pca components of each image
    
    
    """
    
    grad  = np.zeros(shape=X.shape)
    grad_X = np.zeros(shape=(X.shape[0],240))
    for p in range(X.shape[0]):
        inp = np.pad(X[p],(1,1),'constant',constant_values=(0))
        for i in range(1,17):
            for j in range(1,16):
                grad[p][i-1][j-1]=get_intensity(inp,i,j)
        grad_X[p] = grad[p].flatten()
        

    
    return grad_X
    



#LBP
def lbp(X,normalize=True):
    
    """Local binary patterns(lbp)
    https://en.wikipedia.org/wiki/Local_binary_patterns
    
   
    
    Parameter
    ---------
    
    X = dataset of digit images ((number of samples) 16 X 15 matrix)
    normalize (optional) = Boolean variable. If true then normalize the features. defualt = True
    
    
    returns
    -------
    
    A matrix with (number of samples) X 128 dimension as features  
    
    
    """

    lbp_X  = np.zeros(shape=(X.shape[0],128))
    for p in range(X.shape[0]):
        lbp = local_binary_pattern(X[p], 8, 1 ,'default')
        n_bins = 128
        hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))
        lbp_X[p] = hist
    
    if normalize:
        lbp_X = StandardScaler().fit_transform(lbp_X)
    
    return lbp_X




def hog(X,normalize=True):
    
    """Histogram of oriented gradients(HOG)
    https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients
    
   
    
    Parameter
    ---------
    
    X = dataset of digit images ((number of samples) 16 X 15 matrix)
    normalize (optional) = Boolean variable. If true then normalize the features. defualt = True
    
    
    returns
    -------
    
    A matrix with (number of samples) X 144 dimension as features  
    
    
    """
    
    hog_X  = np.zeros(shape=(X.shape[0],324))
    for p in range(X.shape[0]):
        image = resize(X[p], (16,16)) 
        hog_X[p]= skimage.feature.hog(image, orientations=9, pixels_per_cell=(4, 4),cells_per_block=(2, 2),block_norm='L2')
    if normalize:
        hog_X = StandardScaler().fit_transform(hog_X)

    return hog_X



def horizontal(image):
    
    """measures the ratio of black(0) and white(6) pixel counts in each rows of the image
    
    Parameter
    ---------
    
    image = 16 X 15 matrix of a digit image
    
    returns
    --------
    a list of 16 values
    
    """
    
    features = list()
    for i in range(image.shape[0]):
        black = 1
        white = 1
        for j in range(image.shape[1]):
            if image[i][j]==0:
                black+=1
            elif image[i][j]==0:
                white+=1
        features.append(black/white)
    return features

def vertical(image):
    
    """measures the ratio of black(0) and white(6) pixel counts in each columns of the image
    
    Parameter
    ---------
    
    image = 16 X 15 matrix of a digit image
    
    returns
    --------
    a list of 15 values
    
    """
    
    features = list()
    for j in range(image.shape[1]):
        black = 1
        white = 1
        for i in range(image.shape[0]):
            if image[i][j]==0:
                black+=1
            elif image[i][j]==0:
                white+=1
        features.append(black/white)
    return features
        
def zone_cal(image):

        
    """measures the ratio of black(0) and white(6) pixel counts in a zone
    
    Parameter
    ---------
    
    image = 16 X 15 matrix of a digit image
    
    returns
    --------
    
    Ratio of black(0) and white(6) pixel counts
    
    """
    
    black = 1
    white = 1
    m = int(image.shape[0])
    n = int(image.shape[1])
    for i in range(m):
        for j in range(n):
            if image[i][j]==0:
                black+=1
            elif image[i][j]==0:
                white+=1
    return black/white

def zone(image,cell_size):
    
        
    """divides the image into multiple square zones with 2*2 or 4*4 pixels in each of the zone
    
    and measures the ratio of black(0) and white(6) pixel in each zone of the image
    
    Parameter
    ---------
    
    image = 16 X 15 matrix of a digit image
    cell_size = divides the image into multiple sub images of cell_size*cell_size squares. e.g 2,4,8 (must be a factor of 16).
    
    
    returns
    --------
    a list of N values where N is the number of total zones
    
    """
    
    features = list()
    image = np.pad(image,(0,1),'constant',constant_values=(0))
    image = np.delete(image,-1,0)
    
    dim = int(image.shape[0]/cell_size)
   

    sub_images = [np.hsplit(a,dim) for a in np.array_split(image,dim)]
    
    for i in range(dim):
        for j in range(dim):
            features.append(zone_cal(sub_images[i][j]))
            
    return features

def structure(X, cell_size, normalize=True):
    
    """measures the ratio of black(0) and white(6) pixel counts in different orientation (row, col, zones)
    
   
    
    Parameter
    ---------
    
    X = dataset of digit images ((number of samples) 16 X 15 matrix)
    cell_size = divides the image into multiple sub images of cell_size*cell_size squares. e.g 2,4,8 (must be a factor of 16).
    normalize (optional) = Boolean variable. If true then normalize the features. defualt = True
    
    
    returns
    -------
    
    A matrix with (number of samples) X N dimension as features. N = 16+15+number of zones
    
    
    """
    
    structure_X  = np.zeros(shape=(X.shape[0],95))
    for p in range(X.shape[0]):
        h = horizontal(X[p])
        v = vertical(X[p])
        z = zone(X[p],cell_size)
        structure_X[p] = np.asarray(h+v+z)

    if normalize:
        structure_X = StandardScaler().fit_transform(structure_X)
    
    return structure_X
           











