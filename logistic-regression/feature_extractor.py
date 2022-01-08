import numpy as np
from skimage.feature import local_binary_pattern
import skimage.feature
import math


class FeatureExtractor:

    VALID_METHODS = ['structure', 'hog', 'gradient', 'hotspots', 'lbp']

    def __init__(self, method='structure'):
        self.method = method


    def __horizontal(self, image):
        """Measures the ratio of black(0) and white(6) pixel counts in each rows
        of the image.
        
        Parameter
        ---------
        image = 16 X 15 matrix of a digit image
        
        Returns
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


    def __vertical(self, image):
        """Measures the ratio of black(0) and white(6) pixel counts in each
        columns of the image.
        
        Parameter
        ---------
        image = 16 X 15 matrix of a digit image
        
        Returns
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
            

    def __zone_cal(self, image):
        """Measures the ratio of black(0) and white(6) pixel counts in a zone.
        
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


    def __zone(self, image, cell_size):
        """Divides the image into multiple square zones with 2*2 or 4*4 pixels in
        each of the zone and measures the ratio of black(0) and white(6) pixel in
        each zone of the image.
        
        Parameter
        ---------
        image = 16 X 15 matrix of a digit image
        cell_size = divides the image into multiple sub images of 
        cell_size*cell_size squares. e.g 2,4,8 (must be a factor of 16).
        
        Returns
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
                features.append(self.__zone_cal(sub_images[i][j]))
                
        return features
        

    def __apply_structure(self, X, cell_size):
        """Measures the ratio of black(0) and white(6) pixel counts in
        different orientations (row, col, zones).
        
        Parameter
        ---------
        X = dataset of digit images ((number of samples) 16 X 15 matrix)
        cell_size = divides the image into multiple sub images of 
        cell_size*cell_size squares. e.g 2,4,8 (must be a factor of 16).
        normalize (optional) = Boolean variable. If true then normalize the 
        features. default = True

        Returns
        -------
        A matrix with (number of samples) X N dimension as features. 
        N = 16+15+number of zones
        """
        
        structure_X  = np.zeros(shape=(X.shape[0],95))
        for p in range(X.shape[0]):
            h = self.__horizontal(X[p])
            v = self.__vertical(X[p])
            z = self.__zone(X[p],cell_size)
            structure_X[p] = np.asarray(h+v+z)
        
        return structure_X


    def __apply_hog(self, X):
        """Histogram of oriented gradients(HOG)
        https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients
        
        Parameter
        ---------
        X = dataset of digit images ((number of samples) 16 X 15 matrix)
        normalize (optional) = Boolean variable. If true then normalize the features. defualt = True
        
        Returns
        -------
        A matrix with (number of samples) X 144 dimension as features  
        """
        
        hog_X  = np.zeros(shape=(X.shape[0],144))
        for p in range(X.shape[0]):
            hog_X[p]= skimage.feature.hog(X[p], orientations=8, pixels_per_cell=(4, 4))

        return hog_X


    def __get_intensity(self, image,i,j):
        """calculate the change of the intensity of pixel.
        
        Parameter
        ---------
        image = 16 X 15 matrix of digit image
        i,j = coordinates of pixel
        
        Returns
        -------
        floating point: magnitude of intensity change
        """
    
        x = image[i,j-1]-image[i,j+1]
        y = image[i-1,j]-image[i+1,j]
        magnitude = np.sqrt(x*x+y*y)

        return magnitude


    def __apply_gradient(self, X):
        """Replace the pixel values of images with the magnitude of gradient or
        intensity change.mhttps://en.wikipedia.org/wiki/Image_gradient
        
        Parameter
        ---------
        X = dataset of digit images ((number of samples) 16 X 15 matrix)
        n_pca = number of pca components to return. defualt = 100
        
        Returns
        -------
        features with n_pca components of each image    
        """
        
        grad  = np.zeros(shape=X.shape)
        grad_X = np.zeros(shape=(X.shape[0],240))
        for p in range(X.shape[0]):
            inp = np.pad(X[p],(1,1),'constant',constant_values=(0))
            for i in range(1,17):
                for j in range(1,16):
                    grad[p][i-1][j-1]=self.__get_intensity(inp,i,j)
            grad_X[p] = grad[p].flatten()
        
        return grad_X


    def __distance(self, i, j, a, b):
        """Find the euclidian distance between two points A and B.
        
        Parameters
        --------
        i,j = coordinates from point A
        a,b = coordinates from point B
        
        Returns
        ---------
        floating point eucladian distance
        """
        
        return np.sqrt((i-a)*(i-a)+(j-b)*(j-b))
        

    def __get_feature(self, i, j, image):
        """Given the coordinates of a hotspot get the distances from the nearest 
        white pixel in 8 chaincode directions apart from each other at 45 degrees
        
        Parameter
        ---------
        i,j = coordinates of a hotspot
        image = 16 X 15 matrix of digit image
        
        Returns
        ---------
        dictionary: nearest distance of a white pixel in 8 different directions.
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
                    D[idx+1]= min(D[idx+1],self.__distance(i,j,a,b))
                    
                    flag = True
                    break
                    
            
            if flag==False:
                D[idx+1]=min(D[idx+1],self.__distance(i,j,a,b))
                
        return D


    def __apply_hotspots(self, X):
        """Define 9 hotspots in our image and measure the distances from the
        nearest white pixel in 8 chaincode directions.
        
        Parameter
        ---------
        X = dataset of digit images (16 X 15 matrix)
        
        Returns
        -------
        numpy array of features. For each image it finds 72 features
        """
        
        features = list()
        hotspots = list()

        for i in range(4,11,3):
            for j in range(3,12,4):
                hotspots.append((i,j))
                
        for r in range(X.shape[0]):
            f = list()
            for h in hotspots:
                i,j = h
                f.extend(self.__get_feature(i,j,X[r]).values())

            features.append(f)    
        features = np.array(features)

        return features


    def __apply_lbp(self, X):
        """Local binary patterns (lbp).
        https://en.wikipedia.org/wiki/Local_binary_patterns
        
        Parameter
        ---------
        X = dataset of digit images ((number of samples) 16 X 15 matrix)
        normalize (optional) = Boolean variable. If true then normalize the
        features. default = True
        
        Returns
        -------
        A matrix with (number of samples) X 128 dimension as features  
        """

        lbp_X  = np.zeros(shape=(X.shape[0],128))
        for p in range(X.shape[0]):
            lbp = local_binary_pattern(X[p], 8, 1 ,'default')
            n_bins = 128
            hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))
            lbp_X[p] = hist
        
        return lbp_X


    def fit_transform(self, X, cell_size=None):
        '''Apply feature extraction method. If method is structure,
        cell_size needs to be specified.'''

        if isinstance(self.method, str) and self.method in FeatureExtractor.VALID_METHODS:
            if self.method == FeatureExtractor.VALID_METHODS[0]:
                return self.__apply_structure(X, cell_size)
            elif self.method == FeatureExtractor.VALID_METHODS[1]:
                return self.__apply_hog(X)
            elif self.method == FeatureExtractor.VALID_METHODS[2]:
                return self.__apply_gradient(X)
            elif self.method == FeatureExtractor.VALID_METHODS[3]:
                return self.__apply_hotspots(X)
            elif self.method == FeatureExtractor.VALID_METHODS[4]:
                return self.__apply_lbp(X)
        else:
            raise Exception('Please enter a valid method', FeatureExtractor.VALID_METHODS)


