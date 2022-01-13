import numpy as np
from skimage.feature import local_binary_pattern
import skimage.feature
from skimage.transform import resize


class FeatureExtractor:

    VALID_METHODS = ['pca','structure', 'hog', 'gradient', 'hotspots']

    def __init__(self, method='structure'):
        self.method = method


    def __horizontal(self, image):
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
        if len(X.shape) == 2:
               X = X.reshape(X.shape[0],16,15)  
        row = X.shape[1]
        col = X.shape[2]
        dim = row+col+(row*(col+1))/(cell_size*cell_size)  
        structure_X  = np.zeros(shape=(X.shape[0],int(dim)))
        for p in range(X.shape[0]):
            h = self.__horizontal(X[p])
            v = self.__vertical(X[p])
            z = self.__zone(X[p],cell_size)
            structure_X[p] = np.asarray(h+v+z)
        
        return structure_X


    def __apply_hog(self, X, cell_size=4, orientation=9,block_size=2):
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
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0],16,15) 

        dim = block_size * block_size * orientation * ((cell_size-1)*(cell_size-1)) 
        hog_X  = np.zeros(shape=(X.shape[0],dim))
        for p in range(X.shape[0]):
            image = resize(X[p], (16,16)) 
            hog_X[p]= skimage.feature.hog(image, 
                orientations=orientation, 
                pixels_per_cell=(cell_size, cell_size),
                cells_per_block=(block_size, block_size),
                block_norm='L2')
    

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
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0],16,15)
         
        row = X.shape[1]
        col = X.shape[2]
        grad  = np.zeros(shape=X.shape)
        grad_X = np.zeros(shape=(X.shape[0],row*col))
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
        
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0],16,15)

        for i in range(5,12,2):
            for j in range(5,12,2):
                hotspots.append((i,j))
                
        for r in range(X.shape[0]):
            f = list()
            for h in hotspots:
                i,j = h
                f.extend(self.__get_feature(i,j,X[r]).values())

            features.append(f)    
        features = np.array(features)

        return features

    def __apply_pca(self, X):
        """Returns the raw data as it is
        
        Parameter
        ---------
        X = dataset of digit images (16 X 15 matrix)
        
        Returns
        -------
        X
        """

        return X

    def transform(self, X, cell_size=None):
        '''Apply feature extraction method. If method is structure,
        cell_size needs to be specified.'''

        if isinstance(self.method, str) and self.method in FeatureExtractor.VALID_METHODS:
            if self.method == FeatureExtractor.VALID_METHODS[0]:
                return self.__apply_pca(X)
            if self.method == FeatureExtractor.VALID_METHODS[1]:
                return self.__apply_structure(X, cell_size)
            elif self.method == FeatureExtractor.VALID_METHODS[2]:
                return self.__apply_hog(X)
            elif self.method == FeatureExtractor.VALID_METHODS[3]:
                return self.__apply_gradient(X)
            elif self.method == FeatureExtractor.VALID_METHODS[4]:
                return self.__apply_hotspots(X)
         
        else:
            raise Exception('Please enter a valid method', FeatureExtractor.VALID_METHODS)


