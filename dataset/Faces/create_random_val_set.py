from PIL import Image
import glob, os, sys
import scipy.misc
import numpy as np

count_test = 0
count_train = 0

test_folder = "/home/icarus/Documents/DCGAN_Tensorflow/FacesDataset/face_data_test/"
train_folder = "/home/icarus/Documents/DCGAN_Tensorflow/FacesDataset/face_data_train/"

for filename in glob.glob('/home/icarus/Documents/DCGAN_Tensorflow/FacesDataset/face_data/*'):
    
     image = scipy.misc.imread(filename)
     f, e = os.path.splitext(filename) 

     if (np.random.random() > 0.9):
        count_test +=1
        outname = test_folder + str(count_test) + e
     else:    
        count_train +=1
        outname = train_folder + str(count_train) + e

     scipy.misc.imsave(outname, image)
