import json
import collections
import numpy as np
from sklearn.decomposition import PCA
import matplotlib
#import utils
matplotlib.use('Agg')
from pylab import figure, axes, pie, title, show
import matplotlib.pyplot as plt
#import numpy as np
from scipy import misc
#import matplotlib.pyplot as plt
import glob
from PIL import Image
import pandas as pd


train=pd.read_csv("/home/nishant.puri2577/data/train_nparray/train.csv")

train=train.values
print train.shape
train=train.astype(np.uint8)

train=train/256.
image_list = []
for filename in glob.glob('/home/nishant.puri2577/data/generated_faces/*'):
    im=misc.imread(filename)
    image_list.append(im)
test=image_list[0].reshape(12288)
for i in range(1,len(image_list)):
    test=np.vstack((test,image_list[i].reshape(12288)))

test=test/256.

cnt_test=test.shape[0]
cnt_train=train.shape[0]

train_sq_sum=np.sum(train*train,axis=1)
train_sq=np.repeat(train_sq_sum.reshape((1,train_sq_sum.shape[0])),cnt_test,axis=0)

test_sq_sum=np.sum(test*test,axis=1)
test_sq=np.repeat(test_sq_sum.reshape((test_sq_sum.shape[0],1)),cnt_train,axis=1)

c=train_sq+test_sq-2*test.dot(train.T)

closest_image_idx=np.argmin(c,1)
min_dist=np.min(c,1)


def closest_im(i):
    plt.close()
    plt.clf()
    image=(test[i]*256).astype(np.uint8)
    #show_sample(image)
    image_nn=train[closest_image_idx[i],:]
    l2_dist=np.sqrt(min_dist[i])
    #print "l2 distance is: ", l2_dist
    image_nn=(image_nn*256).astype(np.uint8)
    #show_sample(image_nn)
    image=image.reshape((64,64,3))
    image_nn=image_nn.reshape((64,64,3))
    plt.figure()
    plt.suptitle('L2 distance: '+str(l2_dist),fontsize=16)
    plt.figure(1)
    plt.subplot(121)
    plt.imshow(image)
    plt.axis("off")
    plt.title('Generated Image')
    
    plt.subplot(122)
    plt.imshow(image_nn)
    plt.axis("off")
    plt.title('Closest Neighbor')
    #plt.suptitle('L2 distance: '+str(l2_dist),fontsize=16)
    plt.savefig('gen_closest'+ str(i) + '_' + str(l2_dist) + '.png')



for i in range(len(image_list)):
    closest_im(i)    
