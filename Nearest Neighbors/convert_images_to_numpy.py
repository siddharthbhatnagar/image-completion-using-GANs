import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
import glob
from PIL import Image
image_list = []
f=open('img_processed.log','w+')
for filename in glob.glob('/home/nishant.puri2577/data/jpg2/*'):
    im=misc.imread(filename)
    image_list.append(im)
print len(image_list)
train=image_list[0].reshape(12288)
for i in range(1,len(image_list)):
    if(i%100==0):
        f.write('images processed' + str(i))
        f.write('\n')
        f.flush()
    train=np.vstack((train,image_list[i].reshape(12288)))
np.savetxt("train.csv", train, delimiter=",")
