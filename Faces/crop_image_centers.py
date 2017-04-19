from PIL import Image
import glob, os, sys
import scipy.misc

for filename in glob.glob('/home/icarus/Documents/DCGAN_Tensorflow/FacesDataset/data/celebA/*'):
     image = scipy.misc.imread(filename)
     f, e = os.path.splitext(filename)
     outname = f + "_cropped" + e
     im2 = scipy.misc.imresize(image[55:163, 30:148], [64,64])
     scipy.misc.imsave(outname, im2)
