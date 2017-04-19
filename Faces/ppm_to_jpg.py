from PIL import Image
import glob, os, sys

count = 0
print "aya"

for filename in glob.glob('/home/icarus/Documents/DCGAN_Tensorflow/FacesDataset/faces/*'):
    f, e = os.path.splitext(filename)
    outfile = f + ".jpg"
    count += 1
    print outfile
    try:
        Image.open(filename).save(outfile)
    except IOError:
        print "cannot convert" + filename


print count
