#imports
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time as ti
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pympler import asizeof
import time as ti
from PIL import Image
import scipy.misc

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

def lrelu(x,alpha):
    return tf.maximum(x, alpha*x)

def linear(input_tensor, input_dim, output_dim, name=None):
    with tf.variable_scope(name):
        weights = tf.get_variable("weights", [input_dim, output_dim], initializer=tf.truncated_normal_initializer(stddev=math.sqrt(3.0 / (input_dim + output_dim))))
        bias = tf.get_variable("bias", [output_dim], initializer=tf.constant_initializer(0.0))
        return tf.matmul(input_tensor, weights) + bias 

def conv_2d(input_tensor, input_dim, output_dim, name=None):
    with tf.variable_scope(name):
        kernel = tf.get_variable("kernel", [5, 5,input_dim, output_dim], initializer=tf.truncated_normal_initializer(stddev=0.02))
        bias = tf.get_variable("bias", [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(input_tensor, kernel, strides=[1, 2, 2, 1],padding='SAME')
        return conv+bias

def conv_2dtranspose(input_tensor, input_dim, output_shape,name=None):
    output_dim=output_shape[-1]
    with tf.variable_scope(name):
        kernel = tf.get_variable("kernel", [5, 5, output_dim, input_dim], initializer=tf.truncated_normal_initializer(stddev=0.02))
        bias = tf.get_variable("bias", [output_dim], initializer=tf.constant_initializer(0.0))
        deconv = tf.nn.conv2d_transpose(input_tensor, kernel, output_shape=output_shape, strides=[1, 2, 2, 1],padding='SAME')
        return deconv+bias

def batch_norm(input_tensor,name,is_train=True):
    return tf.contrib.layers.batch_norm(input_tensor,decay=0.9, updates_collections=None, epsilon=1e-5, scale=True,    
                                        is_training=is_train, scope=name)

def preprocessing(image):
    return image/127.5 - 1;

def show_sample(X):
    im = X
    plt.imshow(im)
    plt.axis('on')
    plt.show()  

batch_size = 64

def generator(z):
    l1=linear(input_tensor=z,name="g_lin", input_dim=100, output_dim=1024*4*4)  
    l2= tf.reshape(l1, [-1, 4, 4, 1024])
    l3 = tf.nn.relu(batch_norm(input_tensor=l2,name="g_bn0"))
    print l3
    #conv1
    l4=conv_2dtranspose(input_tensor=l3,name="g_c2dt1",input_dim=1024,output_shape=[batch_size,8,8,512])
    l5=tf.nn.relu(batch_norm(input_tensor=l4,name="g_bn1"))
    print l5
    #conv2
    l6=conv_2dtranspose(input_tensor=l5,name="g_c2dt2",input_dim=512,output_shape=[batch_size,16,16,256])
    l7=tf.nn.relu(batch_norm(input_tensor=l6,name='g_bn2'))
    print l7
    #conv3
    l8=conv_2dtranspose(input_tensor=l7,name='g_c2dt3',input_dim=256,output_shape=[batch_size,32,32,128])
    l9=tf.nn.relu(batch_norm(input_tensor=l8,name='g_bn3'))
    print l9
    #conv4
    l10=conv_2dtranspose(input_tensor=l9,name='g_c2dt4',input_dim=128,output_shape=[batch_size,64,64,3])
    l11=tf.nn.tanh(l10)
    print l11
    return l11

def discriminator(images, reuse=False, alpha=0.2):
     with tf.variable_scope("discriminator") as scope:
        if reuse:
            scope.reuse_variables()
            
        #naming of the layers is as per layer number    
        #h0 conv2d no batch_norm
        l1 = conv_2d(input_tensor=images, input_dim=3, output_dim= 64, name='d_c2d0')
        l2 = lrelu(l1,alpha)

        #h1 conv2d with batch_norm
        l3 = conv_2d(input_tensor=l2, input_dim=64, output_dim=64*2, name='d_c2d1')
        l4 = batch_norm(input_tensor=l3,name="d_bn1")
        l5 = lrelu(l4,alpha)

        #h2 conv2d with batch_norm
        l6 = conv_2d(input_tensor=l5, input_dim=64*2, output_dim=64*4, name='d_c2d2')
        l7 = batch_norm(input_tensor=l6,name="d_bn2")
        l8 = lrelu(l7,alpha)

        #h3 conv2d with batch_norm
        l9 = conv_2d(input_tensor=l8, input_dim=64*4, output_dim=64*8, name='d_c2d3')
        l10 = batch_norm(input_tensor=l9,name="d_bn3")
        l11 = lrelu(l10,alpha)

        #h4 reshape and linear
        l12 = tf.reshape(l11, [-1, 8192]) #l12 = tf.reshape(l11, [32, -1]) #l12 = tf.reshape(l11, [64, -1])
        l13 = linear(input_tensor=l12, input_dim=8192, output_dim=1, name="d_lin4")
        print l13.get_shape().as_list()
        #sigmoid
        l14 = tf.nn.sigmoid(l13)
        print l14
        return l14, l13

#place holders for images and z
z = tf.placeholder(tf.float32, [None, 100], name='z')
G=generator(z)

#placeholder for images
images = tf.placeholder(tf.float32, [None,64,64,3], name='images')
alpha = 0.2
D1, D1_logits = discriminator(images, False, alpha)
D2, D2_logits = discriminator(G, True, alpha)

#create list of discrim and gen vars
t_vars=tf.trainable_variables()
for var in t_vars:
    print var.name
discrim_vars = [var for var in t_vars if 'd_' in var.name]
gen_vars = [var for var in t_vars if 'g_' in var.name]

for var in discrim_vars:
    print var.name

for var in gen_vars:
    print var.name

img_width, img_height = 64, 64
data_dir = './data'
learning_rate= 0.0002
beta1= 0.5
batch_size=64

#LOSS
discrim_loss_real_img = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D1_logits, labels=tf.scalar_mul(0.9,tf.ones_like(D1_logits))))
discrim_loss_fake_img = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D2_logits, labels=tf.zeros_like(D2_logits)))
discrim_loss = discrim_loss_real_img + discrim_loss_fake_img
gen_loss = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(logits=D2_logits, labels=tf.ones_like(D2_logits)))

#optimizers
dopt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1).minimize(discrim_loss, var_list=discrim_vars)
gopt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1).minimize(gen_loss, var_list=gen_vars)

load_img_datagen = ImageDataGenerator(preprocessing_function = preprocessing)
img_input = load_img_datagen.flow_from_directory(
        data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

d_loss_all=[]
g_loss_all=[]

def merge_images(image_batch, size):
    h,w = image_batch.shape[1], image_batch.shape[2]
    c = image_batch.shape[3]
    img = np.zeros((int(h*size[0]), w*size[1], c))
    for idx, im in enumerate(image_batch):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w,:] = im
    return img


def save_image(X, iter, fl):
    name = 'Iteration' + str(iter) + 'time' + str(ti.time()) + '.png'
    if (fl == False):
        name = 'Random'+name
    size = [8,8]
    X[0] = (X[0] + 1.)/2

    im = merge_images(X[0], size)
    scipy.misc.imsave(name, im)

def save_sample(X, iter, fl):
    im = X
    plt.imshow(im)
    plt.axis('on')
    name = 'Iteration' + str(iter) + 'time' + str(ti.time()) + '.png'
    plt.savefig(name)

disp_img_noise = np.random.uniform(-1,1,size=[batch_size,100])
saver = tf.train.Saver()
f = open('train.log', 'w+')
iters = 50000
GEN_LOSS_LIMIT = 150

for i in range(iters):
    #train discriminator
    real_images=next(img_input)
    noise= np.random.uniform(-1,1,size=[batch_size,100])
    sess.run([dopt],feed_dict={z:noise,images:real_images})

    #train discriminator
    real_images=next(img_input)
    noise= np.random.uniform(-1,1,size=[batch_size,100])
    sess.run([dopt],feed_dict={z:noise,images:real_images})    
    
    #gen
    noise= np.random.uniform(-1,1,size=[batch_size,100])
    sess.run([gopt],feed_dict={z:noise})

    if (np.sum(g_loss_all[-100:]) > GEN_LOSS_LIMIT):
        #adaptive generator update
        noise= np.random.uniform(-1,1,size=[batch_size,100])
        sess.run([gopt],feed_dict={z:noise})
        f.write('Extra Generator in iteration: ' + str(i) + ' sum of last 100: ' + str(np.sum(g_loss_all[-100:])) + '\n')
        print 'Extra Generator in iteration: ' + str(i) + ' sum of last 100: ' + str(np.sum(g_loss_all[-100:]))

    #evaluate 
    noise_tr= np.random.uniform(-1,1,size=[batch_size,100])
    real_images=next(img_input)
    #train generator
    #d_loss_all.append(discrim_loss.eval({z: noise_tr,images:real_images}))
    #g_loss_all.append(gen_loss.eval({z: noise_tr}))    

    d_loss_all.append(sess.run([discrim_loss],feed_dict={z:noise_tr, images:real_images}))
    g_loss_all.append(sess.run([gen_loss], feed_dict={z:noise_tr}))

    print i, g_loss_all[-1], d_loss_all[-1]
    
    if (i%1000 == 0):
        losses_list = [g_loss_all, d_loss_all]
        with open('loss.csv', 'w') as loss_file:
            writer = csv.writer(loss_file)
            writer.writerows(losses_list)
        fake_img = sess.run([G],feed_dict={z:disp_img_noise})
	save_image(fake_img, i, True)
    random_noise = np.random.uniform(-1,1,size=[batch_size,100])
	save_image(sess.run([G],feed_dict={z:random_noise}), i, False)
        name = 'saved_model.ckpt'
        saver.save(sess,name)
    
    log_string =  'iteration: ' + str(i) + ' g_loss:' + str(g_loss_all[-1]) + ' d_loss:' + str(d_loss_all[-1])
    f.write(log_string)
    f.write('\n')
    f.flush()
