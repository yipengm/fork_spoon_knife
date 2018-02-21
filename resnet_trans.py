from __future__ import division
from __future__ import print_function

from PIL import Image



import dataset_utils
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.slim import arg_scope
from tensorflow.contrib.slim.python.slim.nets import resnet_utils
from tensorflow.contrib.slim.python.slim.nets import resnet_v1
from tensorflow.contrib.slim.python.slim.nets.resnet_v1 import resnet_v1_block
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step

import os
import time

import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from c4 import resnet_v1_101_c4
from res5 import resnet_v1_101_s5


slim = tf.contrib.slim

tf.set_random_seed(1)
np.random.seed(1)

BATCH_SIZE = 25
initial_learning_rate = 0.001              
decay_steps = 100
learning_rate_decay_factor = 0.85

#mnist = input_data.read_data_sets('./mnist', one_hot=True)  # they has been normalized to range (0,1)
#test_x = mnist.test.images[:2000]
#test_y = mnist.test.labels[:2000]

# plot one example
#print(mnist.train.images.shape)     # (55000, 28 * 28)
#print(mnist.train.labels.shape)   # (55000, 10)
#plt.imshow(mnist.train.images[0].reshape((28, 28)), cmap='gray')
#plt.title('%i' % np.argmax(mnist.train.labels[0])); plt.show()

image_data_x = np.zeros([994,200,200,3])
image_data_y = np.zeros([1210,200,200,3])
image_data_z = np.zeros([1966,200,200,3])
image_data = np.zeros([4170,200,200,3])


i = 0
for file in os.listdir('./knifey-spoony/forky'):
    if file.endswith('.jpg'):
        img_path = os.path.join("./knifey-spoony/forky",file)
        img = Image.open(img_path)
        image_data_x[i,:,:,:] = np.array(img).astype(float)
        i=i+1

i = 0
for file in os.listdir('./knifey-spoony/knifey'):
    if file.endswith('.jpg'):
        img_path = os.path.join("./knifey-spoony/knifey",file)
        img = Image.open(img_path)
        image_data_y[i,:,:,:] = np.array(img).astype(float)
        i=i+1

i = 0
for file in os.listdir('./knifey-spoony/spoony'):
    if file.endswith('.jpg'):
        img_path = os.path.join("./knifey-spoony/spoony",file)
        img = Image.open(img_path)
        image_data_z[i,:,:,:] = np.array(img).astype(float)
        i=i+1

label_data_x = np.matlib.repmat(np.array([1,0,0]),994,1)
label_data_y = np.matlib.repmat(np.array([0,1,0]),1210,1)
label_data_z = np.matlib.repmat(np.array([0,0,1]),1966,1)
label_data = np.zeros([4170,3])


image_data[0:994,:,:,:] = image_data_x[:,:,:,:]
image_data[994:2204,:,:,:] = image_data_y[:,:,:,:]
image_data[2204:4170,:,:,:] = image_data_z[:,:,:,:]

label_data[0:994,:] = label_data_x[:,:]
label_data[994:2204,:] = label_data_y[:,:]
label_data[2204:4170,:] = label_data_z[:,:]



tf_x = tf.placeholder(tf.float32, [None, 200*200*3]) / 255.
image = tf.reshape(tf_x, [-1, 200, 200, 3])              
tf_y = tf.placeholder(tf.int32, [None, 3])            





log_dir = './log'
checkpoint_file = './resnet_v1_101.ckpt'


'''
output_res, _ = resnet_v1_101_c4(inputs=image,num_classes=None,global_pool=False,is_training=True)

exclude = ['resnet_v1_101/block4','resnet_v1_101/logits','resnet_v1_101/predictions']
variables_to_restore = slim.get_variables_to_restore(exclude = exclude)
flat = tf.reshape(output_res, [-1, 1*1*1024])
output = tf.layers.dense(inputs = flat,units = 10)
'''


####################################


#with slim.arg_scope([slim.conv2d,slim.fully_connected],normalizer_fn = slim.batch_norm, normalizer_params = {'is_training': True, 'updates_collections':None}):
#    output_res, _ = resnet_v1.resnet_v1_101(inputs=image,num_classes=None,global_pool=False,is_training=False)
#exclude = ['resnet_v1_101/logits']
#variables_to_restore = slim.get_variables_to_restore(exclude = exclude)
#flat = tf.reshape(output_res, [-1, 7*7*2048])




######################################################C4!##################################################

with slim.arg_scope([slim.conv2d,slim.fully_connected],normalizer_fn = slim.batch_norm, normalizer_params = {'is_training': True, 'updates_collections':None}):
    output_res, _ = resnet_v1_101_c4(inputs=image,num_classes=None,global_pool=False,is_training=True)




exclude = ['resnet_v1_101/block4','resnet_v1_101/logits','resnet_v1_101/predictions']
variables_to_restore = slim.get_variables_to_restore(exclude = exclude)
flat_0 = tf.reshape(output_res, [-1,1024])
W_1 = tf.get_variable("W_1",shape = (1024,1),initializer = tf.random_normal_initializer())
flat_1 = tf.matmul(flat_0,W_1)
flat_2 = tf.reshape(flat_1,[-1,7*7])




output = tf.layers.dense(inputs = flat_2,units = 3,kernel_initializer = tf.random_normal_initializer())
part_v = [v for v in tf.global_variables() if v not in variables_to_restore]





loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y, logits=output) 

global_step = tf.Variable(0, trainable=False)


lr = tf.train.exponential_decay(
            learning_rate = initial_learning_rate,
            global_step = global_step,
            decay_steps = decay_steps,
            decay_rate = learning_rate_decay_factor,
            staircase = True)


#train_op = tf.train.AdamOptimizer(learning_rate = initial_learning_rate).minimize(loss,var_list = part_v)

train_op = tf.train.AdamOptimizer(learning_rate = initial_learning_rate).minimize(loss,var_list = part_v)

label = tf.argmax(tf_y, axis=1)
predictions = tf.argmax(output, axis=1)
correct_bol = tf.equal(label,predictions)
correct = tf.cast(correct_bol,tf.int32)
accuracy = tf.reduce_sum(correct)/tf.shape(label)[0]



sess = tf.Session()


sess.run(tf.global_variables_initializer())

#saver = tf.train.Saver(variables_to_restore)
#saver.restore(sess, './resnet_v1_101.ckpt')




for epoch in range(100):
    i=0
    index = np.arange(4170)
    index_rand = np.random.permutation(index)

    for step in range(30):
        i = i+100
        batch_xs = image_data[index_rand[i:i+100],:,:,:]
        batch_xs = np.reshape(batch_xs,[-1, 200*200*3])
        batch_ys = label_data[index_rand[i:i+100],:]

        _,loss_, acc = sess.run([train_op,loss, accuracy],feed_dict={tf_x: batch_xs, tf_y: batch_ys})
        print(acc)
        #_,loss_ = sess.run([train_op,loss],{tf_x: batch_xs, tf_y: batch_ys})
        print('Step:', step, '| train loss: %.4f' % loss_)
        #print('epoch:',epoch,'Step:', step, '| train loss: %.4f' % loss, '| test accuracy: %.2f' % acc)