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
from c4 import resnet_v1_101_c4, resnet_v1_101_c1
from ROIalign import ROIalignLayer


slim = tf.contrib.slim

tf.set_random_seed(1)
np.random.seed(1)

BATCH_SIZE = 25
initial_learning_rate = 0.001              
decay_steps = 100
learning_rate_decay_factor = 0.85

N_batch = 777





image = tf.placeholder(tf.float32, [-1, 800, 800, 3])

#Height = tf.shape(image)[1]
#Width = tf.shape(image)[2]

box_true = tf.placeholder(tf.float32, [N_batch,18,50,50])    



log_dir = './log'
checkpoint_file = './resnet_v1_101.ckpt'

with slim.arg_scope([slim.conv2d,slim.fully_connected],normalizer_fn = slim.batch_norm, normalizer_params = {'is_training': True, 'updates_collections':None}):
    feature_map, _ = resnet_v1_101_c4(inputs=image,num_classes=None,global_pool=False,is_training=True)



Dim1024 = tf.layers.conv2d(inputs = feature_map, filters = 1024, kernel_size = 3 , padding = 'same')
score = tf.layers.conv2d(inputs = Dim1024, filters = 18, kernel_size = 1, padding = 'same')
refine = tf.layers.conv2d(inputs = Dim1024, filters = 36, kernel_size = 1, padding = 'same')
soft_input = tf.reshape(score,[-1,50*50*9,2])
exp_soft = tf.exp(soft_input)
sum_exp_soft = tf.reshape(tf.reduce_sum(exp_soft,axis = 2),[-1,50*50*9,1])
prob_matrix = exp_soft/sum_exp_soft
log_matrix = tf.log(prob_matrix)
one_milog_matrix = tf.log(1-prob_matrix)

box_valid = box_true[:,0:9,:,:]
box_overlap = box_true[:,9:18,:,:]
box_valid_tr = tf.transpose(box_valid,(0,2,3,1))
box_overlap_tr = tf.transpose(box_overlap,(0,2,3,1))

box_valid_re = tf.reshape(box_valid_tr,[-1,50*50*9])
box_overlap_re = tf.reshape(box_overlap_tr,[-1,50*50*9])
loss_matrix = -(log_matrix[:,:,0]*box_overlap_re+(1-box_overlap_re)*one_milog_matrix[:,:,0])/tf.reduce_sum(box_valid_re)
loss_cls = tf.reduce_sum(loss_matrix * box_valid_re)

#score  :whether is a background or a foreground
#refine :anchor move



ROI_pred_set = produce_ROI(score,refine)  #TODO, gather all the proper anchor on the 7 by 7 by 1024 feature map then get the corresponding ROI(On the scale of origin pic)

seven_by_seven_1024 = ROIalignLayer(feature_map,ROI_pred_set)# Noticed that the ROI_pred_set is corresponding to the origin pic  

seven_by_seven_2048 = resnet5(seven_by_seven_1024) #this is stride 1 instead of stride 2





avg = tf.nn.avg_pool(value = seven_by_seven_2048,ksize=[1,7,7,1],strides=[1,1,1,1],padding = 'VALID')
fast_cls_logits = tf.layers.dense(inputs = avg,units = 80+1,kernel_initializer = tf.random_normal_initializer(),activation = None)

fast_cls_prob = tf.nn.softmax(fast_cls_logits,name = "ROI class prob")

fast_regr = tf.layers.dense(inputs = avg,units = 4*80,kernel_initializer = tf.random_normal_initializer(),activation = None)






mask_256 = tf.layers.conv2d_transpose(inputs = seven_by_seven_2048,filters=256,strides = [2,2],kernel_size = 2,padding='SAME',activation = tf.nn.relu)
mask_80 = tf.layers.conv2d(inputs = mask_256,filters=80,kernel_size=1,padding='SAME',activation = tf.nn.sigmoid)

















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