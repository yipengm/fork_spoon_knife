import tensorflow as tf
#input(x,y)
tf.InteractiveSession()
batch_num = 2
R_N_preset = 77


ROI_pred_set = tf.placeholder(tf.float32,[batch_num,R_N_preset,4])   #[batch_num,contain_num,4]

R_N = tf.shape(ROI_pred_set)[1]

x_1 = ROI_pred_set[:,:,0]
y_1 = ROI_pred_set[:,:,1]
x_2 = ROI_pred_set[:,:,2]
y_2 = ROI_pred_set[:,:,3]

x_diff = x_2-x_1
y_diff = y_2-y_1

x_resolution = x_diff/28
y_resolution = y_diff/28


x_list = tf.zeros([tf.size(x_1),14])
y_list = tf.zeros([tf.size(x_1),14])




lincut = tf.linspace(1.0,27.0,14)
lincut_m = tf.tile(tf.reshape(lincut,[14,1]),[1,14])
lincut_m_x = tf.reshape(lincut_m,[1,1,196])
lincut_m_y = tf.reshape(tf.transpose(lincut_m),[1,1,196])



x_increase = tf.reshape(x_resolution,[-1,1])*tf.reshape(lincut_m_x,[1,196])
y_increase = tf.reshape(y_resolution,[-1,1])*tf.reshape(lincut_m_y,[1,196])




x_base = tf.tile(tf.reshape(x_1,[-1,1]),[1,196])
y_base = tf.tile(tf.reshape(y_1,[-1,1]),[1,196])

#x_coor,y_coor are (None,196)
x_coor = tf.reshape(x_increase+x_base,[batch_num,-1,196])
y_coor = tf.reshape(y_increase+y_base,[batch_num,-1,196])

x_coor_t = tf.transpose(x_coor,[2,1,0])
y_coor_t = tf.transpose(y_coor,[2,1,0])

x_coor_2D = tf.reshape(x_coor_t,[-1,batch_num,1])
y_coor_2D = tf.reshape(y_coor_t,[-1,batch_num,1])

bn_coor = tf.reshape(tf.cast(tf.range(batch_num),tf.float32),[1,-1])
bn_coor_2D = tf.reshape(tf.tile(bn_coor,[R_N*196,1]),[-1,batch_num,1])

index_3D = tf.concat([x_coor_2D,y_coor_2D,bn_coor_2D],2)
index = tf.reshape(index_3D,[-1,3])

index_int_00 = tf.cast(index,tf.int32)
index_int_01 = index_int_00 + tf.constant([0,1,0],tf.int32)
index_int_10 = index_int_00 + tf.constant([1,0,0],tf.int32)
index_int_11 = index_int_00 + tf.constant([1,1,0],tf.int32)

delta_down = index[:,0:2] - tf.cast(tf.cast(index[:,0:2],tf.int32),tf.float32)
delta_up = 1-delta_down

a = tf.tile(tf.reshape(delta_down[:,0],[-1,1]),[1,1024])
b = tf.tile(tf.reshape(delta_up[:,0],[-1,1]),[1,1024])
c = tf.tile(tf.reshape(delta_down[:,1],[-1,1]),[1,1024])
d = tf.tile(tf.reshape(delta_up[:,1],[-1,1]),[1,1024])


feature_map = tf.placeholder(tf.float32,[batch_num,50,50,1024])
feature_map_t = tf.transpose(feature_map,[1,2,0,3])





unfolden_14_14_00 = tf.gather_nd(feature_map_t,index_int_00)
unfolden_14_14_01 = tf.gather_nd(feature_map_t,index_int_01)
unfolden_14_14_10 = tf.gather_nd(feature_map_t,index_int_10)
unfolden_14_14_11 = tf.gather_nd(feature_map_t,index_int_11)


tobe_maxpool = a*c*unfolden_14_14_00+b*c*unfolden_14_14_10+a*d*unfolden_14_14_01+b*d*unfolden_14_14_11

tobe_maxpool_14_14_pre = tf.reshape(tobe_maxpool,[14,14,-1,1024])

tobe_maxpool_14_14 = tf.transpose(tobe_maxpool_14_14_pre,[2,0,1,3])

ROIalign_out = tf.nn.max_pool(value = tobe_maxpool_14_14,ksize = [1,2,2,1],strides = [1,2,2,1], padding = 'VALID')