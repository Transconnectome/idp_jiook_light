#!/usr/bin/env/ python

# MLP class and functions
import numpy as np
import tensorflow as tf
import time

    
def CNN(input_x, input_y,
          img_len=164, channel_num=2, output_size=2,
          conv_featmap=[6, 16, 32], fc_units=[500,200,100],
          conv_kernel_size=[5, 5], pooling_size=[2, 2],
          l2_norm=0.01, seed=235):
    
    # conv layer
    conv_layer_0 = conv_layer(input_x=input_x,
                              in_channel=channel_num,
                              out_channel=conv_featmap[0],
                              kernel_shape=conv_kernel_size[0],
                              rand_seed=seed)
    
    conv_layer_1 = conv_layer(input_x=conv_layer_0.output(),
                              in_channel=conv_featmap[0],
                              out_channel=conv_featmap[1],
                              kernel_shape=conv_kernel_size[0],
                              rand_seed=seed,index=1)
    
    pooling_layer_1 = max_pooling_layer(input_x=conv_layer_1.output(),
                                        k_size=pooling_size[0],
                                        padding="VALID")
    
    # new conv layer to test
    conv_layer_2 = conv_layer(input_x=pooling_layer_1.output(),
                              in_channel=conv_featmap[1],
                              out_channel=conv_featmap[2],
                              kernel_shape=conv_kernel_size[0],
                              rand_seed=seed,index=2)
    
    pooling_layer_2 = max_pooling_layer(input_x=conv_layer_2.output(),
                                        k_size=pooling_size[0],
                                        padding="VALID")
    
    pool_shape=pooling_layer_2.output().get_shape()
    img_vector_length = pool_shape[1].value * pool_shape[2].value * pool_shape[3].value
    flatten = tf.reshape(pooling_layer_2.output(), shape=[-1, img_vector_length])
    
    
    fc_w=[]
    
    dropout_config0={"enabled":True}
    fc_layer_0=fc_layer(input_x=flatten,in_size=img_vector_length,out_size=fc_units[0],rand_seed=seed,activation_function=tf.nn.relu,index=0,dropout_config=dropout_config0)
    fc_w.append(fc_layer_0.weight)
    
    
    
    fc_layer_1=fc_layer(input_x=fc_layer_0.output(),in_size=fc_units[0],out_size=fc_units[1],rand_seed=seed,activation_function=tf.nn.relu,index=1)
    fc_w.append(fc_layer_1.weight)

    
    conv_w=[conv_layer_0.weight,conv_layer_1.weight,conv_layer_2.weight]
    
    
    # loss
    with tf.name_scope("loss"):
        l2_loss = tf.reduce_sum([tf.norm(w) for w in fc_w])
        l2_loss += tf.reduce_sum([tf.norm(w, axis=[-2, -1]) for w in conv_w])

        label = tf.one_hot(input_y, fc_units[1])
        cross_entropy_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=fc_layer_1.output()),
            name='cross_entropy')
        loss = tf.add(cross_entropy_loss, l2_norm * l2_loss, name='loss')

        tf.summary.scalar('CNN_loss', loss)


    return fc_layer_1.output(), loss

class conv_layer(object):
    def __init__(self, input_x, in_channel, out_channel, kernel_shape, rand_seed, index=0,dropout_config=None):
        """
        :param input_x: The input of the conv layer. Should be a 4D array like (batch_num, img_len, img_len, channel_num)
        :param in_channel: The 4-th demension (channel number) of input matrix. For example, in_channel=3 means the input contains 3 channels.
        :param out_channel: The 4-th demension (channel number) of output matrix. For example, out_channel=5 means the output contains 5 channels (feature maps).
        :param kernel_shape: the shape of the kernel. For example, kernal_shape = 3 means you have a 3*3 kernel.
        :param rand_seed: An integer that presents the random seed used to generate the initial parameter value.
        :param index: The index of the layer. It is used for naming only.
        """
        assert len(input_x.shape) == 4 and input_x.shape[1] == input_x.shape[2] and input_x.shape[3] == in_channel

        with tf.variable_scope('conv_layer_%d' % index):
            with tf.name_scope('conv_kernel'):
                w_shape = [kernel_shape, kernel_shape, in_channel, out_channel]
                weight = tf.get_variable(name='conv_kernel_%d' % index, shape=w_shape,initializer=tf.glorot_uniform_initializer(seed=rand_seed))
                self.weight = weight

            with tf.variable_scope('conv_bias'):
                b_shape = [out_channel]
                bias = tf.get_variable(name='conv_bias_%d' % index, shape=b_shape,initializer=tf.glorot_uniform_initializer(seed=rand_seed))
                self.bias = bias

            conv_out = tf.nn.conv2d(input_x, weight, strides=[1, 1, 1, 1], padding="SAME")
            
            ###########################################################################################################
            # Batch normalization, this happens to all convolution layers. 
            # It seems like a good idea to have normalization at every step
            
            # get mean and var with tf.nn.moments func
            #mean, variance = tf.nn.moments(conv_out, axes=[0], keep_dims=True) 
            # perform batch normalization
            #batch_out=tf.nn.batch_normalization(conv_out,mean, variance, offset=None,scale=None,variance_epsilon=1e-6,name=None)
            ########################################################################################################
            cell_out=tf.nn.relu(conv_out+bias)
            #cell_out = tf.nn.relu(batch_out + bias)
            
            ########################################################################################################
            # drop out at probs=0.9. Kept this at a flat 0.9 because it seems wise to
            # keep a majority of the neurons while at the convolution step. 
            
            # copied from MLP portion. if dropout_config has no element, don't drop out. if do, dropout
            if dropout_config == None:
                dropout_config = dict()
                dropout_config["enabled"] = False
            if dropout_config['enabled']:
                cell_out=tf.nn.dropout(cell_out,keep_prob=0.9,noise_shape=None,seed=None,name=None)
            ########################################################################################################
            
            self.cell_out = cell_out

            tf.summary.histogram('conv_layer/{}/kernel'.format(index), weight)
            tf.summary.histogram('conv_layer/{}/bias'.format(index), bias)

    def output(self):
        return self.cell_out

# max pool layer from sample code
class max_pooling_layer(object):
    def __init__(self, input_x, k_size, padding="SAME"):
        """
        :param input_x: The input of the pooling layer.
        :param k_size: The kernel size you want to behave pooling action.
        :param padding: The padding setting. Read documents of tf.nn.max_pool for more information.
        """
        with tf.variable_scope('max_pooling'):
            # strides [1, k_size, k_size, 1]
            pooling_shape = [1, k_size, k_size, 1]
            cell_out = tf.nn.max_pool(input_x, strides=pooling_shape,
                                      ksize=pooling_shape, padding=padding)
            
            self.cell_out = cell_out

    def output(self):
        return self.cell_out

class fc_layer(object):
    def __init__(self, input_x, in_size, out_size, rand_seed, activation_function=None, index=0,dropout_config=None):
        """
        :param input_x: The input of the FC layer. It should be a flatten vector.
        :param in_size: The length of input vector.
        :param out_size: The length of output vector.
        :param rand_seed: An integer that presents the random seed used to generate the initial parameter value.
        :param keep_prob: The probability of dropout. Default set by 1.0 (no drop-out applied)
        :param activation_function: The activation function for the output. Default set to None.
        :param index: The index of the layer. It is used for naming only.

        """
        with tf.variable_scope('fc_layer_%d' % index):
            with tf.name_scope('fc_kernel'):
                w_shape = [in_size, out_size]
                weight = tf.get_variable(name='fc_kernel_%d' % index, shape=w_shape,initializer=tf.glorot_uniform_initializer(seed=rand_seed))
                self.weight = weight

            with tf.variable_scope('fc_kernel'):
                b_shape = [out_size]
                bias = tf.get_variable(name='fc_bias_%d' % index, shape=b_shape,
                                       initializer=tf.glorot_uniform_initializer(seed=rand_seed))
                self.bias = bias

            cell_out = tf.add(tf.matmul(input_x, weight), bias)
            ###########################################################################################################
            # Batch normalization, same reasoning as conv layer
            #mean, variance = tf.nn.moments(cell_out, axes=[0], keep_dims=True)
            #cell_out=tf.nn.batch_normalization(cell_out,mean, variance, offset=None,scale=None,variance_epsilon=1e-6,name=None)
            ########################################################################################################
            
            if activation_function is not None:
                cell_out = activation_function(cell_out)
            ########################################################################################################
            # drop out at probs=0.7, dropout more at fc layer because it is further down the network
            
            # same idea as before, use dropout_config variable to determine if dropout
            if dropout_config == None:
                dropout_config = dict()
                dropout_config["enabled"] = False
            if dropout_config['enabled']:
                cell_out=tf.nn.dropout(cell_out,keep_prob=0.7,noise_shape=None,seed=None,name=None)
            ########################################################################################################
            
            self.cell_out = cell_out

            tf.summary.histogram('fc_layer/{}/kernel'.format(index), weight)
            tf.summary.histogram('fc_layer/{}/bias'.format(index), bias)

    def output(self):
        return self.cell_out

def cross_entropy(output, input_y):
    with tf.name_scope('cross_entropy'):
        label = tf.one_hot(input_y, 10)
        ce = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=output))

    return ce


def train_step(loss, learning_rate=1e-3):
    with tf.name_scope('train_step'):
        step = tf.train.AdamOptimizer(learning_rate).minimize(loss) 

    return step


def evaluate(output, input_y):
    with tf.name_scope('evaluate'):
        pred = tf.argmax(output, axis=1)
        error_num = tf.count_nonzero(pred - input_y, name='error_num')
        tf.summary.scalar('cnn_error_num', error_num)
    return error_num



def adni_training(X_train, y_train, X_val, y_val, 
             fc_units=[500,200,100], conv_featmap=[6,16,32],
             l2_norm=0.01,
             seed=235,
             learning_rate=1e-2,
             epoch=20,
             batch_size=245,
             verbose=False,
             pre_trained_model=None,imglen=164):
    print("Building Network Parameters: ")
    print("fc_units={}".format(fc_units))
    print("l2_norm={}".format(l2_norm))
    print("seed={}".format(seed))
    print("learning_rate={}".format(learning_rate))

    # define the variables and parameter needed during training
    with tf.name_scope('inputs'):
        xs = tf.placeholder(shape=[None,imglen,imglen,2], dtype=tf.float32)
        ys = tf.placeholder(shape=[None, ], dtype=tf.int64)
        
    output,loss=CNN(xs,ys,img_len=imglen, channel_num=2, output_size=2,conv_featmap=conv_featmap,fc_units=fc_units,conv_kernel_size=[5, 5], pooling_size=[2,2],l2_norm=l2_norm, seed=seed)

    iters = int(X_train.shape[0] / batch_size)
    print('number of batches for training: {}'.format(iters))

    step = train_step(loss)
    eve = evaluate(output, ys)

    iter_total = 0
    best_acc = 0
    cur_model_name = 'cnn_{}'.format(int(time.time()))

    with tf.Session() as sess:
        merge = tf.summary.merge_all()

        writer = tf.summary.FileWriter("log/{}".format(cur_model_name), sess.graph)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        # try to restore the pre_trained
        if pre_trained_model is not None:
            try:
                print("Load the model from: {}".format(pre_trained_model))
                saver.restore(sess, 'model/{}'.format(pre_trained_model))
            except Exception:
                print("Load model Failed!")
                pass
        
        
        store_trainacc=[]
        store_valacc=[]
        for epc in range(epoch):
            print("epoch {} ".format(epc + 1))

            for itr in range(iters):
                iter_total += 1

                training_batch_x = X_train[itr * batch_size: (1 + itr) * batch_size]
                training_batch_y = y_train[itr * batch_size: (1 + itr) * batch_size]

                _, cur_loss = sess.run([step, loss], feed_dict={xs: training_batch_x, ys: training_batch_y})

                if itr == iters-1:
                    # do validation
                    train_eve=sess.run(eve,feed_dict={xs:X_train,ys:y_train})
                    train_acc=100-train_eve*100/y_train.shape[0]
                    store_trainacc.append(train_acc)
                    valid_eve, merge_result = sess.run([eve, merge], feed_dict={xs: X_val, ys: y_val})
                    valid_acc = 100 - valid_eve * 100 / y_val.shape[0]
                    store_valacc.append(valid_acc)
                    
                    if verbose:
                        print('loss: {} validation accuracy : {}%'.format(
                            cur_loss,
                            valid_acc))

                    # save the merge result summary
                    writer.add_summary(merge_result, iter_total)

                    # when achieve the best validation accuracy, we store the model paramters
                    if valid_acc > best_acc:
                        print('Best validation accuracy! iteration:{} accuracy: {}%'.format(iter_total, valid_acc))
                        best_acc = valid_acc
                        saver.save(sess, 'model/{}'.format(cur_model_name))

    print("Traning ends. The best valid accuracy is {}. Model named {}.".format(best_acc, cur_model_name))
    return best_acc,store_trainacc,store_valacc









