#!/usr/bin/env/ python

# MLP class and functions
import numpy as np
import tensorflow as tf
import time
import math
import tflearn
from random import *
from scipy.ndimage.interpolation import rotate
    
def NNN(input_x, input_y,output_size=2,fc_units=[500,200,2],l2_norm=0.01, seed=235):
    
    
    fc_w=[]
    
    
    fc_layer_0=fc_layer(input_x=input_x,in_size=input_x.shape[1],out_size=fc_units[0],rand_seed=seed,activation_function=tf.nn.relu,index=0)
    fc_w.append(fc_layer_0.weight)
    
    
    
    fc_layer_1=fc_layer(input_x=fc_layer_0.output(),in_size=fc_units[0],out_size=fc_units[1],rand_seed=seed,activation_function=tf.nn.relu,index=1)
    fc_w.append(fc_layer_1.weight)
    
    fc_layer_2=fc_layer(input_x=fc_layer_1.output(),in_size=fc_units[1],out_size=fc_units[2],rand_seed=seed,activation_function=None,index=2)
    fc_w.append(fc_layer_2.weight)
    
    
    # loss
    with tf.name_scope("loss"):
        l2_loss = tf.reduce_sum([tf.norm(w) for w in fc_w])

        label = tf.one_hot(input_y, fc_units[2])
        cross_entropy_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=fc_layer_2.output()),
            name='cross_entropy')
        loss = tf.add(cross_entropy_loss, l2_norm * l2_loss, name='loss')

        tf.summary.scalar('CNN_loss', loss)


    return fc_layer_2.output(), loss


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
            mean, variance = tf.nn.moments(cell_out, axes=[0], keep_dims=True)
            cell_out=tf.nn.batch_normalization(cell_out,mean, variance, offset=None,scale=None,variance_epsilon=1e-6,name=None)
            ########################################################################################################
            
            if activation_function is not None:
                cell_out = activation_function(cell_out)
            ########################################################################################################
            # drop out at probs=0.7, dropout more at fc layer because it is further down the network
            
            # same idea as before, use dropout_config variable to determine if dropout
            #if dropout_config == None:
            #    dropout_config = dict()
            #    dropout_config["enabled"] = False
            #if dropout_config['enabled']:
            #    cell_out=tf.nn.dropout(cell_out,keep_prob=0.7,noise_shape=None,seed=None,name=None)
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
    return error_num,pred



def nnn_training(X_train, y_train, X_val, y_val, 
             fc_units=[500,200,100],
             l2_norm=0.01,
             seed=235,
             learning_rate=1e-2,
             epoch=20,
             batch_size=245,
             verbose=False,
             pre_trained_model=None):
    print("Building Network Parameters: ")
    print("fc_units={}".format(fc_units))
    print("l2_norm={}".format(l2_norm))
    print("seed={}".format(seed))
    print("learning_rate={}".format(learning_rate))
    
    num_units=X_train.shape[1]
    
    # define the variables and parameter needed during training
    with tf.name_scope('inputs'):
        xs = tf.placeholder(shape=[None,num_units], dtype=tf.float32)
        ys = tf.placeholder(shape=[None, ], dtype=tf.int64)
        
    #xs,hmm=augment(xs,ys,horizontal_flip=True,rotate=3,crop_probability=0.4) 

    
    output,loss=NNN(xs,ys,output_size=2,fc_units=fc_units,l2_norm=l2_norm, seed=seed)

    iters = int(X_train.shape[0] / batch_size)
    print('number of batches for training: {}'.format(iters))

    step = train_step(loss)
    eve,pred = evaluate(output, ys)

    iter_total = 0
    best_acc = 0
    cur_model_name = 'nnn_{}'.format(int(time.time()))

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
                    
                    #y=graph.get_tensor_by_name("evaluate/ArgMax:0")
                    
                    result=sess.run(pred,feed_dict={xs:X_val})
                    
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
                        store_pred=result
                        store_truelabel=y_val

    print("Traning ends. The best valid accuracy is {}. Model named {}.".format(best_acc, cur_model_name))
    return best_acc,store_trainacc,store_valacc,store_pred,store_truelabel



