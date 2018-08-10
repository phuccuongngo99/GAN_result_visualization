#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 10:38:45 2018

@author: root
"""
import json
import sys
import tensorflow as tf
from keras import losses

loss_dict = {'dispersion_loss':[],'binary_loss':[]}

def tf_print(op, tensors, message=None):
    def print_message(x):
        sys.stdout.write(message + " %s\n" % x)
        loss_dict[message].append(float(x))
        with open('loss.json','w+') as file:
            json.dump(loss_dict, file, indent=4)
        return x

    prints = [tf.py_func(print_message, [tensor], tensor.dtype) for tensor in tensors]
    with tf.control_dependencies(prints):
        op = tf.identity(op)
    return op

### Average distance from the Center of Mass
def com(G_out, weight, round_up):
    def dispersion_loss(y_true, y_pred):
        binary_loss = losses.binary_crossentropy(y_true, y_pred)
        
        center = tf.reduce_mean(G_out)
        distance_xy = tf.square(tf.subtract(G_out,center))
        distance = tf.reduce_sum(distance_xy, 1)
        avg_distance = tf.reduce_mean(tf.sqrt(distance))
        dispersion_loss = tf.reciprocal(avg_distance)
        
        loss = binary_loss + weight*dispersion_loss
        return loss
    return dispersion_loss

### Average electric force 1/r^2 between any pair
def electric(G_out, weight, round_up):
    def dispersion_loss(y_true, y_pred):
        loss_b = tf.reduce_mean(losses.binary_crossentropy(y_true, y_pred))
        
        r = tf.reduce_sum(G_out*G_out, 1)
        # turn r into column vector
        r = tf.reshape(r, [-1, 1])
        D = r - 2*tf.matmul(G_out, tf.transpose(G_out)) + tf.transpose(r)
        D = tf.cast(D, dtype=tf.float32)
        #Round up to a certain extent
        D = tf.maximum(D,1e-4)
        D + 1e6*tf.identity(D)
        D = tf.sqrt(D)
        D = tf.maximum(D,round_up)
        electric_force = tf.reciprocal(tf.pow(D,2))
        loss_d = tf.reduce_mean(electric_force)
        
        loss_b = tf_print(loss_b, [loss_b], "binary_loss")
        loss_d = tf_print(loss_d, [loss_d], "dispersion_loss")
        
        loss = loss_b + weight*loss_d
        return loss
    return dispersion_loss

### Average Lennard-Jones potential between any pair
### I just used the 1/r^6 - 1/r^3
def neutral(G_out, weight, round_up):
    def dispersion_loss(y_true, y_pred):
        loss_b = tf.reduce_mean(losses.binary_crossentropy(y_true, y_pred))
        
        r = tf.reduce_sum(G_out*G_out, 1)
        # turn r into column vector
        r = tf.reshape(r, [-1, 1])
        D = r - 2*tf.matmul(G_out, tf.transpose(G_out)) + tf.transpose(r) 
        D = tf.cast(D, dtype=tf.float32)
        #Round up distance to certain extent, if not when distance tend to zero, loss to infinity
        D = tf.maximum(D,1e-4)
        D = D + 1e6*tf.identity(D)
        D = tf.sqrt(D)
        D = tf.maximum(D,round_up)
        len_jon = tf.reciprocal(tf.pow(D,6)) - tf.reciprocal(tf.pow(D,3))
        loss_d = tf.reduce_mean(len_jon)
        
        #tensorflow
        loss_b = tf_print(loss_b, [loss_b], "binary_loss")
        loss_d = tf_print(loss_d, [loss_d], "dispersion_loss")
        
        loss = loss_b + weight*loss_d
        return loss
    return dispersion_loss
