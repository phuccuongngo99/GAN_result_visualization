#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 19:10:05 2018

@author: root
"""
import os
import math
import json
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Model
from collections import OrderedDict
from keras.layers import Input, Add
from keras.layers.core import Dense
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, SGD
from keras.callbacks import TensorBoard
from keras import regularizers
from keras import losses

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(1)

parser = argparse.ArgumentParser('Train your GAN')
parser.add_argument(
    '-c',
    '--conf',
    help='path to configuration file')
args = parser.parse_args()
config_path = args.conf
#"""

config = json.loads(open((config_path),'r').read())

epoch = config['epoch']
batch_size = config['batch_size']
dispersion_weight = config['dispersion_weight']
loss_type = config['loss_type']
round_up = config['round_up']
l2_reg = config['l2_reg']

if not os.path.exists('./pictures'):
    os.makedirs('./pictures')

def animate(G,D,epoch,v_animate):
    plt.figure()
    xlist = np.linspace(0, 60, 60)
    ylist = np.linspace(0, 60, 60)
    X, Y = np.meshgrid(xlist, ylist)
    In = np.array(np.meshgrid(xlist,ylist)).T.reshape(-1,2) #preping the suitable input (1st aforementioned type)
    Out = D.predict(In)
    Z = Out.reshape(60,60).T #reshape the output (2nd aforementioned type) back to the grid for isoline plotting
    c = ('#66B2FF','#99CCFF','#CCE5FF','#FFCCCC','#FF9999','#FF6666')
    cp = plt.contourf(X, Y, Z,[0.0,0.2,0.4,0.5,0.6,0.8,1.0],colors=c)
    plt.colorbar(cp)
 
    rx,ry = data_D(G,500)[0][:500].T
    gx,gy = data_D(G,500)[0][500:].T
    #plotting the sample data, generated data
    plt.scatter(rx,ry,color='red')
    plt.scatter(gx,gy,color='blue')
    plt.xlabel('x-axis')
    plt.xlim(0,60)
    plt.ylabel('y-axis')
    plt.ylim(0,60)
    plt.title('Epoch'+str(epoch))
    plt.savefig('pictures/'+str(int(epoch/v_animate))+'.png')
    plt.close()


def real_data(n):
    circle_r = 10
    circle_x = 30
    circle_y = 30
    rd = []
    for i in range(n):
        alpha = 2 * math.pi * np.random.random()
        r = circle_r * np.random.random()
        x = r * math.cos(alpha) + circle_x
        y = r * math.sin(alpha) + circle_y
        rd.append([x,y])
    return np.asarray(rd)

def noise_data(n):
    return np.random.normal(0,8,[n,2])
    
def data_D(G, batch_size):
    x_r = real_data(batch_size)
    x_g = G.predict(noise_data(batch_size))
    
    X = np.concatenate((x_r,x_g))    
    y = np.zeros(batch_size*2)           
    y[:batch_size] = 1
    y[batch_size:] = 0
    #y[:batch_size] = np.random.uniform(0.9,1) #first half is 'good'
    #y[batch_size:] = np.random.uniform(0,0.1) #second half is 'bullshit'
    return X, y

def data_G(batch_size):
    X = noise_data(batch_size)
    y = np.zeros(batch_size)
    y[:] = 0.2    #setting 1 from 'bullshit' data so that (G+D) can backprop to achieve this fake
    return X, y

def get_generative(lr=1e-3):
    G_in = Input(shape=(2,))
    x = Dense(10, activation='relu')(G_in)
    x = Dense(10, activation='relu')(x)
    #G_out = Dense(2)(x)
    x = Dense(2)(x)
    G_out = Add()([G_in,x])
    G = Model(G_in, G_out)
    opt = Adam(lr=lr)
    G.compile(loss='binary_crossentropy', optimizer=opt)
    return G

G = get_generative()
#G.summary()

def get_discriminative(lr=1e-3):
    D_in = Input(shape=(2,))
    x = Dense(15, activation='relu', kernel_regularizer=regularizers.l2(l2_reg))(D_in)
    x = Dense(15, activation='relu', kernel_regularizer=regularizers.l2(l2_reg))(x)
    #x = Dense(15, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    #x = Dense(15, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    D_out = Dense(1, activation='sigmoid')(x)
    D = Model(D_in, D_out)
    #dopt = Adam(lr=lr)
    dopt = Adam(lr=lr)
    D.compile(loss='binary_crossentropy', optimizer=dopt)
    return D

D = get_discriminative()
#D.summary()

def set_trainability(model, trainable=False):
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable


if loss_type == 'electric':
    from custom_losses import electric
    loss_type = electric
elif loss_type == 'com':
    from custom_losses import com
    loss_type = com
elif loss_type == 'neutral':
    from custom_losses import neutral
    loss_type = neutral

    
def make_gan(G, D):  #making (G+D) framework
    set_trainability(D, False)
    GAN_in = Input(shape=(2,))
    G_out = G(GAN_in)
    GAN_out = D(G_out)
    GAN = Model(GAN_in, GAN_out)
    #GAN.compile(loss=loss_type(G_out,dispersion_weight,round_up), optimizer=G.optimizer)
    GAN.compile(loss='binary_crossentropy', optimizer=G.optimizer)
    return GAN

GAN = make_gan(G,D)
#GAN.summary()

def pretrain(G, D, batch_size=batch_size): #pretrain D
    X, y = data_D(G, batch_size=batch_size)
    set_trainability(D, True)
    D.fit(X, y, epochs=20, batch_size=batch_size)

pretrain(G, D)
    
def train(GAN, G, D, epochs=epoch, batch_size=batch_size, v_freq=50, v_animate=500):
    d_loss = []
    g_loss = []
#    data_show = sample_noise(n_samples=n_samples)[0]
    for epoch in range(epochs):
        try:
            d_count = 0
            X, y = data_D(G,batch_size)
            set_trainability(D, True)
            discriminator_loss = D.train_on_batch(X,y)
            """
            while discriminator_loss > 0.05:
                d_count += 1
                print('d_loss: ',discriminator_loss)
                X, y = data_D(G,batch_size)
                set_trainability(D, True)
                discriminator_loss = D.train_on_batch(X,y)
                if d_count > 10000:
                    break
            #"""
            d_loss.append(discriminator_loss)
            
            g_count = 0
            X, y = data_G(batch_size)
            set_trainability(D, False)
            generator_loss = GAN.train_on_batch(X,y)
            """
            while generator_loss >= max(0.6-epoch*(10**-1.2),0.55):
                g_count += 1
                print('g_loss: ',generator_loss)
                X, y = data_G(batch_size)
                set_trainability(D, False)
                generator_loss = GAN.train_on_batch(X,y)
                if g_count > 10000:
                    break
            #"""
            g_loss.append(generator_loss)

            #print('Epoch: ', epoch,'D_count: ',d_count, 'G_count: ',g_count)
            if (epoch + 1) % v_freq == 0:
                print("Epoch #{}: Generative Loss: {}, Discriminative Loss: {}".format(epoch + 1, g_loss[-1], d_loss[-1]))
            if epoch % v_animate == 0:
                animate(G,D,epoch,v_animate)
        except KeyboardInterrupt: #hit control-C to exit and save video there
            break
    G.save('Circle_G.h5')
    D.save('Circle_D.h5')   
    return d_loss, g_loss

d_loss, g_loss = train(GAN, G, D)

#Figure 1
fig1 = plt.figure()
plt.plot(g_loss, label='Generative _Loss')
plt.plot(d_loss, label='Discriminative_Loss')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('./DG.jpg')
plt.close(fig1)

"""
#Figure 2
import json
file = open('loss.json','r')
loss = json.loads(file.read())

fig2 = plt.figure()
plt.plot(loss[list(loss)[0]],'r-', label=list(loss)[0])
plt.plot(loss[list(loss)[1]],'b-', label=list(loss)[1])
plt.title('Generative Losses')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('./loss_graph.jpg')
plt.close(fig2)

result = [("gif","result.gif"),("loss_graph","loss_graph.jpg")]
#"""
result = [("gif","result.gif")]

with open('result.json','w+') as outfile:
    json.dump(OrderedDict(result), outfile, indent=4)

import os
import subprocess
cwd = os.getcwd()
def make_gif(path):
    count = 0
    for file in os.listdir(path):
        if file.endswith('.png'):
            count += 1
    input_frame = count//10 #20 seconds mah
    subprocess.call(['ffmpeg', '-f', 'image2', '-framerate', str(input_frame), '-i', '%d.png', '../result.gif'], cwd=os.path.join(cwd,path))
    subprocess.call(['ffmpeg', '-f', 'image2', '-framerate', str(input_frame), '-i', '%d.png', '../result.mp4'], cwd=os.path.join(cwd,path))
    
    for file in os.listdir(path):
        if file.endswith('.png'):
            os.remove(path+'/'+file)
            
make_gif('./pictures')