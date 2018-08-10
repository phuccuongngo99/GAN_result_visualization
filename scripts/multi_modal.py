#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 09:58:23 2018

@author: root
"""
import json
import argparse
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import OrderedDict
import tensorflow as tf
from keras.models import Model
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

def animate(G,D,epoch,v_animate):
    plt.figure()
    xlist = np.linspace(0, 60, 300)
    ylist = np.linspace(0, 60, 300)
    X, Y = np.meshgrid(xlist, ylist)
    In = np.array(np.meshgrid(xlist,ylist)).T.reshape(-1,2) #preping the suitable input (1st aforementioned type)
    Out = D.predict(In)
    Z = Out.reshape(300,300).T #reshape the output (2nd aforementioned type) back to the grid for isoline plotting
    c = ('#66B2FF','#99CCFF','#CCE5FF','#FFCCCC','#FF9999','#FF6666')
    cp = plt.contourf(X, Y, Z,[0.0,0.2,0.4,0.5,0.6,0.8,1.0],colors=c)
    plt.colorbar(cp)
 
    rx,ry = data_D(G,n_samples=500)[0][:1000].T
    gx,gy = data_D(G,n_samples=500)[0][1000:].T
    #plotting the sample data, generated data
    plt.scatter(rx,ry,color='red')
    plt.scatter(gx,gy,color='blue')
    plt.xlabel('x-axis')
    plt.xlim(0,60)
    plt.ylabel('y-axis')
    plt.ylim(0,60)
    plt.title('Epoch'+str(epoch))
    plt.savefig(str(int(epoch/v_animate))+'.png')
    plt.close()

###Points real and noise distribution
circle_1 = [20,20,8]
circle_2 = [40,50,8]

def real_data(circle,n_samples):
    circle_x = circle[0]
    circle_y = circle[1]
    circle_r = circle[2]
    rd = []
    for i in range(n_samples):
        alpha = 2 * math.pi * np.random.random()
        r = circle_r * np.random.random()
        x = r * math.cos(alpha) + circle_x
        y = r * math.sin(alpha) + circle_y
        rd.append([x,y])
    return np.asarray(rd)

def noise_data(n_samples):
    return np.random.normal(0,8,[n_samples,2])

### Prepare training data to feed to D & G
def data_D(G, n_samples):
    noise = noise_data(n_samples)
    G_out = G.predict(noise)
    
    real_1 = real_data(circle_1,n_samples) #ratio between real, generated data
    real_2 = real_data(circle_2,n_samples)
    
    X = np.concatenate((real_1,real_2,G_out))    
    y = np.zeros(n_samples*3)           
    y[:n_samples*2] = np.random.uniform(0.9,1) #first half is 'good'
    y[n_samples*2:] = np.random.uniform(0,0.1) #second half is 'bullshit'
    
    return X, y

def data_G(n_samples):
    X = noise_data(n_samples)
    y = np.zeros(n_samples)
    y[:] = 0.2    #setting 1 from 'bullshit' data so that (G+D) can backprop to achieve this fake
    return X, y

### Build Discriminator, Generator, GAN networks and custom loss function
def get_generative(lr=5e-3):
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

def get_discriminative(lr=5e-3):
    D_in = Input(shape=(2,))
    x = Dense(15, activation='relu')(D_in)
    x = Dense(15, activation='relu')(x)
    x = Dense(15, activation='relu')(x)
    x = Dense(15, activation='relu')(x)
    D_out = Dense(1, activation='sigmoid')(x)
    D = Model(D_in, D_out)
    dopt = Adam(lr=lr)
    D.compile(loss='binary_crossentropy', optimizer=dopt)
    return D

D = get_discriminative()
#D.summary()

def set_trainability(model, trainable=False): #alternate to freeze D network while training only G in (G+D) combination
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
    
def make_gan(G, D):
    set_trainability(D, False)
    GAN_in = Input(shape=(2,))
    G_out = G(GAN_in)
    GAN_out = D(G_out)
    GAN = Model(GAN_in, GAN_out)
    #GAN.compile(loss='binary_crossentropy', optimizer=G.optimizer)
    GAN.compile(loss=loss_type(G_out,dispersion_weight,round_up), optimizer=G.optimizer)
    return GAN

GAN = make_gan(G,D)
#GAN.summary()

def pretrain(G, D, batch_size=batch_size): #pretrain D
    X, y = data_D(G, batch_size)
    set_trainability(D, True)
    D.fit(X, y, epochs=15, batch_size=batch_size)
pretrain(G, D)
    
def train(GAN, G, D, epochs=epoch, batch_size=batch_size, v_freq=10, v_animate = 10):
    d_loss = []
    g_loss = []
    for epoch in range(epochs):
        try:
            X, y = data_D(G,batch_size)
            set_trainability(D, True)
            d_loss.append(D.train_on_batch(X, y))
            
            X, y = data_G(batch_size)
            set_trainability(D, False)
            g_loss.append(GAN.train_on_batch(X, y))
            
            if (epoch + 1) % v_freq == 0:
                print("Epoch #{}: Generative Loss: {}, Discriminative Loss: {}".format(epoch + 1, g_loss[-1], d_loss[-1]))
            if epoch % v_animate == 0:
                animate(G,D,epoch,v_animate)
        except KeyboardInterrupt: #hit control-C to exit and save video there
            break
    G.save('generator.h5')
    return d_loss, g_loss

d_loss, g_loss= train(GAN, G, D)

#plotting loss graph
ax = pd.DataFrame(
    {
        'Generative Loss': g_loss,
        'Discriminative Loss': d_loss,
    }
).plot(title='Training loss')
ax.set_xlabel("Epochs")
ax.set_ylabel("Loss")
fig = ax.get_figure()
fig.savefig('./DG.jpg')

import json
file = open('loss.json','r')
loss = json.loads(file.read())
ax = pd.DataFrame(
    {
        list(loss.keys())[0]: loss[list(loss.keys())[0]],
        list(loss.keys())[1]: loss[list(loss.keys())[1]],
    }
).plot(title='Loss')
ax.set_xlabel("Epochs")
ax.set_ylabel("Loss")
fig = ax.get_figure()
fig.savefig('./loss_graph.jpg')

result = [("gif","result.gif"),("loss_graph","loss_graph.jpg")]

with open('result.json','w+') as outfile:
    json.dump(OrderedDict(result), outfile, indent=4)