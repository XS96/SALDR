#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Reshape, Conv2D, add, LeakyReLU, ReLU, Lambda, dot
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation
import scipy.io as sio
import numpy as np
import math
import time
import os
# from clr_callback import CyclicLR



# tf.compat.v1.reset_default_graph()
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True   # 不全部占满显存, 按需分配
# sess = tf.compat.v1.Session(config=config)
# tf.compat.v1.keras.backend.set_session(sess)

envir = 'indoor'  # 'indoor' or 'outdoor'
# image params
img_height = 32
img_width = 32
img_channels = 2
img_total = img_height * img_width * img_channels  # 2048
# network params
encoded_dim1 = 256  # compress rate=1/4->dim.=512

batchsize = 200


def residual_block_decoded(y):
    y = Conv2D(32, (3, 3), padding='same', data_format="channels_first")(y)
    y = BatchNormalization()(y)
    y = LeakyReLU()(y)

    y = Conv2D(16, (3, 3), padding='same', data_format="channels_first")(y)
    y = BatchNormalization()(y)
    y = LeakyReLU()(y)

    y = Conv2D(8, (3, 3), padding='same', data_format="channels_first")(y)
    y = BatchNormalization()(y)
    y = LeakyReLU()(y)
    return y


def DenseRefine(x):
    ip = x
    x1 = ip
    x = residual_block_decoded(ip)
    x2 = x
    x = keras.layers.concatenate([x1, x2], axis=1)

    x = residual_block_decoded(x)
    x3 = x
    x = keras.layers.concatenate([x1, x2, x3], axis=1)

    x = residual_block_decoded(x)
    x4 = x
    x = keras.layers.concatenate([x1, x2, x3, x4], axis=1)

    x = residual_block_decoded(x)
    x5 = x
    x = keras.layers.concatenate([x1, x2, x3, x4, x5], axis=1)

    x = residual_block_decoded(x)
    x6 = x
    x = keras.layers.concatenate([x1, x2, x3, x4, x5, x6], axis=1)
    return x


# Bulid the autoencoder model of CsiNet
def residual_network(x):
    # encoder Net
    x = Conv2D(2, (3, 3), padding='same', data_format="channels_first")(x)   #
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Reshape((img_total,), name='encoder_reshape1')(x)
    encoded1 = Dense(encoded_dim1, activation='linear', name='encoder_cr8_dense')(x)

    # decoder
    x1 = Dense(img_total, activation='linear', name='decoder_cr8_dense')(encoded1)

    # decoder Net
    x1 = Reshape((img_channels, img_height, img_width,))(x1)  # reshape to real channel and im channel
    x1 = DenseRefine(x1)
    x1 = Conv2D(2, (3, 3), activation='sigmoid', padding='same', data_format="channels_first")(x1)
    return x1


image_tensor = Input(shape=(img_channels, img_height, img_width))
network_output = residual_network(image_tensor)
autoencoder = Model(inputs=[image_tensor], outputs=[network_output])
autoencoder.compile(optimizer='adam', loss='mse')
print(autoencoder.summary())

outfile = './CsiNetv2_result/ck_DRCsiNet_indoor25604-19_23-19.hdf5'

autoencoder.load_weights(outfile, by_name=True)

# Data loading
if envir == 'indoor':
    mat = sio.loadmat('./data/DATA_Htrainin.mat')
    x_train = mat['HT']  # array     100000*2048
    mat = sio.loadmat('./data/DATA_Hvalin.mat')
    x_val = mat['HT']  # array         30000*2048
    mat = sio.loadmat('./data/DATA_Htestin.mat')
    x_test = mat['HT']  # array     20000*2048

elif envir == 'outdoor':
    mat = sio.loadmat('./data/DATA_Htrainout.mat')
    x_train = mat['HT']  # array
    mat = sio.loadmat('./data/DATA_Hvalout.mat')
    x_val = mat['HT']  # array
    mat = sio.loadmat('./data/DATA_Htestout.mat')
    x_test = mat['HT']  # array

x_train = x_train.astype('float32')
x_val = x_val.astype('float32')
x_test = x_test.astype('float32')
x_train = np.reshape(x_train, (
    len(x_train), img_channels, img_height, img_width))  # adapt this if using `channels_first` image data format
x_val = np.reshape(x_val, (
    len(x_val), img_channels, img_height, img_width))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (
    len(x_test), img_channels, img_height, img_width))  # adapt this if using `channels_first` image data format


# Testing data
tStart = time.time()
x_hat = autoencoder.predict(x_test)
tEnd = time.time()
print("It cost %f sec" % ((tEnd - tStart) / x_test.shape[0]))  # calculate the time of recontribute the CSI for
# every channel matrix


# Calcaulating the NMSE and rho
if envir == 'indoor':
    mat = sio.loadmat('./data/DATA_HtestFin_all.mat')
    X_test = mat['HF_all']  # array     20000*4000 complex   4000=32*125   non_truncated data?

elif envir == 'outdoor':
    mat = sio.loadmat('./data/DATA_HtestFout_all.mat')
    X_test = mat['HF_all']  # array

X_test = np.reshape(X_test, (len(X_test), img_height, 125))         # 20000*32*125
x_test_real = np.reshape(x_test[:, 0, :, :], (len(x_test), -1))         # 20000*1024
x_test_imag = np.reshape(x_test[:, 1, :, :], (len(x_test), -1))
x_test_C = x_test_real - 0.5 + 1j * (x_test_imag - 0.5)         # recover complex,  why subtract 0.5
x_hat_real = np.reshape(x_hat[:, 0, :, :], (len(x_hat), -1))
x_hat_imag = np.reshape(x_hat[:, 1, :, :], (len(x_hat), -1))
x_hat_C = x_hat_real - 0.5 + 1j * (x_hat_imag - 0.5)
x_hat_F = np.reshape(x_hat_C, (len(x_hat_C), img_height, img_width))        # 20000*32*32
# zero fill subcarrier and do FFT on subcarrier axis
X_hat = np.fft.fft(np.concatenate((x_hat_F, np.zeros((len(x_hat_C), img_height, 257 - img_width))), axis=2), axis=2)
X_hat = X_hat[:, :, 0:125]          # 20000*32*125

# caculate the cosine similarity
n1 = np.sqrt(np.sum(np.conj(X_test) * X_test, axis=1))
n1 = n1.astype('float64')
n2 = np.sqrt(np.sum(np.conj(X_hat) * X_hat, axis=1))
n2 = n2.astype('float64')
aa = abs(np.sum(np.conj(X_test) * X_hat, axis=1))
rho = np.mean(aa / (n1 * n2), axis=1)

X_hat = np.reshape(X_hat, (len(X_hat), -1))
X_test = np.reshape(X_test, (len(X_test), -1))
power = np.sum(abs(x_test_C) ** 2, axis=1)
power_d = np.sum(abs(X_hat) ** 2, axis=1)
mse = np.sum(abs(x_test_C - x_hat_C) ** 2, axis=1)

print("In " + envir + " environment")
print("When dimension is", encoded_dim1)
print("CsiNET: NMSE is ", 10 * math.log10(np.mean(mse / power)))
print("CsiNet: Correlation is ", np.mean(rho))




