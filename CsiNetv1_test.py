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

# Enable GPU
tf.compat.v1.reset_default_graph()
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True   
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)


# image params
img_height = 32
img_width = 32
img_channels = 2
img_total = img_height * img_width * img_channels  # 2048
# network params
envir = 'indoor'  # 'indoor' or 'outdoor'
encoded_dim1 = 256  #
residual_num = 2

batchsize = 200

class UnfoldLayer(tf.keras.layers.Layer):
    def __init__(self, Kh, Kw, stride=1, **kwargs):
        self.Kh = Kh
        self.Kw = Kw
        self.stride = stride
        super(UnfoldLayer, self).__init__(**kwargs)

    def call(self, x, **kwargs):
        return self.Unfold(x)

    def Unfold(self, x):
        _, C, H, W = x.shape
        Co = C * self.Kw * self.Kh
        x_out = x[:, :, 0:self.Kh, 0:self.Kw]
        x_out = tf.reshape(x_out, [-1, Co, 1])
        for i in range(H - (self.Kh - 1)):
            for j in range(W - (self.Kw - 1)):
                Hstart = i * self.stride
                Hend = Hstart + self.Kh
                Wstart = j * self.stride
                Wend = Wstart + self.Kw
                xi = x[:, :, Hstart:Hend, Wstart:Wend]
                xi = tf.reshape(xi, [-1, Co, 1])
                x_out = tf.concat([x_out, xi], axis=2)
        x_out = x_out[:, :, 1:]
        return x_out

    def get_config(self):
        base_config = super(UnfoldLayer, self).get_config()
        base_config['Kw'] = self.Kw
        return base_config


def unfold(x, kh=3, kw=3):
    x = tf.transpose(x, perm=[0, 2, 3, 1])
    x = tf.image.extract_patches(
        images=x,
        sizes=[1, kh, kw, 1],
        strides=[1, 1, 1, 1],
        rates=[1, 1, 1, 1],
        padding="VALID",
    )
    x = Reshape((1024, 36))(x)
    x = tf.transpose(x, perm=[0, 2, 1])
    return x


def SA_block(y, k, rel_planes, mid_planes, out_planes):
    # pre
    shortcut = y
    y = BatchNormalization()(y)
    y = ReLU()(y)
    # y1
    y1 = Conv2D(rel_planes, (1, 1), padding='same', use_bias=False, data_format='channels_first',
                kernel_initializer='he_normal', name='SA_y1conv')(y)
    y1 = Reshape((rel_planes, 1, -1,))(y1)

    # y2
    y2 = Conv2D(rel_planes, (1, 1), padding='same', use_bias=False, data_format='channels_first',
                kernel_initializer='he_normal', name='SA_y2conv')(y)
    y2 = Lambda(lambda e: tf.pad(e, [[0, 0], [0, 0], [1, 1], [1, 1]], mode='SYMMETRIC'))(y2)
    # y2 = UnfoldLayer(Kh=k, Kw=k, stride=1)(y2)
    y2 = unfold(y2, kh=k, kw=k)
    y2 = Lambda(lambda e: tf.expand_dims(e, axis=2))(y2)

    # y3
    y3 = Conv2D(mid_planes, (1, 1), padding='same', use_bias=False, data_format='channels_first',
                kernel_initializer='he_normal', name='SA_y3conv')(y)
    y3 = BatchNormalization()(y3)
    y3 = ReLU()(y3)

    # cat y12
    y12 = keras.layers.concatenate([y1, y2], axis=1)
    # convW
    y12 = BatchNormalization()(y12)
    y12 = ReLU()(y12)
    y12 = Conv2D(mid_planes, (1, 1), padding='same', use_bias=False, data_format='channels_first',
                 kernel_initializer='he_normal', name='SA_conw1')(y12)
    y12 = BatchNormalization()(y12)
    y12 = ReLU()(y12)
    # y12 = Conv2D(mid_planes, (1, 1), padding='same', use_bias=False, data_format='channels_first',
    #              kernel_initializer='he_normal', name='SA_conw2')(y12)
    y12 = Reshape((mid_planes, y3.shape[2], y3.shape[3],))(y12)

    # multi
    y = tf.multiply(y12, y3)
    y = BatchNormalization()(y)
    y = ReLU()(y)
    y = Conv2D(out_planes, (3, 3), padding='same', use_bias=False, data_format='channels_first',
               kernel_initializer='he_normal', name='SA_multi_conv')(y)
    y = add([shortcut, y])
    y = BatchNormalization()(y)
    y = ReLU()(y)
    return y


def add_common_layers(y):
    y = BatchNormalization()(y)
    y = LeakyReLU()(y)
    return y


def residual_block_decoded(y):
    shortcut = y
    y = Conv2D(8, kernel_size=(3, 3), padding='same', data_format='channels_first')(y)   # image fomat: c,h,w
    y = add_common_layers(y)

    y = Conv2D(16, kernel_size=(3, 3), padding='same', data_format='channels_first')(y)
    y = add_common_layers(y)

    y = Conv2D(2, kernel_size=(3, 3), padding='same', data_format='channels_first')(y)
    y = BatchNormalization()(y)

    y = add([shortcut, y])
    y = LeakyReLU()(y)
    return y


# Bulid the autoencoder model of CsiNet-v1
def residual_network(x):
    ip = x
    # encoder Net
    x1 = Conv2D(16, (1, 1), padding='same', use_bias=False, data_format='channels_first',
                kernel_initializer='he_normal', name='conv00')(ip)
    x1 = BatchNormalization(name='encoder_bn0')(x1)
    x1 = LeakyReLU(name='encoder_leakyrulu0')(x1)
    x1 = SA_block(x1, k=3, rel_planes=4, mid_planes=8, out_planes=16)

    x1 = Conv2D(2, (3, 3), padding='same', data_format="channels_first", name='encoder_conv4')(x1)  #
    x = tf.add(x1, ip)         # concate or add
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    # x = keras.layers.concatenate([x1, ip], axis=1)
    x = Conv2D(2, (3, 3), padding='same', data_format="channels_first", name='encoder_conv5')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Reshape((img_total,), name='encoder_reshape1')(x)

    encoded1 = Dense(encoded_dim1, activation='linear', name='encoder_cr8_dense')(x)

    # decoder
    x = Dense(img_total, activation='linear', name='decoder_cr4_dense')(encoded1)
    x = Reshape((img_channels,img_height,img_width,))(x)

    for i in range(residual_num):
        x = residual_block_decoded(x)

    x = Conv2D(2, (3, 3), activation='sigmoid', padding='same', data_format="channels_first")(x)
    return x


image_tensor = Input(shape=(img_channels, img_height, img_width))
network_output = residual_network(image_tensor)
autoencoder = Model(inputs=[image_tensor], outputs=[network_output])
autoencoder.compile(optimizer='adam', loss='mse')
print(autoencoder.summary())



outfile = './CsiNetv1_result/ck_SACsiNet_indoor25604-19_15-40.hdf5'
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
print("It cost %f sec" % ((tEnd - tStart) / x_test.shape[0]))       # calculate the time of recontribute the CSI

# Calcaulating the NMSE and rho
if envir == 'indoor':
    mat = sio.loadmat('./data/DATA_HtestFin_all.mat')
    X_test = mat['HF_all']  # array     20000*4000 complex   4000=32*125

elif envir == 'outdoor':
    mat = sio.loadmat('./data/DATA_HtestFout_all.mat')
    X_test = mat['HF_all']  # array

X_test = np.reshape(X_test, (len(X_test), img_height, 125))         # 20000*32*125
x_test_real = np.reshape(x_test[:, 0, :, :], (len(x_test), -1))         # 20000*1024
x_test_imag = np.reshape(x_test[:, 1, :, :], (len(x_test), -1))
x_test_C = x_test_real - 0.5 + 1j * (x_test_imag - 0.5)         
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
print("CsiNET-v1: NMSE is ", 10 * math.log10(np.mean(mse / power)))
print("CsiNet-v1: Correlation is ", np.mean(rho))

