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


tf.compat.v1.reset_default_graph()
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True   # 不全部占满显存, 按需分配
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

envir = 'indoor'  # 'indoor' or 'outdoor'
B = 3       # quantization bits; set B <= 0 to disable quantization
CR = 8      # compression ratio

# image params
img_height = 32
img_width = 32
img_channels = 2
img_total = img_height * img_width * img_channels  # 2048
# network params
encoded_dim1 = 256  # compress rate=1/4->dim.=512
encoded_dim2 = 128  # compress rate=1/8->dim.=256
encoded_dim3 = 64  # compress rate=1/16->dim.=128
encoded_dim4 = 32   # compress rate=1/32->dim.=64

batchsize = 200
limit1 = 50
limit2 = 100
limit3 = 150


def Num2Bit(Num, B):
    Num_ = Num.numpy()
    bit = (np.unpackbits(np.array(Num_, np.uint8), axis=1).reshape(-1, Num_.shape[1], 8)[:, :,8-B:]).reshape(-1,
                                                                                                             Num_.shape[
                                                                                                                 1] * B)
    # unpackbit:(None,32)-(None, 32*8)-reshape-(None, 32, 8)-[::4]-(None,32,4)-reshape-(None,128)
    bit.astype(np.float32)
    return tf.convert_to_tensor(bit, dtype=tf.float32)


def Bit2Num(Bit, B):
    Bit_ = Bit.numpy()
    Bit_.astype(np.float32)
    Bit_ = np.reshape(Bit_, [-1, int(Bit_.shape[1] / B), B])
    num = np.zeros(shape=np.shape(Bit_[:, :, 1]))
    for i in range(B):
        num = num + Bit_[:, :, i] * 2 ** (B - 1 - i)
    return tf.cast(num, dtype=tf.float32)


@tf.custom_gradient
def QuantizationOp(x, B):
    step = tf.cast((2 ** B), dtype=tf.float32)

    result = tf.cast((tf.round(x * step - 0.5)), dtype=tf.float32)

    result = tf.py_function(func=Num2Bit, inp=[result, B], Tout=tf.float32)

    def custom_grad(dy):
        grad = dy
        return grad, grad

    return result, custom_grad


class QuantizationLayer(tf.keras.layers.Layer):
    def __init__(self, B, **kwargs):
        self.B = B
        super(QuantizationLayer, self).__init__()

    def call(self, x):
        return QuantizationOp(x, self.B)

    def get_config(self):
        # Implement get_config to enable serialization. This is optional.
        base_config = super(QuantizationLayer, self).get_config()
        base_config['B'] = self.B
        return base_config


@tf.custom_gradient
def DequantizationOp(x, B):
    x = tf.py_function(func=Bit2Num, inp=[x, B], Tout=tf.float32)
    step = tf.cast((2 ** B), dtype=tf.float32)
    result = tf.cast((x + 0.5) / step, dtype=tf.float32)

    def custom_grad(dy):
        grad = dy
        return grad, grad

    return result, custom_grad


class DeuantizationLayer(tf.keras.layers.Layer):
    def __init__(self, B,**kwargs):
        self.B = B
        super(DeuantizationLayer, self).__init__()

    def call(self, x):
        return DequantizationOp(x, self.B)

    def get_config(self):
        base_config = super(DeuantizationLayer, self).get_config()
        base_config['B'] = self.B
        return base_config


class UnfoldLayer(tf.keras.layers.Layer):
    def __init__(self, Kh, Kw, stride=1, **kwargs):
        self.Kh = Kh
        self.Kw = Kw
        self.stride = stride
        super(UnfoldLayer, self).__init__(**kwargs)

    def call(self, x, **kwargs):
        return self.Unfold(x)
        # return DequantizationOp(x, self.B)

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
    y2 = UnfoldLayer(Kh=k, Kw=k, stride=1)(y2)
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
    y12 = Conv2D(mid_planes, (1, 1), padding='same', use_bias=False, data_format='channels_first',
                 kernel_initializer='he_normal', name='SA_conw2')(y12)
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


def residual_block_decoded(y, n):
    y = Conv2D(32, (3, 3), padding='same', data_format="channels_first", name='resi%s_conv1' % n)(y)
    y = BatchNormalization(name='resi%s_bn1' % n)(y)
    y = LeakyReLU(name='resi%s_leakyrelu1' % n)(y)

    y = Conv2D(16, (3, 3), padding='same', data_format="channels_first", name='resi%s_conv2' % n)(y)
    y = BatchNormalization(name='resi%s_bn2' % n)(y)
    y = LeakyReLU(name='resi%s_leakyrelu2' % n)(y)

    y = Conv2D(8, (3, 3), padding='same', data_format="channels_first", name='resi%s_conv3' % n)(y)
    y = BatchNormalization(name='resi%s_bn3' % n)(y)
    y = LeakyReLU(name='resi%s_leakyrelu3' % n)(y)
    return y


# Bulid the autoencoder model of CsiNet
def residual_network(x):
    ip = x
    # encoder Net
    x1 = Conv2D(16, (1, 1), padding='same', use_bias=False, data_format='channels_first',
                kernel_initializer='he_normal', name='conv00')(ip)
    x1 = BatchNormalization(name='encoder_bn0')(x1)
    x1 = LeakyReLU(name='encoder_leakyrulu0')(x1)
    x1 = Conv2D(16, (1, 1), padding='same', use_bias=False, data_format='channels_first',
                kernel_initializer='he_normal', name='conv01')(x1)  #
    x1 = SA_block(x1, k=3, rel_planes=4, mid_planes=8, out_planes=16)

    x1 = Conv2D(32, (3, 3), padding='same', data_format="channels_first", name='encoder_conv1')(x1)
    x1 = BatchNormalization()(x1)
    x1 = LeakyReLU()(x1)

    x1 = Conv2D(16, (3, 3), padding='same', data_format="channels_first", name='encoder_conv2')(x1)
    x1 = BatchNormalization()(x1)
    x1 = LeakyReLU()(x1)

    x1 = Conv2D(8, (3, 3), padding='same', data_format="channels_first", name='encoder_conv3')(x1)
    x1 = BatchNormalization()(x1)
    x1 = LeakyReLU()(x1)

    x1 = Conv2D(2, (3, 3), padding='same', data_format="channels_first", name='encoder_conv4')(x1)  #
    # x = tf.add(x1, ip)         # concate or add

    x = keras.layers.concatenate([x1, ip], axis=1)
    x = Conv2D(2, (1, 1), padding='same', data_format="channels_first", name='encoder_conv5')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Reshape((img_total,), name='encoder_reshape1')(x)

    encoded1 = Dense(encoded_dim1, activation='linear', name='encoder_cr8_dense')(x)
    # sigmoid: Limit the output to [0, 1] for quantization
    encoded1 = Activation('sigmoid')(encoded1)
    encoded2 = Dense(encoded_dim2, activation='linear', name='encoder_cr16_dense')(encoded1)
    encoded2 = Activation('sigmoid')(encoded2)
    encoded3 = Dense(encoded_dim3, activation='linear', name='encoder_cr32_dense')(encoded2)
    encoded3 = Activation('sigmoid')(encoded3)
    encoded4 = Dense(encoded_dim4, activation='linear', name='encoder_cr64_dense')(encoded3)
    encoded4 = Activation('sigmoid')(encoded4)
    if B > 0:
        encoded1 = QuantizationLayer(B)(encoded1)
        encoded2 = QuantizationLayer(B)(encoded2)
        encoded3 = QuantizationLayer(B)(encoded3)
        encoded4 = QuantizationLayer(B)(encoded4)

    # decoder
    if B > 0:
        decoder1 = DeuantizationLayer(B)(encoded1)
        decoder1 = Reshape((encoded_dim1,))(decoder1)

        decoder2 = DeuantizationLayer(B)(encoded2)
        decoder2 = Reshape((encoded_dim2,))(decoder2)

        decoder3 = DeuantizationLayer(B)(encoded3)
        decoder3 = Reshape((encoded_dim3,))(decoder3)

        decoder4 = DeuantizationLayer(B)(encoded4)
        decoder4 = Reshape((encoded_dim4,))(decoder4)
    else:
        decoder1 = encoded1
        decoder2 = encoded2
        decoder3 = encoded3
        decoder4 = encoded4

    if CR == 8:
        x = Dense(img_total, activation='linear', name='decoder_cr8_dense')(decoder1)
    if CR == 16:
        x = Dense(img_total, activation='linear', name='decoder_cr16_dense')(decoder2)
    if CR == 32:
        x = Dense(img_total, activation='linear', name='decoder_cr32_dense')(decoder3)
    if CR == 64:
        x = Dense(img_total, activation='linear', name='decoder_cr64_dense')(decoder4)

    x = Reshape((2, 32, 32,), name='decoder_reshape2')(x)
    ip = x
    x1 = ip
    x = residual_block_decoded(ip, 0)
    x2 = x
    x = keras.layers.concatenate([x1, x2], axis=1, name='resi_concate1')

    x = residual_block_decoded(x, 1)
    x3 = x
    x = keras.layers.concatenate([x1, x2, x3], axis=1, name='resi_concate2')

    x = residual_block_decoded(x, 2)
    x4 = x
    x = keras.layers.concatenate([x1, x2, x3, x4], axis=1, name='resi_concate3')

    x = residual_block_decoded(x, 3)
    x5 = x
    x = keras.layers.concatenate([x1, x2, x3, x4, x5], axis=1, name='resi_concate4')

    x = residual_block_decoded(x, 4)
    x6 = x
    x = keras.layers.concatenate([x1, x2, x3, x4, x5, x6], axis=1, name='resi_concate5')

    x = Conv2D(2, (3, 3), activation='sigmoid', padding='same', data_format="channels_first", name='decoder_convo')(x)
    return x


image_tensor = Input(shape=(img_channels, img_height, img_width))
network_output = residual_network(image_tensor)
autoencoder = Model(inputs=[image_tensor], outputs=[network_output])
adam = keras.optimizers.Adam(lr=0.0001)
autoencoder.compile(optimizer=adam, loss='mse')
print(autoencoder.summary())


outfile = '../SALDR_result/SALDR_U_indoor.h5'         #
autoencoder.load_weights(outfile, by_name=True)

# Data loading
if envir == 'indoor':
    mat = sio.loadmat('../data/DATA_Htrainin.mat')
    x_train = mat['HT']  # array     100000*2048
    mat = sio.loadmat('../data/DATA_Hvalin.mat')
    x_val = mat['HT']  # array         30000*2048
    mat = sio.loadmat('../data/DATA_Htestin.mat')
    x_test = mat['HT']  # array     20000*2048

elif envir == 'outdoor':
    mat = sio.loadmat('../data/DATA_Htrainout.mat')
    x_train = mat['HT']  # array
    mat = sio.loadmat('../data/DATA_Hvalout.mat')
    x_val = mat['HT']  # array
    mat = sio.loadmat('../data/DATA_Htestout.mat')
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

x_test_real = np.reshape(x_test[:, 0, :, :], (len(x_test), -1))  # 20000*1024
x_test_imag = np.reshape(x_test[:, 1, :, :], (len(x_test), -1))
x_test_C = x_test_real - 0.5 + 1j * (x_test_imag - 0.5)  # recover complex,  why subtract 0.5

x_hat_real = np.reshape(x_hat[:, 0, :, :], (len(x_hat), -1))
x_hat_imag = np.reshape(x_hat[:, 1, :, :], (len(x_hat), -1))
x_hat_C = x_hat_real - 0.5 + 1j * (x_hat_imag - 0.5)
# ###############################################################################################
power = np.sum(abs(x_test_C) ** 2, axis=1)
mse = np.sum(abs(x_test_C - x_hat_C) ** 2, axis=1)

print("In " + envir + " environment")
print("NMSE of CR:%s is" % CR, 10 * math.log10(np.mean(mse / power)))
