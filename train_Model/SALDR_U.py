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


tf.compat.v1.reset_default_graph()
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True   # 不全部占满显存, 按需分配
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

envir = 'indoor'  # 'indoor' or 'outdoor'
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
                kernel_initializer='he_normal', name='conv01')(x1)       #
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

    # decoder
    x1 = Dense(img_total, activation='linear', name='decoder_cr8_dense')(encoded1)
    x2 = Dense(img_total, activation='linear', name='decoder_cr16_dense')(encoded2)
    x3 = Dense(img_total, activation='linear', name='decoder_cr32_dense')(encoded3)
    x4 = Dense(img_total, activation='linear', name='decoder_cr64_dense')(encoded4)

    x1 = Lambda(lambda e: e[0:limit1, :], name='lambda_cr8_split')(x1)
    x2 = Lambda(lambda e: e[limit1:limit2, :], name='lambda_cr16_split')(x2)
    x3 = Lambda(lambda e: e[limit2:limit3, :], name='lambda_cr32_split')(x3)
    x4 = Lambda(lambda e: e[limit3:batchsize, :], name='lambda_cr64_split')(x4)
    x = keras.layers.concatenate([x1, x2, x3, x4], axis=0, name='decoder_concate1')

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


def SM_loss(y_actual, y_pred):    #
    y_pred1 = y_pred[0:limit1, :, :, :]
    y_pred2 = y_pred[limit1:limit2, :, :, :]
    y_pred3 = y_pred[limit2:limit3, :, :, :]
    y_pred4 = y_pred[limit3:batchsize, :, :, :]

    y_actual1 = y_actual[0:limit1, :, :, :]
    y_actual2 = y_actual[limit1:limit2, :, :, :]
    y_actual3 = y_actual[limit2:limit3, :, :, :]
    y_actual4 = y_actual[limit3:batchsize, :, :, :]

    loss_c8 = tf.reduce_mean(tf.square(y_actual1-y_pred1))
    loss_c16 = tf.reduce_mean(tf.square(y_actual2 - y_pred2))
    loss_c32 = tf.reduce_mean(tf.square(y_actual3 - y_pred3))
    loss_c64 = tf.reduce_mean(tf.square(y_actual4 - y_pred4))
    sm_loss = 3*loss_c64 + 4*loss_c32 + 7*loss_c16 + 25*loss_c8
    sm_loss = sm_loss/39
    return sm_loss


image_tensor = Input(shape=(img_channels, img_height, img_width))
network_output = residual_network(image_tensor)
autoencoder = Model(inputs=[image_tensor], outputs=[network_output])
adam = keras.optimizers.Adam(lr=0.001)
autoencoder.compile(optimizer=adam, loss=SM_loss)
print(autoencoder.summary())
#

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


reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.6, patience=10, mode='auto',
                                              verbose=0, min_delta=3e-6, cooldown=0, min_lr=0)
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=80, mode='auto',
                                               verbose=0, min_delta=3e-6)
# #
ckfile = 'SA19S_' + envir + time.strftime('%m-%d_%H-%M')
filepath = "../SALDR_result/ck_%s.h5" % ckfile
ck = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto',
                                     save_weights_only=True)

autoencoder.fit(x_train, x_train,
                epochs=1000,
                batch_size=batchsize,
                shuffle=True,
                validation_data=(x_val, x_val),
                callbacks=[ck, early_stopping])

# Testing data
tStart = time.time()
x_hat = autoencoder.predict(x_test)
tEnd = time.time()
print("It cost %f sec" % ((tEnd - tStart) / x_test.shape[0]))  # calculate the time of recontribute the CSI for
# every channel matrix

# Calcaulating the NMSE and rho
if envir == 'indoor':
    mat = sio.loadmat('../data/DATA_HtestFin_all.mat')
    X_test = mat['HF_all']  # array     20000*4000 complex   4000=32*125   non_truncated data?

elif envir == 'outdoor':
    mat = sio.loadmat('../data/DATA_HtestFout_all.mat')
    X_test = mat['HF_all']  # array


batch_num = int(len(x_test)/batchsize)
X_test = np.reshape(X_test, (len(X_test), img_height, 125))  # 20000*32*125


# ###############################################################################################
x_test_cr4 = x_test[0:limit1, :, :]
for i in range(1, batch_num):
    start = int(i*200)
    x_test_i = x_test[start:(start+limit1), :, :]
    x_test_cr4 = keras.layers.concatenate([x_test_cr4, x_test_i], axis=0)

x_test_cr4_real = np.reshape(x_test_cr4[:, 0, :, :], (len(x_test_cr4), -1))  # 20000*1024
x_test_cr4_imag = np.reshape(x_test_cr4[:, 1, :, :], (len(x_test_cr4), -1))
x_test_C = x_test_cr4_real - 0.5 + 1j * (x_test_cr4_imag - 0.5)  # recover complex,  why subtract 0.5


x_hat_cr4 = x_hat[0:limit1, :, :, :]
for i in range(1, batch_num):
    start = int(i*200)
    x_hat_i = x_hat[start:(start+limit1), :, :, :]
    x_hat_cr4 = keras.layers.concatenate([x_hat_cr4, x_hat_i], axis=0)

x_hat_cr4_real = np.reshape(x_hat_cr4[:, 0, :, :], (len(x_hat_cr4), -1))       # 修改：按照batch切片方式提取各个CR的预测结果
x_hat_cr4_imag = np.reshape(x_hat_cr4[:, 1, :, :], (len(x_hat_cr4), -1))
# print('x_hat_cr4.shape:', x_hat_cr4.shape)
x_hat_C = x_hat_cr4_real - 0.5 + 1j * (x_hat_cr4_imag - 0.5)
# print('x_hat_C.shape:', x_hat_C.shape)      # 5000, 1024
x_hat_F = np.reshape(x_hat_C, (len(x_hat_C), img_height, img_width))  # 5000*32*32
# zero fill subcarrier and do FFT on subcarrier axis
X_hat = np.fft.fft(np.concatenate((x_hat_F, np.zeros((len(x_hat_C), img_height, 257 - img_width))), axis=2), axis=2)
X_hat = X_hat[:, :, 0:125]  # 5000*32*125


X_test_cr4 = X_test[0:limit1, :, :]
for i in range(1, batch_num):
    start = int(i*200)
    X_test_i = X_test[start:(start+limit1), :, :]
    X_test_cr4 = keras.layers.concatenate([X_test_cr4, X_test_i], axis=0)

# caculate the cosine similarity
n1 = np.sqrt(np.sum(np.conj(X_test_cr4) * X_test_cr4, axis=1))
n1 = n1.astype('float64')
n2 = np.sqrt(np.sum(np.conj(X_hat) * X_hat, axis=1))
n2 = n2.astype('float64')
aa = abs(np.sum(np.conj(X_test_cr4) * X_hat, axis=1))
rho = np.mean(aa / (n1 * n2), axis=1)

X_hat = np.reshape(X_hat, (len(X_hat), -1))
X_test_cr4 = np.reshape(X_test_cr4, (len(X_test_cr4), -1))
power = np.sum(abs(x_test_C) ** 2, axis=1)
power_d = np.sum(abs(X_hat) ** 2, axis=1)
mse = np.sum(abs(x_test_C - x_hat_C) ** 2, axis=1)

print("In " + envir + " environment")
print("SM: NMSE of CR:8 is", 10 * math.log10(np.mean(mse / power)))
print("SM: Correlation of CR:8 is ", np.mean(rho))

