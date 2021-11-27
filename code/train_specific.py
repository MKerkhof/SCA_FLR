import math
import os
import random
import time
import sys
from datetime import datetime

import h5py
import numpy as np
import tensorflow as tf
from sklearn import preprocessing
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.keras.layers import Conv1D, AveragePooling1D, Dense
from tensorflow.python.keras.utils.layer_utils import count_params

from tensorflow.python.keras.utils.np_utils import to_categorical

from rankingloss import rankingloss
from rankingloss import dummyloss
from focal_loss import *

from cer import cer_loss
from flr import flr_loss


# tf.debugging.set_log_device_placement(True)

dataset = sys.argv[1]
cluster = sys.argv[2] == 'true'
arch = sys.argv[3]
leakage = sys.argv[4]
loss = sys.argv[5]

if len(sys.argv) > 6:
    amount = int(sys.argv[6])
else:
    amount = 1

tag = ""
from_files = False



print(f'Generating {arch} model: dataset={dataset} leakage={leakage} loss={loss}')

if cluster:
    # Root folder on TU Delft computing cluster
    cluster_root = "/tudelft.net/staff-bulk/ewi/insy/CYS/spicek/mkerkhof/"

    files = {"aes_hd": cluster_root + "thesis/data/aes_hd.h5",
             "aes_hd_ext": cluster_root + "thesis/data/aes_hd_ext.h5",
             "ches_ctf": cluster_root + "thesis/data/ches_ctf.h5",
             "ascad": cluster_root + "thesis/data/ASCAD.h5",
             "ascad_desync50": cluster_root + "thesis/data/ASCAD_desync50.h5",
             "ascad_variable": cluster_root + "thesis/data/ascad-variable.h5"}
else:
    files = {"aes_hd": "../data/aes_hd.h5",
             "aes_hd_ext": "../data/aes_hd_ext.h5",
             "ches_ctf": "../data/ches_ctf.h5",
             "ascad": "../data/ASCAD.h5",
             "ascad_desync50": "../data/ASCAD_desync50.h5",
             "ascad_variable": "../data/ascad-variable.h5"}

sbox = (
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16)


with h5py.File(files[dataset], "r") as f:
    attack = f['Attack_traces']
    profiling = f['Profiling_traces']

    attack_traces = attack['traces'][()]
    attack_metadata = attack['metadata'][()]

    profiling_traces = profiling['traces'][()]
    profiling_metadata = profiling['metadata'][()]

profiling_plaintext = [i[0] for i in profiling_metadata]
attack_plaintext = [i[0] for i in attack_metadata]


if dataset == "ascad_variable":
    known_keys = [list(trace[1]) for trace in profiling_metadata]
    attack_key = attack_metadata[0][1]

else:
    known_keys = [list(trace[2]) for trace in profiling_metadata]
    attack_key = attack_metadata[0][2]

keybyte = 2

print(f'Attacking keybyte {keybyte} of {dataset}. Known key: {known_keys[0]}')


sbox_output = np.array(
    [sbox[int(profiling_plaintext[i][keybyte]) ^ int(known_keys[i][keybyte])] for i in range(len(profiling_plaintext))])


scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
profiling_traces = scaler.fit_transform(profiling_traces)


if leakage == 'id':
    labels = sbox_output
    num_classes = 256

elif leakage == 'hamming':
    hamming = [bin(n).count("1") for n in range(256)]
    outputSboxHW = [hamming[s] for s in sbox_output]
    labels = np.array(outputSboxHW)
    num_classes = 9



if dataset == "ascad_variable":

    profiling_traces = np.array(profiling_traces)
    labels = np.array(labels)

    profiling_traces = profiling_traces[0:50000]
    labels = labels[0:50000]


# Calculate alpha for focal loss
_, class_counts = np.unique(labels, return_counts=True)


# alpha = (0.06*(alpha - np.min(alpha))/np.ptp(alpha))
# alpha[alpha < 0.01] = 0.01
gamma = 2
alpha = np.full(num_classes, 0.005)

# Class Balanced weights
# beta = 0.999
# alpha = [(1 - beta) / (1 - math.pow(beta, x)) for x in class_counts]
# gamma = 0.5


#
# # print(class_counts)
# print(alpha)
# print(beta)
# print(gamma)



def create_mlp(nr_layers, nr_nodes, activation, optimiser, learning_rate, loss, a=0.25, g=2.0):
    start = tf.keras.layers.Input(shape=(np.shape(profiling_traces)[1],))

    x = tf.keras.layers.Dense(nr_nodes, input_dim=np.shape(profiling_traces)[1], activation=activation)(start)

    for i in range(nr_layers - 1):
        x = tf.keras.layers.Dense(nr_nodes, activation=activation)(x)

    score_layer = tf.keras.layers.Dense(num_classes, activation=None, name='scores')(x)

    predictions = tf.keras.layers.Activation('softmax', name='preds')(score_layer)

    if loss == 'rkl':
        model = tf.keras.models.Model(start, outputs=[predictions, score_layer])
        model.compile(loss={'scores': rankingloss(alpha_value=0.5, nb_class=num_classes), 'preds': dummyloss},
                      optimizer=optimiser(learning_rate=learning_rate), metrics=['accuracy'])
    elif loss == 'cer_loss':
        model = tf.keras.models.Model(start, predictions)
        model.compile(loss=cer_loss(10),
                      optimizer=optimiser(learning_rate=learning_rate), metrics=['accuracy'])
    elif loss == 'cos_cross_loss':
        model = tf.keras.models.Model(start, predictions)
        model.compile(loss=cosine_crossentropy(),
                      optimizer=optimiser(learning_rate=learning_rate), metrics=['accuracy'])
    elif loss == 'focal_loss':
        model = tf.keras.models.Model(start, predictions)
        model.compile(loss=categorical_focal_loss(alpha=a, gamma=g),
                      optimizer=optimiser(learning_rate=learning_rate), metrics=['accuracy'])
    elif loss == 'cer_focal_loss':
        model = tf.keras.models.Model(start, predictions)
        model.compile(loss=cer_focal_loss(alpha=a, gamma=g),
                      optimizer=optimiser(learning_rate=learning_rate), metrics=['accuracy'])
    elif loss == 'cosine_focal_loss':
        model = tf.keras.models.Model(start, predictions)
        model.compile(loss=cosine_focal_loss(alpha=a, gamma=g),
                      optimizer=optimiser(learning_rate=learning_rate), metrics=['accuracy'])
    elif loss == 'corr_cross_hinge_loss':
        model = tf.keras.models.Model(start, predictions)
        model.compile(loss=corr_cros_hinge_loss,
                      optimizer=optimiser(learning_rate=learning_rate), metrics=['accuracy'])
    elif loss == 'correntropy':
        model = tf.keras.models.Model(start, predictions)
        model.compile(loss=correntropy,
                      optimizer=optimiser(learning_rate=learning_rate), metrics=['accuracy'])
    elif loss == 'flr_loss':
        model = tf.keras.models.Model(start, predictions)
        model.compile(loss=flr_loss(3, num_classes, alpha=a, gamma=g),
                      optimizer=optimiser(learning_rate=learning_rate), metrics=['accuracy'])
    else:
        model = tf.keras.models.Model(start, predictions)
        model.compile(loss=loss,
                      optimizer=optimiser(learning_rate=learning_rate), metrics=['accuracy'])
    return model


def create_cnn(cov_layers, cov_filters, cov_kernel, pool_size, pool_stride, pool_type, dense_layers,
               dense_neurons, activation, optimiser, learning_rate, loss, a=0.25, g=2.0):

    start = tf.keras.layers.Input(shape=(np.shape(profiling_traces)[1], 1))

    # Convolution layer
    x = tf.keras.layers.Conv1D(kernel_initializer='he_uniform', filters=cov_filters, kernel_size=cov_kernel,
                               activation=activation, padding='same')(
        start)
    x = tf.keras.layers.BatchNormalization()(x)

    if pool_type == 'avg_pooling':
        x = tf.keras.layers.AveragePooling1D(pool_size=pool_size, strides=pool_stride, padding='same')(x)
    elif pool_type == 'max_pooling':
        x = tf.keras.layers.MaxPool1D(pool_size=pool_size, strides=pool_stride, padding='same')(x)

    for i in range(cov_layers - 1):
        x = tf.keras.layers.Conv1D(filters=cov_filters, kernel_size=cov_kernel, activation=activation, padding='same')(
            x)
        x = tf.keras.layers.BatchNormalization()(x)

        if pool_type == 'avg_pooling':
            x = tf.keras.layers.AveragePooling1D(pool_size=pool_size, strides=pool_stride, padding='same')(x)
        elif pool_type == 'max_pooling':
            x = tf.keras.layers.MaxPool1D(pool_size=pool_size, strides=pool_stride, padding='same')(x)

    # Flatten for dense layers
    x = tf.keras.layers.Flatten()(x)

    # Dense layers
    for i in range(dense_layers):
        x = tf.keras.layers.Dense(dense_neurons, activation=activation)(x)

    # Classification layer
    score_layer = tf.keras.layers.Dense(num_classes, activation=None, name='scores')(x)
    predictions = tf.keras.layers.Activation(activation='softmax', name='preds')(score_layer)


    if loss == 'rkl':
        model = tf.keras.models.Model(start, outputs=[predictions, score_layer])
        model.compile(loss={'scores': rankingloss(alpha_value=0.5, nb_class=num_classes), 'preds': dummyloss},
                      optimizer=optimiser(), metrics=['accuracy'])
    elif loss == 'focal_loss':
        model = tf.keras.models.Model(start, predictions)
        model.compile(loss=categorical_focal_loss(alpha=[np.full(num_classes, fill_value=a)], gamma=g),
                      optimizer=optimiser(learning_rate=learning_rate), metrics=['accuracy'])
    elif loss == 'cer_loss':
        model = tf.keras.models.Model(start, predictions)
        model.compile(loss=cer_loss(10),
                      optimizer=optimiser(learning_rate=learning_rate), metrics=['accuracy'])
    elif loss == 'flr_loss':
        model = tf.keras.models.Model(start, predictions)
        model.compile(loss=flr_loss(3, num_classes, alpha=a, gamma=g),
                      optimizer=optimiser(learning_rate=learning_rate), metrics=['accuracy'])
    elif loss == 'cos_cross_loss':
        model = tf.keras.models.Model(start, predictions)
        model.compile(loss=cosine_crossentropy(),
                      optimizer=optimiser(learning_rate=learning_rate), metrics=['accuracy'])
    elif loss == 'cer_focal_loss':
        model = tf.keras.models.Model(start, predictions)
        model.compile(loss=cer_focal_loss(alpha=[np.full(num_classes, fill_value=0.25)], gamma=2),
                      optimizer=optimiser(learning_rate=learning_rate), metrics=['accuracy'])
    elif loss == 'cosine_focal_loss':
        model = tf.keras.models.Model(start, predictions)
        model.compile(loss=cosine_focal_loss(alpha=[np.full(num_classes, fill_value=0.25)], gamma=2),
                      optimizer=optimiser(learning_rate=learning_rate), metrics=['accuracy'])
    elif loss == 'corr_cross_hinge_loss':
        model = tf.keras.models.Model(start, predictions)
        model.compile(loss=corr_cros_hinge_loss,
                      optimizer=optimiser(learning_rate=learning_rate), metrics=['accuracy'])
    elif loss == 'correntropy':
        model = tf.keras.models.Model(start, predictions)
        model.compile(loss=correntropy,
                      optimizer=optimiser(learning_rate=learning_rate), metrics=['accuracy'])

    else:
        model = tf.keras.models.Model(start, predictions)
        model.compile(loss=loss,
                      optimizer=optimiser(), metrics=['accuracy'])
    return model

# create the CNN_best architecture from ascad paper
def create_cnn_best(cov_layers, cov_filters, cov_kernel, pool_size, pool_stride, pool_type, dense_layers,
               dense_neurons, activation, optimiser, learning_rate, loss):

    print(f'Creating CNN_best with loss {loss}')

    start = tf.keras.layers.Input(shape=(np.shape(profiling_traces)[1], 1))

    x = tf.keras.layers.Conv1D(64, 11, activation='relu', padding='same', name='block1_conv1')(start)
    x = tf.keras.layers.AveragePooling1D(pool_size=2, strides=2, name='block1_pool')(x)
    # Block 2
    x = Conv1D(128, 11, strides=1, activation='relu', padding='same', name='block2_conv1')(x)
    x = AveragePooling1D(pool_size=2, strides=2, name='block2_pool')(x)
    # Block 3
    x = Conv1D(256, 11, strides=1, activation='relu', padding='same', name='block3_conv1')(x)
    x = AveragePooling1D(pool_size=2, strides=2, name='block3_pool')(x)
    # Block 4
    x = Conv1D(512, 11, strides=1, activation='relu', padding='same', name='block4_conv1')(x)
    x = AveragePooling1D(pool_size=2, strides=2, name='block4_pool')(x)
    # Block 5
    x = Conv1D(512, 11, strides=1, activation='relu', padding='same', name='block5_conv1')(x)
    x = AveragePooling1D(pool_size=2, strides=2, name='block5_pool')(x)

    x = tf.keras.layers.Flatten()(x)

    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    score_layer = Dense(num_classes, activation=None, name='scores')(x)
    predictions = tf.keras.layers.Activation(activation='softmax', name='preds')(score_layer)

    if loss == 'rkl':
        model = tf.keras.models.Model(start, outputs=[predictions, score_layer])
        model.compile(loss={'scores': rankingloss(alpha_value=0.5, nb_class=num_classes), 'preds': dummyloss},
                      optimizer=optimiser(), metrics=['accuracy'])
    elif loss == 'focal_loss':
        model = tf.keras.models.Model(start, predictions)
        model.compile(loss=categorical_focal_loss(alpha=[np.full(num_classes, fill_value=0.25)], gamma=2),
                      optimizer=optimiser(learning_rate=learning_rate), metrics=['accuracy'])
    elif loss == 'cer_loss':
        model = tf.keras.models.Model(start, predictions)
        model.compile(loss=cer_loss(10),
                      optimizer=optimiser(learning_rate=learning_rate), metrics=['accuracy'])
    elif loss == 'flr_loss':
        model = tf.keras.models.Model(start, predictions)
        model.compile(loss=flr_loss(10, num_classes),
                      optimizer=optimiser(learning_rate=learning_rate), metrics=['accuracy'])
    else:
        model = tf.keras.models.Model(start, predictions)
        model.compile(loss=loss,
                      optimizer=optimiser(), metrics=['accuracy'])
    return model

# create the methodology paper for ascad
def create_cnn_methodology(cov_layers, cov_filters, cov_kernel, pool_size, pool_stride, pool_type, dense_layers,
               dense_neurons, activation, optimiser, learning_rate, loss):

    print(f'Creating CNN_methodology with loss {loss}')

    start = tf.keras.layers.Input(shape=(np.shape(profiling_traces)[1], 1))

    x = tf.keras.layers.Conv1D(4, 1, kernel_initializer='he_uniform', activation='selu', padding='same', name='block1_conv1')(start)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.AveragePooling1D(pool_size=2, strides=2, name='block1_pool')(x)

    x = tf.keras.layers.Flatten()(x)

    x = Dense(10, activation='selu', name='fc1')(x)
    x = Dense(10, activation='selu', name='fc2')(x)

    score_layer = Dense(num_classes, activation=None, name='scores')(x)
    predictions = tf.keras.layers.Activation(activation='softmax', name='preds')(score_layer)

    if loss == 'rkl':
        model = tf.keras.models.Model(start, outputs=[predictions, score_layer])
        model.compile(loss={'scores': rankingloss(alpha_value=0.5, nb_class=num_classes), 'preds': dummyloss},
                      optimizer=optimiser(), metrics=['accuracy'])
    elif loss == 'focal_loss':
        model = tf.keras.models.Model(start, predictions)
        model.compile(loss=categorical_focal_loss(alpha=[np.full(num_classes, fill_value=0.25)], gamma=2),
                      optimizer=optimiser(learning_rate=learning_rate), metrics=['accuracy'])
    elif loss == 'cer_loss':
        model = tf.keras.models.Model(start, predictions)
        model.compile(loss=cer_loss(10),
                      optimizer=optimiser(learning_rate=learning_rate), metrics=['accuracy'])
    else:
        model = tf.keras.models.Model(start, predictions)
        model.compile(loss=loss,
                      optimizer=optimiser(), metrics=['accuracy'])
    return model


def fit_models(x, y, models, store=True):
    for i in range(len(models)):
        model = models[i]
        model_data = {}

        print(f'Training model {model["hyperparams"]}, alpha {alpha}, gamma {gamma}')
        start = datetime.now()

        model["model"].fit(x, y, epochs=model["hyperparams"][-2], batch_size=model["hyperparams"][-1])

        end = datetime.now()
        training_time = (end - start).total_seconds()
        print(f'Finished training {model["hyperparams"]}, training time: {end - start}')

        model_data['hyperparams'] = model['hyperparams']
        model_data['training_time'] = training_time
        # model_data['alpha'] = model['alpha']
        # model_data['gamma'] = model['gamma']
        model_data['alpha'] = alpha
        model_data['gamma'] = gamma

        if store:
            # Fix for some bug with the cluster
            time.sleep(1)

            model['model_id'] = datetime.now().strftime('%S_%M_%H_%d_%m')

            if cluster:
                model['model'].save(
                    cluster_root + 'thesis/models/' + arch + '_' + dataset + '-' + leakage + '_' + lf + '_' + ident + str(
                        model[
                            'model_id']) + '.h5')
                np.save(
                    cluster_root + 'thesis/models/' + arch + '_' + dataset + '-' + leakage + '_' + lf + '_' + ident + str(
                        model[
                            'model_id']) + '.npy',
                    model_data)
            else:
                model['model'].save(
                    '../models/' + arch + '_' + dataset + '-' + leakage + '_' + lf + '_' + ident + str(
                        model[
                            'model_id']) + '.h5')
                np.save(
                    '../models/' + arch + '_' + dataset + '-' + leakage + '_' + lf + '_' + ident + str(
                        model[
                            'model_id']) + '.npy',
                    model_data)

    print('Finished training all models')


with tf.device('/GPU:0'):
    models = []

    if from_files:
        if cluster:
            path = cluster_root + 'thesis/configs/'
        else:
            path = '../configs'


        for file in os.listdir(path):
            if not file.startswith('.'):
                result = np.load(path + file, allow_pickle=True).item()
                hyperparams = list(result['config'])

                if 'balanced' in file:
                    t = '_balanced_'
                elif 'optimised' in file:
                    t = '_optimised_'
                elif 'default' in file:
                    t = '_default_'
                else:
                    t = ''

                if 'alpha' in result:
                    alpha = result['alpha']
                    gamma = result['gamma']

                    print(f'Loaded alpha and gamma from {file}')
                else:

                    # alpha = np.full(num_classes, 0.25)
                    # beta = 0.999
                    # alpha = [(1 - beta) / (1 - math.pow(beta, x)) for x in class_counts]
                    # gamma = 2
                    print(f'Using class balancing')

                for i in range(amount):
                    if arch == 'CNN':
                        model = {'model': create_cnn(*hyperparams[:12], alpha, gamma), 'hyperparams': hyperparams,
                                 'alpha': alpha, 'gamma': gamma}
                        print(
                            f'Creating {amount} model(s) from loaded config: {hyperparams}, alpha: {alpha}, gamma: {gamma}')
                        model['trainable'] = count_params(model['model'].trainable_weights)
                        model['tag'] = t
                        models.append(model)
                    if arch == 'MLP':
                        model = {'model': create_mlp(*hyperparams[:6], alpha, gamma), 'hyperparams': hyperparams,
                                 'alpha': alpha, 'gamma': gamma}
                        print(
                            f'Creating {amount} model(s) from loaded config: {hyperparams}, alpha: {alpha}, gamma: {gamma}')
                        model['trainable'] = count_params(model['model'].trainable_weights)
                        model['tag'] = t
                        models.append(model)

                # for i in range(amount):
                #     if arch == 'CNN':
                #         model = {'model': create_cnn(*hyperparams[:12]), 'hyperparams': hyperparams
                #                  }
                #         print(
                #             f'Creating {amount} model(s) from loaded config: {hyperparams}')
                #         model['trainable'] = count_params(model['model'].trainable_weights)
                #         # model['tag'] = t
                #         models.append(model)
                #     if arch == 'MLP':
                #         model = {'model': create_mlp(*hyperparams[:6]), 'hyperparams': hyperparams}
                #         print(
                #             f'Creating {amount} model(s) from loaded config: {hyperparams}')
                #         model['trainable'] = count_params(model['model'].trainable_weights)
                #         # model['tag'] = t
                #         models.append(model)

        print(f"Created {len(models)} models from config files, starting training..")
        fit_models(profiling_traces, to_categorical(labels, num_classes=num_classes), models)
        exit()

    # Define a specific CNN architecture to train
    if arch == 'CNN':
        cov_layers = 1
        cov_filters = 4
        cov_kernel = 1
        pool_size = 2
        pool_stride = 2
        pool_type = 'avg_pooling'
        dense_layers = 2
        dense_neurons = 10
        activation = 'selu'
        optimiser = tf.keras.optimizers.Adam
        # optimiser = tf.keras.optimizers.RMSprop
        learning_rate = 0.00005

        epochs = 50
        batchsize = 50

        if loss == 'all':
            for lf in ['categorical_crossentropy', 'mse', 'log_cosh',
                       'mean_squared_logarithmic_error', "categorical_hinge", 'rkl', 'cer_loss', 'focal_loss']:
                hyperparams = [cov_layers, cov_filters, cov_kernel, pool_size, pool_stride, pool_type, dense_layers,
                               dense_neurons, activation, optimiser, learning_rate, lf, epochs, batchsize]

                model = {'model': create_cnn(*hyperparams[:12]), 'hyperparams': hyperparams}
                model['trainable'] = count_params(model['model'].trainable_weights)

                models.append(model)

        else:
            hyperparams = [cov_layers, cov_filters, cov_kernel, pool_size, pool_stride, pool_type, dense_layers,
                           dense_neurons, activation, optimiser, learning_rate, loss, epochs, batchsize]

            for i in range(amount):
                model = {'model': create_cnn(*hyperparams[:12]), 'hyperparams': hyperparams}
                model['trainable'] = count_params(model['model'].trainable_weights)

                models.append(model)

        fit_models(profiling_traces, to_categorical(labels, num_classes=num_classes), models)

    elif arch == 'MLP':

        # Define a specific CNN architecture to train
        layers = 6
        nodes = 200
        activation = 'relu'
        optimiser = tf.keras.optimizers.RMSprop
        learning_rate = 0.00001

        epochs = 10
        batchsize = 100

        if loss == 'all':
            for lf in ['categorical_crossentropy', 'mse', 'log_cosh',
                       'mean_squared_logarithmic_error', "categorical_hinge", 'rkl', 'cer_loss', 'focal_loss']:
                hyperparams = [layers, nodes, activation, optimiser, learning_rate, lf, epochs, batchsize]

                model = {'model': create_mlp(*hyperparams[:6]), 'hyperparams': hyperparams}
                model['trainable'] = count_params(model['model'].trainable_weights)

                models.append(model)

        else:
            hyperparams = [layers, nodes, activation, optimiser, learning_rate, loss, epochs, batchsize]

            for i in range(amount):
                model = {'model': create_mlp(*hyperparams[:6]), 'hyperparams': hyperparams}
                model['trainable'] = count_params(model['model'].trainable_weights)

                models.append(model)

        fit_models(profiling_traces, to_categorical(labels, num_classes=num_classes), models)