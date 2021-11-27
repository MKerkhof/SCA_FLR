import argparse
import random
from datetime import datetime

import h5py
from sklearn import preprocessing
from tensorflow.python.keras.utils.layer_utils import count_params
from tensorflow.python.keras.utils.np_utils import to_categorical

# tf.debugging.set_log_device_placement(True)
from cer import cer_loss
from flr import flr_loss
from focal_loss import *
from myloss import *
from rankingloss import *

argparser = argparse.ArgumentParser(description="Script to train CNN/MLP models with random architectures on various "
                                                "SCA datasets. Datasets should be in a 'data' folder in the project "
                                                "root, in .h5 format similar to datasets available at "
                                                "http://aisylabdatasets.ewi.tudelft.nl/. Results are stored in .h5 "
                                                "format, additional model "
                                                "info is stored in a .npy file with the same name in the 'models' "
                                                "directory.\n"
                                                "Example: python random_models.py --dataset ascad --cluster false "
                                                "--architecture MLP --leakage id --amount 100 --lf cer_loss")

argparser.add_argument('--dataset', required=True, help="Profiling dataset to use: aes_hd, aes_hd_ext, ches_ctf, "
                                                        "ascad, ascad_desync50, ascad_variable.")
argparser.add_argument('--cluster', type=bool, required=True, help="Running on the TU Delft cluster: true/false")
argparser.add_argument('--architecture', required=True, help="Architecture type: MLP/CNN")
argparser.add_argument('--amount', type=int, required=True, help="Amount of random models to train")
argparser.add_argument('--leakage', required=True, help="Leakage type: ID/HW")
argparser.add_argument('--lf', default=None, help="Optional, specify a loss function. Use string identifier, "
                                                  "like 'cer_loss', 'rkl', or Keras losses like "
                                                  "'categorical_crossentropy'. If not provided, random losses are "
                                                  "used")
argparser.add_argument('--id', default='', help="Optional identifier to add to file names")
argparser.add_argument('--keybyte', default=2, type=int, help="Keybyte to attack. Default third keybyte (so keybyte=2)")

args = argparser.parse_args()

dataset = args.dataset
cluster = args.cluster == 'true'
arch = args.architecture
model_count = args.amount
leakage = args.leakage
lf = args.lf
ident = args.id
keybyte = args.keybyte

print(f'Generating {model_count} random models: dataset={dataset} arch={arch} leakage={leakage} ')

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

# Chipwhisperer data
if dataset == 'chipwhisperer':
    raw_traces = np.load(
        "C:\\Users\\maike\\PycharmProjects\\Thesis\\data\\chipwhisperer_data\\chipwhisperer_data\\data\\traces.npy")
    known_key = \
        np.load(
            "C:\\Users\\maike\\PycharmProjects\\Thesis\\data\\chipwhisperer_data\\chipwhisperer_data\\data\\key.npy")[0]
    pt = np.load(
        "C:\\Users\\maike\\PycharmProjects\\Thesis\\data\\chipwhisperer_data\\chipwhisperer_data\\data\\plain.npy")

    attack_key = known_key
    profiling_traces = raw_traces[0:8500]
    profiling_plaintext = pt[0:8500]

    attack_traces = raw_traces[8500:]
    attack_plaintext = pt[8500:]

    known_keys = []
    [known_keys.append(known_key) for i in range(len(profiling_traces))]

else:
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

print(f'Training on keybyte {keybyte} of {dataset}. Known key: {known_keys[0]}, attack key: {attack_key}')

sbox_output = np.array(
    [sbox[int(profiling_plaintext[i][keybyte]) ^ int(known_keys[i][keybyte])] for i in range(len(profiling_plaintext))])

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

scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
profiling_traces = scaler.fit_transform(profiling_traces)


def create_mlp(nr_layers, nr_nodes, activation, optimiser, learning_rate, loss, alpha=0.25, gamma=2.0):
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
        model.compile(loss=categorical_focal_loss(alpha=[np.full(num_classes, fill_value=alpha)], gamma=gamma),
                      optimizer=optimiser(learning_rate=learning_rate), metrics=['accuracy'])
    elif loss == 'flr_loss':
        model = tf.keras.models.Model(start, predictions)
        model.compile(loss=flr_loss(3, num_classes, alpha=alpha, gamma=gamma),
                      optimizer=optimiser(learning_rate=learning_rate), metrics=['accuracy'])
    else:
        model = tf.keras.models.Model(start, predictions)
        model.compile(loss=loss,
                      optimizer=optimiser(learning_rate=learning_rate), metrics=['accuracy'])
    return model


def create_cnn(cov_layers, cov_filters, cov_kernel, pool_size, pool_stride, pool_type, dense_layers,
               dense_neurons, activation, optimiser, learning_rate, loss, alpha=0.25, gamma=2.0):
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
        model.compile(loss=categorical_focal_loss(alpha=[np.full(num_classes, fill_value=alpha)], gamma=gamma),
                      optimizer=optimiser(learning_rate=learning_rate), metrics=['accuracy'])
    elif loss == 'flr_loss':
        model = tf.keras.models.Model(start, predictions)
        model.compile(loss=flr_loss(3, num_classes, alpha=alpha, gamma=gamma),
                      optimizer=optimiser(learning_rate=learning_rate), metrics=['accuracy'])
    else:
        model = tf.keras.models.Model(start, predictions)
        model.compile(loss=loss,
                      optimizer=optimiser(learning_rate=learning_rate), metrics=['accuracy'])
    return model


def select_random_mlp_hyperparams(amount, lf=None):
    layers = random.choices(range(2, 9, 1), k=amount)
    nodes = random.choices(range(100, 1100, 100), k=amount)
    batchsize = random.choices(range(100, 1100, 100), k=amount)
    epochs = random.choices([200], k=amount)
    learning_rate = random.choices(np.arange(0.000001, 0.001, 0.00002),
                                   k=amount)
    optimiser = random.choices([tf.keras.optimizers.Adam, tf.keras.optimizers.RMSprop], k=amount)
    activation = random.choices(['relu', 'tanh', 'elu', 'selu'], k=amount)
    if lf is None:
        loss = random.choices(
            ['categorical_crossentropy', 'mse', 'log_cosh', 'mean_squared_logarithmic_error',
             "categorical_hinge", "rkl", "cer_loss"], k=amount)
    else:
        loss = random.choices([lf], k=amount)

    all_hyperparams = list(
        zip(layers, nodes, activation, optimiser, learning_rate, loss, epochs, batchsize))

    print(f'Created {len(all_hyperparams)} sets of MLP hyperparameters')

    return all_hyperparams


def create_mlp_models(hyperparam_set):
    models = []
    for hyperparams in hyperparam_set:
        # TODO: Random alpha/gamma are chosen and stored regardless of loss, so e.g. CCE they are stored but not used
        alpha = random.choice([0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9])
        gamma = random.choice([0, 0.5, 1, 2, 5])
        model = {'model': create_mlp(*hyperparams[:6], alpha, gamma), 'hyperparams': hyperparams, 'alpha': alpha,
                 'gamma': gamma}
        model['trainable'] = count_params(model['model'].trainable_weights)
        models.append(model)
    return models


def select_random_cnn_hyperparams(amount, lf=None):
    cov_layers = random.choices(range(1, 3, 1), k=amount)
    cov_filters = random.choices(range(8, 36, 4), k=amount)
    cov_kernel = random.choices(range(10, 22, 2), k=amount)
    pool_size = random.choices(range(2, 6, 1), k=amount)
    pool_stride = random.choices(range(5, 15, 5), k=amount)
    pool_type = random.choices(['max_pooling', 'avg_pooling'], k=amount)

    dense_layers = random.choices(range(2, 4, 1), k=amount)
    dense_neurons = random.choices(range(100, 1100, 100), k=amount)

    optimiser = random.choices([tf.keras.optimizers.Adam, tf.keras.optimizers.RMSprop], k=amount)
    learning_rate = random.choices(np.arange(0.000001, 0.001, 0.00002),
                                   k=amount)
    activation = random.choices(['relu', 'tanh', 'elu', 'selu'], k=amount)

    if lf is None:
        loss = random.choices(
            ['categorical_crossentropy', 'mse', 'log_cosh', 'mean_squared_logarithmic_error',
             "categorical_hinge", "rkl", "cer_loss"], k=amount)
    else:
        loss = random.choices([lf], k=amount)

    batchsize = random.choices(range(100, 1100, 100), k=amount)
    epochs = random.choices([200], k=amount)

    all_hyperparams = list(
        zip(cov_layers, cov_filters, cov_kernel, pool_size, pool_stride, pool_type, dense_layers,
            dense_neurons, activation, optimiser, learning_rate, loss, epochs, batchsize))

    print(f'Created {len(all_hyperparams)} random sets of CNN hyperparameters')

    return all_hyperparams


def create_cnn_models(hyperparam_set):
    models = []

    for hyperparams in hyperparam_set:
        alpha = random.choice([0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9])
        gamma = random.choice([0, 0.5, 1, 2, 5])
        model = {'model': create_cnn(*hyperparams[:12], alpha, gamma), 'hyperparams': hyperparams, 'alpha': alpha,
                 'gamma': gamma}
        model['trainable'] = count_params(model['model'].trainable_weights)
        models.append(model)
    return models


def fit_models(x, y, models, store=True):
    for i in range(len(models)):

        model_data = {}

        model = models[i]
        print(f'Training model {model}')
        start = datetime.now()
        model["model"].fit(x, y, epochs=model["hyperparams"][-2], batch_size=model["hyperparams"][-1])
        end = datetime.now()
        training_time = (end - start).total_seconds()
        print(f'Finished training {model["hyperparams"]}, training time: {end - start}')

        model_data['hyperparams'] = model['hyperparams']
        model_data['training_time'] = training_time
        model_data['alpha'] = model['alpha']
        model_data['gamma'] = model['gamma']

        if store:
            if lf is not None:
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
            else:
                model['model_id'] = datetime.now().strftime('%S_%M_%H_%d_%m')

                if cluster:
                    model['model'].save(
                        cluster_root + 'thesis/models/' + arch + '_' + dataset + '-' + leakage + '_' + ident + str(
                            model[
                                'model_id']) + '.h5')
                    np.save(cluster_root + 'thesis/models/' + arch + '_' + dataset + '-' + leakage + '_' + ident + str(
                        model[
                            'model_id']) + '.npy',
                            model_data)
                else:
                    model['model'].save(
                        '../models/' + arch + '_' + dataset + '-' + leakage + '_' + ident + str(
                            model[
                                'model_id']) + '.h5')
                    np.save('../models/' + arch + '_' + dataset + '-' + leakage + '_' + ident + str(
                        model[
                            'model_id']) + '.npy',
                            model_data)

    print('Finished training all models')


def train_random_mlp_models(amount, device):
    with tf.device('/' + device + ':0'):
        print(f"Selecting {amount} random sets of hyperparameters")
        hyperparam_set = select_random_mlp_hyperparams(amount, lf)

        print(f"Creating models")
        models = create_mlp_models(hyperparam_set)

        print(f'Training models')
        fit_models(profiling_traces, to_categorical(labels, num_classes=num_classes), models)


def train_random_cnn_models(amount, device):
    with tf.device('/' + device + ':0'):
        print(f"Selecting {amount} random sets of hyperparameters")
        hyperparam_set = select_random_cnn_hyperparams(amount, lf)

        print(f"Creating models")
        models = create_cnn_models(hyperparam_set)

        print(f'Training models')
        fit_models(profiling_traces, to_categorical(labels, num_classes=num_classes), models)


if arch == 'CNN':
    train_random_cnn_models(model_count, 'GPU')

elif arch == 'MLP':
    train_random_mlp_models(model_count, 'GPU')
