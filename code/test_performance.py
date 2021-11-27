import argparse
import os
import random
import sys

import h5py
from sklearn import preprocessing
from tensorflow.python.keras.utils.layer_utils import count_params

from cer import *
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
argparser.add_argument('--leakage', required=True, help="Leakage type: ID/HW")
argparser.add_argument('--keybyte', default=2, type=int, help="Keybyte to attack. Default third keybyte (so keybyte=2)")
argparser.add_argument('--stepsize', default=20, type=int, help="Stepsize in attack traces to use ")
argparser.add_argument('--runs', default=100, type=int, help="Amount of attacks per amount of traces")

args = argparser.parse_args()

dataset = args.dataset
cluster = args.cluster == 'true'
arch = args.architecture
leakage = args.leakage
keybyte = args.keybyte
runs = args.runs
stepsize = args.stepsize

print(args)

if cluster:
    # Root folder on TU Delft computing cluster
    cluster_root = "/tudelft.net/staff-bulk/ewi/insy/CYS/spicek/mkerkhof/"

    files = {"aes_hd": cluster_root + "thesis/data/aes_hd.h5",
             "aes_hd_ext": cluster_root + "thesis/data/aes_hd_ext.h5",
             "ches_ctf": cluster_root + "thesis/data/ches_ctf.h5",
             "ascad": cluster_root + "thesis/data/ASCAD.h5",
             "ascad_desync50": cluster_root + "thesis/data/ASCAD_desync50.h5",
             "ascad_variable": cluster_root + "thesis/data/ascad-variable.h5",
             "ascad_plain": cluster_root + "thesis/data/ASCAD.h5"}
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
    known_key = None
    attack_key = attack_metadata[0][1]
else:
    known_key = profiling_metadata[0][2]
    attack_key = attack_metadata[0][2]

if dataset == "ascad_variable":
    profiling_traces = np.array(profiling_traces)
    profiling_traces = profiling_traces[0:50000]

print(f'Attacking keybyte {keybyte} of {dataset}. Known key: {known_key}, attack key: {attack_key}')

hamming = [bin(n).count("1") for n in range(256)]

if dataset == "ascad_plain":
    attack_masks = [list(metadata[3]) for metadata in attack_metadata]

scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))

profiling_traces = scaler.fit(profiling_traces)
attack_traces = scaler.transform(attack_traces)


def ge_sr(traces, plaintexts, key, keybyte, model, sr_order, max_nr_attack_traces, nr_runs, stepsize):
    key_ranks = {}
    successes = {}

    zipped = list(zip(traces, plaintexts))

    for i in range(0, max_nr_attack_traces, stepsize):
        key_ranks[i] = 0
        successes[i] = 0

    for run in range(nr_runs):

        if run % 10 == 0:
            print(f"Run {run}...")

        traces_subset, plaintexts_subset = zip(*random.sample(zipped, max_nr_attack_traces + stepsize))

        key_probabilities = np.zeros(256)

        preds_all_traces = model.predict(np.array(traces_subset))

        if np.shape(preds_all_traces)[0] == 2:
            preds_all_traces = preds_all_traces[0]

        predictions = np.log(preds_all_traces + 1e-20)

        if leakage == 'id':
            for i in range(max_nr_attack_traces):
                for k in range(256):
                    key_probabilities[k] += predictions[i, sbox[int(plaintexts_subset[i][keybyte]) ^ k]]

                if i % stepsize == 0:
                    key_rank = list(np.argsort(key_probabilities)[::-1]).index(key[keybyte])
                    success = int(key_rank < sr_order)

                    key_ranks[i] += key_rank / runs
                    successes[i] += success / runs


        elif leakage == 'hamming':
            for i in range(max_nr_attack_traces):
                for k in range(256):
                    key_probabilities[k] += predictions[i, hamming[sbox[int(plaintexts_subset[i][keybyte]) ^ k]]]

                if i % stepsize == 0:
                    key_rank = list(np.argsort(key_probabilities)[::-1]).index(key[keybyte])
                    success = int(key_rank < sr_order)

                    key_ranks[i] += key_rank / runs
                    successes[i] += success / runs

    return key_ranks, successes


def test_performance(sz, model, model_id, model_data, arch='', save=True):
    result = {}

    print(f'Testing model {model_data}')

    max_amount = min(3000, len(attack_traces))

    ges, srs = ge_sr(attack_traces, attack_plaintext, attack_key, keybyte, model, sr_order=1,
                     max_nr_attack_traces=max_amount,
                     nr_runs=runs, stepsize=sz)

    print(model_data)
    print(model)

    result['arch'] = arch
    result['dataset'] = dataset
    result['config'] = model_data['hyperparams']
    result['training_time'] = model_data['training_time']
    result['trainable'] = count_params(model.trainable_weights)
    result['ges'] = ges
    result['srs'] = srs
    result['alpha'] = model_data['alpha']
    result['gamma'] = model_data['gamma']

    print(f'Done testing model! {result}')

    if save:
        if cluster:
            try:
                np.save(
                    cluster_root + "thesis/results/" + dataset + '/' + arch + '/' + dataset + '_' + arch + '_' + leakage + '_' + model_id,
                    result)
            except:
                print('Could not write results')
        else:
            np.save(
                "../results/" + dataset + '_' + arch + '_' + leakage + '_' + model_id,
                result)


def load_models(directory):
    models = []
    hyperparams = []
    for file in os.listdir(directory):
        if file.endswith('.h5') and dataset in file:
            model = tf.keras.models.load_model(directory + file)
            models.append(model)
            print('Loaded model')
            hp = np.load(directory + file.replace('.h5', '.npy'), allow_pickle=True)
            print(f'Loaded config:')
            print(hp)
            hyperparams.append(hp)
        else:
            print(f'Can not process {file}')
    return models, hyperparams


def load_model(file):
    return tf.keras.models.load_model(file)


def load_and_test(directory):
    for file in os.listdir(directory):
        if file.endswith('.h5') and dataset in file:
            model = tf.keras.models.load_model(directory + file, custom_objects={'ranking_loss_sca': rankingloss(),
                                                                                 'dummyloss': dummyloss,
                                                                                 'cer': cer_loss,
                                                                                 'cos_cross_loss': cosine_crossentropy(),
                                                                                 'categorical_focal_loss_fixed': categorical_focal_loss,
                                                                                 'flr': flr_loss})
            print('Loaded model')
            print(directory + file)

            hyperparams = np.load(directory + file.replace('.h5', '.npy'), allow_pickle=True)[()]

            model_id = file.split('.')[0].replace(arch + '_' + dataset + '-' + leakage + '_', '')
            print(model_id)

            print(hyperparams)
            test_performance(stepsize, model, model_id, hyperparams, arch, save=True)


with tf.device('/CPU:0'):
    if cluster:
        load_and_test(cluster_root + 'thesis/models/')
    else:
        load_and_test('../models/')
