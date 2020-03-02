#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/24 17:06
# @Author  : Evan
# @File    : speaker_diarization.py
from __future__ import absolute_import
from __future__ import print_function
import os

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import warnings

import toolkits

# ===========================================
#        Parse the argument
# ===========================================
import argparse

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
# set up training configuration.
parser.add_argument('--gpu', default='', type=str)
parser.add_argument('--resume', default=r'pretrained/weights.h5', type=str)
parser.add_argument('--data_path', default='4persons', type=str)
parser.add_argument('--sample_rate', default=16000, type=int)
# set up network configuration.
parser.add_argument('--net', default='resnet34s', choices=['resnet34s', 'resnet34l'], type=str)
parser.add_argument('--ghost_cluster', default=2, type=int)
parser.add_argument('--vlad_cluster', default=8, type=int)
parser.add_argument('--bottleneck_dim', default=512, type=int)
parser.add_argument('--aggregation_mode', default='gvlad', choices=['avg', 'vlad', 'gvlad'], type=str)
# set up learning rate, training loss and optimizer.
parser.add_argument('--loss', default='softmax', choices=['softmax', 'amsoftmax'], type=str)
parser.add_argument('--test_type', default='normal', choices=['normal', 'hard', 'extend'], type=str)

args = parser.parse_args()


# global args
def euclid_similar(matrix_1, matrix_2):  # calc speaker-embeddings similarity in pretty format output.
    dist = np.linalg.norm(matrix_1 - matrix_2)

    return dist


# ===============================================
#       code from Arsha for loading data.
# ===============================================
def load_wav(vid_path, sr=args.sample_rate):
    wa, sr_ret = librosa.load(vid_path, sr=sr, mono=True)  # 对输入音频读取为单声道，采样平率16000KHz/s
    assert sr_ret == sr

    intervals = librosa.effects.split(wa, top_db=25)  # 去静音，对响度低于20db的部分
    # wav_output = []
    # for sliced in intervals:
    #     wav_output.extend(wav[sliced[0]:sliced[1]])
    # wav_output = np.array(wav_output)

    # return wav, wav_output
    return wa, intervals


def lin_spectogram_from_wav(wavs, hop_length, win_length, n_fft=1024):
    linear = librosa.stft(wavs, n_fft=n_fft, win_length=win_length, hop_length=hop_length)  # linear spectrogram

    return linear.T


def load_data(wv, split=False, win_length=400, sr=args.sample_rate, hop_length=160, n_fft=512, min_slice=720):
    linear_spect = lin_spectogram_from_wav(wv, hop_length, win_length, n_fft)
    mag, _ = librosa.magphase(linear_spect)  # magnitude
    mag_t = mag.T
    freq, time = mag_t.shape
    spec_mag = mag_t

    utterances_spec = []

    if split:
        min_spec = min_slice // (1000 // (sr // hop_length))  # The minimum timestep of each slice in spectrum
        rand_starts = np.random.randint(0, time, 10)  # generate 10 slices at most.
        for start in rand_starts:
            if time - start <= min_spec:
                continue
            rand_duration = np.random.randint(min_spec, time - start)
            spec_mag = mag_t[:, start:start + rand_duration]

            # preprocessing, subtract mean, divided by time-wise var
            mu = np.mean(spec_mag, 0, keepdims=True)
            std = np.std(spec_mag, 0, keepdims=True)
            spec_mag = (spec_mag - mu) / (std + 1e-5)
            utterances_spec.append(spec_mag)
    else:
        # preprocessing, subtract mean, divided by time-wise var
        mu = np.mean(spec_mag, 0, keepdims=True)
        std = np.std(spec_mag, 0, keepdims=True)
        spec_mag = (spec_mag - mu) / (std + 1e-5)
        utterances_spec.append(spec_mag)

    return utterances_spec


def init_model():
    # gpu configuration
    toolkits.initialize_GPU(args)

    import model

    # construct the data generator.
    params = {'dim': (257, None, 1),
              'nfft': 512,
              'min_slice': 720,
              'win_length': 400,
              'hop_length': 160,
              'n_classes': 5994,
              'sampling_rate': 16000,
              'normalize': True,
              }

    network_eval = model.vggvox_resnet2d_icassp(input_dim=params['dim'],
                                                num_class=params['n_classes'],
                                                mode='eval', args=args)

    if args.resume:
        if os.path.isfile(args.resume):
            network_eval.load_weights(os.path.join(args.resume), by_name=True)
        else:
            raise IOError("==> no checkpoint found at '{}'".format(args.resume))
    else:
        raise IOError('==> please type in the model to load')

    return network_eval


def get_id_embedding(eval_model, input_data):
    spec = np.expand_dims(input_data, -1)
    v = eval_model.predict(spec)

    return v


def split_wave_2_index_base_1by1(model, audio_clips, raw_wav, threshold=0.5):
    """
    tips：不建议使用
    将音频按静音分成片段，每一段依次与后一段比较相似度，小于阈值断定为同一个人的声音
    """
    a_people = []
    b_people = []
    a_slice = []
    b_slice = []
    for sliced in audio_clips:
        need_check = raw_wav[sliced[0]:sliced[1]]
        spec = load_data(need_check)
        id_vector = get_id_embedding(model, spec)

        # a，b均空
        if all([not a_people, not b_people]):
            a_people.extend(id_vector)
            a_slice.append(sliced)
            continue

        # a不空，b空
        if all([a_people, not b_people]):
            a_score = euclid_similar(a_people[-1], id_vector)
            if a_score <= threshold:
                a_people.extend(id_vector)
                a_slice.append(sliced)
            else:
                b_people.extend(id_vector)
                b_slice.append(sliced)
            continue

        # a，b均不空
        if all([a_people, b_people]):
            a_score = euclid_similar(a_people[-1], id_vector)
            b_score = euclid_similar(b_people[-1], id_vector)
            if a_score <= b_score:
                a_people.extend(id_vector)
                a_slice.append(sliced)
            else:
                b_people.extend(id_vector)
                b_slice.append(sliced)
            continue

    a_slices = np.array(a_slice)
    b_slices = np.array(b_slice)

    return a_slices, b_slices


def split_wave_2_index_base_mean(model, audio_clips, raw_wav, threshold=0.5):
    """
    tips：不建议使用
    将音频按静音分成片段，预设a、b两组，每一段依次分别与a和b组内所有向量的均值比较相似度，哪个更相近就归入哪一组
    """
    a_people = np.zeros([1, 512])
    b_people = np.zeros([1, 512])
    a_slice = []
    b_slice = []
    for sliced in audio_clips:
        need_check = raw_wav[sliced[0]:sliced[1]]
        spec = load_data(need_check)
        id_vector = get_id_embedding(model, spec)

        if all([not np.sum(a_people), not np.sum(b_people)]):
            a_people = np.add(a_people, id_vector)
            a_slice.append(sliced)
            print('1')
            continue

        # a不空，b空
        if all([bool(np.sum(a_people)), not np.sum(b_people)]):
            a_score = euclid_similar(np.mean(a_people, axis=0, keepdims=True), id_vector)
            if a_score <= threshold:
                a_people = np.append(a_people, id_vector, axis=0)
                a_slice.append(sliced)
            else:
                b_people = np.add(b_people, id_vector)
                b_slice.append(sliced)

            print('2')
            continue

        # a，b均不空
        if all([bool(np.sum(a_people)), bool(np.sum(b_people))]):
            a_score = euclid_similar(np.mean(a_people, axis=0, keepdims=True), id_vector)
            b_score = euclid_similar(np.mean(b_people, axis=0, keepdims=True), id_vector)
            if a_score <= b_score:
                a_people = np.append(a_people, id_vector, axis=0)
                a_slice.append(sliced)
            else:
                b_people = np.append(b_people, id_vector, axis=0)
                b_slice.append(sliced)
            print('3')
            continue

    a_slices = np.array(a_slice)
    b_slices = np.array(b_slice)

    return a_slices, b_slices


def split_wave_2_index_base_kmeans(model, audio_clips, raw_wav):
    """
    tips：建议使用
    将音频按静音分成片段，对所有片段对应的向量做聚类，k=2
    """
    vectors = []
    for sliced in audio_clips:
        need_check = raw_wav[sliced[0]:sliced[1]]
        # print(type(need_check))
        # print(need_check.shape)
        # break
        spec = load_data(need_check)
        id_vector = get_id_embedding(model, spec)
        vectors.append(id_vector)

    vectors = np.squeeze(np.array(vectors))
    km_model = KMeans(n_clusters=2).fit(vectors)

    label_index = km_model.labels_

    a_index = np.squeeze(np.argwhere(label_index == 1))
    b_index = np.squeeze(np.argwhere(label_index == 0))

    a_slices = np.array([audio_clips[int(i)] for i in a_index])
    b_slices = np.array([audio_clips[int(i)] for i in b_index])

    return a_slices, b_slices


def split_wave_2_index_base_kmeans_finer(model, audio_clips, raw_wav):
    """
    tips：不建议使用
    针对聚类效果不佳问题，对聚类后数量较多的组重聚类
    """
    no_silenced = []
    for sliced in audio_clips:
        need_check = raw_wav[sliced[0]:sliced[1]].tolist()
        no_silenced.extend(need_check)

    vectors = []
    split_num = round(len(no_silenced) / (16000 * 2))
    segs = []
    for i in range(split_num):
        if i != (split_num - 1):
            seg = no_silenced[16000 * i * 2:16000 * (i + 1) * 2]
        else:
            seg = no_silenced[16000 * i * 2:]

        segs.append(seg)
        spec = load_data(np.array(seg))
        id_vector = get_id_embedding(model, spec)
        vectors.append(id_vector)

    diff_value = [euclid_similar(vectors[i], vectors[i - 1]) for i in range(1, len(vectors))]

    triple_vlaue = [diff_value[i:i + 3] for i in range(len(diff_value) - 3 + 1)]
    split_point = []
    for i, v in enumerate(triple_vlaue):
        if all([v[1] >= v[0], v[1] >= v[2]]):
            split_point.append(i)
            continue

    true_point = [i + 1 for i in split_point]
    seg_point = [0]
    for i in true_point:
        seg_point.append(i)
        seg_point.append(i + 1)
    seg_point = seg_point + [-2]

    audio_cs = []
    for i in range(len(seg_point) // 2):
        audio_cs.append([seg_point[2 * i], seg_point[2 * i + 1] + 1])

    final_vectors = []
    for sld in audio_cs:
        need_ck = segs[sld[0]:sld[1]]

        new_seg = []
        for j in need_ck:
            new_seg.extend(j)

        new_seg = np.array(new_seg)
        sp = load_data(new_seg)
        id_vr = get_id_embedding(model, sp)
        final_vectors.append(id_vr)

    final_vectors = np.squeeze(np.array(final_vectors))

    km_model = KMeans(n_clusters=2).fit(final_vectors)

    label_index = km_model.labels_

    a_index = np.squeeze(np.argwhere(label_index == 1))
    b_index = np.squeeze(np.argwhere(label_index == 0))

    a_slices = np.array([audio_cs[int(i)] for i in a_index])
    b_slices = np.array([audio_cs[int(i)] for i in b_index])

    ta = []
    for i in a_slices:
        nd_ck = segs[i[0]:i[1]]
        ang = []
        for j in nd_ck:
            ang.extend(j)
        ta.extend(ang)

    tb = []
    for i in b_slices:
        nd_ck = segs[i[0]:i[1]]
        bng = []
        for j in nd_ck:
            bng.extend(j)
        tb.extend(bng)

    a_w = np.squeeze(np.array(ta))
    b_w = np.squeeze(np.array(tb))

    return a_w, b_w


def save_wave_from_index(file_name, all_wav, slices, sr=args.sample_rate, recive_index=True):
    if recive_index:
        wav = []
        for sliced in slices:
            wav.extend(all_wav[sliced[0]:sliced[1]])
        a_wav = np.array(wav)
    else:
        a_wav = slices

    librosa.output.write_wav(file_name, a_wav, sr)


def fill_segment(wav, slices):
    total_point = wav.shape[0]
    sub_wav = np.zeros([total_point, ])
    for sliced in slices:
        indexes = np.array(range(sliced[0], sliced[1] + 1))
        select_wav = np.take(wav, indexes)
        np.put(sub_wav, indexes, select_wav)

    return sub_wav


def plot_wave(wav, slices_a, slices_b, title):
    sub_wav_a = fill_segment(wav, slices_a)
    sub_wav_b = fill_segment(wav, slices_b)

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.title('{}.wav'.format(title))
    librosa.display.waveplot(wav, sr=args.sample_rate)
    plt.subplot(3, 1, 2)
    plt.title('{}_part_1.wav'.format(title))
    librosa.display.waveplot(sub_wav_a, sr=args.sample_rate)
    plt.subplot(3, 1, 3)
    plt.title('{}_part_2.wav'.format(title))
    librosa.display.waveplot(sub_wav_b, sr=args.sample_rate)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                        wspace=None, hspace=0.8)
    plt.savefig(title)
    plt.show()


if __name__ == "__main__":
    # name_list = ['013379309735-DVS+20170501132732720-80012',
    #              '013226321221-DVS+20170505224008863-80331',
    #              '013293660543-DVS+20170503175702171-80349']
    #
    # model = init_model()
    #
    # for num, file_name in enumerate(name_list):
    #     wav, audio_clips = load_wav('{}.wav'.format(file_name))
    #     # a_slices, b_slices = split_wave_2_index_base_1by1(model, audio_clips, wav, threshold=0.6)
    #     # a_slices, b_slices = split_wave_2_index_base_mean(model, audio_clips, wav, threshold=0.6)
    #     a_slices, b_slices = split_wave_2_index_base_knn(model, audio_clips, wav)
    #
    #     save_wave_from_index('{}_a.wav'.format(file_name), wav, a_slices, 16000)
    #     save_wave_from_index('{}_b.wav'.format(file_name), wav, b_slices, 16000)
    #     # plot_wave(wav, a_slices, b_slices, title='sample_{}'.format(num))

    model = init_model()

    # for fn in wav_files:
    #     fname = os.path.basename(fn).split('.wav')[0]
    #     wav, audio_clips = load_wav(fn)
    #     a_slices, b_slices = split_wave_2_index_base_kmeans(model, audio_clips, wav)
    #
    #     save_wave_from_index(os.path.join(out_folder, '{}_a.wav'.format(fname)), wav, a_slices)
    #     save_wave_from_index(os.path.join(out_folder, '{}_b.wav'.format(fname)), wav, b_slices)

    wav, audio_clips = load_wav(r'your\target.wav')
    # print(wav)
    # print(min([(i[1] - i[0]) / 16000 for i in audio_clips.tolist()]))

    a_slices, b_slices = split_wave_2_index_base_kmeans(model, audio_clips, wav)
    save_wave_from_index(os.path.join('cache', '{}_a.wav'.format('test-1')), wav, a_slices)
    save_wave_from_index(os.path.join('cache', '{}_b.wav'.format('test-1')), wav, b_slices)

    # a_slices, b_slices = split_wave_2_index_base_kmeans_finer(model, audio_clips, wav)
    # save_wave_from_index(os.path.join('cache', '{}_a.wav'.format('test-1')), wav, a_slices, recive_index=False)
    # save_wave_from_index(os.path.join('cache', '{}_b.wav'.format('test-1')), wav, b_slices, recive_index=False)
