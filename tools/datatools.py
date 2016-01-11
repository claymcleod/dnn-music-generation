# -*- coding: utf-8 -*-
#
# Copyright © 2016 Clay L. McLeod <clay.l.mcleod@gmail.com>
#
# Distributed under terms of the MIT license.

from __future__ import print_function

import os, glob
import numpy as np

from scipy.io import wavfile

class datatools(object):

    @staticmethod
    def ensure_dir_exists(dirname):
        if not os.path.exists(dirname):
            os.makedirs(dirname)

    @staticmethod
    def convert_mp3s_to_wav(data_dir):

        tmp_dir = os.path.join(data_dir, 'tmp')
        wav_dir = os.path.join(data_dir, 'wav')

        # Ensure that our output dirs exist
        datatools.ensure_dir_exists(tmp_dir)
        datatools.ensure_dir_exists(wav_dir)

        # Begin MP3 processing
        print('Converting to stereo to mono...')
        data_glob_path = os.path.join(data_dir, '*.mp3')
        for data_filename in glob.glob(data_glob_path):
            name = data_filename.split('/')[-1]
            tmp_filename = os.path.join(tmp_dir, name)
            cmd = 'lame --quiet -a -m m {} {}'.format(data_filename, tmp_filename)
            os.system(cmd)

        print('Converting to mono to wav...')
        tmp_data_glob_path = os.path.join(tmp_dir, '*.mp3')
        for tmp_filename in glob.glob(tmp_data_glob_path):
            name = tmp_filename.split('/')[-1].replace('.mp3','.wav')
            wav_filename = os.path.join(wav_dir, name)
            cmd = 'lame --quiet --decode {} {} --resample 44.1'.format(tmp_filename, wav_filename)
            os.system(cmd)

    @staticmethod
    def convert_wav_to_fft(data_dir, block_size, seql):

        fft_dir = os.path.join(data_dir, 'fft')
        datatools.ensure_dir_exists(fft_dir)

        # Split into chunks

        print('Converting wav to fft...')
        wav_dir = os.path.join(data_dir, 'wav')
        wav_glob_path = os.path.join(wav_dir, '*.wav')
        for wav_data_file in glob.glob(wav_glob_path):
            np_filename = wav_data_file.split('/')[-1].replace('.wav','')
            np_data_file = os.path.join(fft_dir, np_filename)
            wav_np_array = wavfile.read(wav_data_file)[1].astype('float32') / 32767.0

            block_lists = []
            total_samples = wav_np_array.shape[0]
	    num_samples_so_far = 0
	    while(num_samples_so_far < total_samples):
	        block = wav_np_array[num_samples_so_far:num_samples_so_far+block_size]
		if(block.shape[0] < block_size):
		    padding = np.zeros((block_size - block.shape[0],))
		    block = np.concatenate((block, padding))
                fft_block = np.fft.fft(block)
                new_block = np.concatenate((np.real(fft_block), np.imag(fft_block)))
		block_lists.append(new_block)
		num_samples_so_far += block_size

            # Create X and Y training sets

            start_index = 0
            training_instances = len(block_lists)-seql
            freq_bins = len(block_lists[0])
            X_train = np.zeros((training_instances, seql, freq_bins))
            #Y_train = np.zeros((training_instances, 1, freq_bins))
            Y_train = np.zeros((training_instances, freq_bins))

            while start_index+seql+1 < len(block_lists):
                print(str(start_index)+"/"+str(len(block_lists)-seql))

                for i in range(seql):
                    for j in range(freq_bins):
                        X_train[start_index][i][j] = block_lists[start_index+i][j]
                for j in range(freq_bins):
                    Y_train[start_index][j] = block_lists[start_index+seql+1][j]

                start_index = start_index+1

            np.save(np_data_file+'_x', X_train)
            np.save(np_data_file+'_y', Y_train)

        return block_lists

    @staticmethod
    def convert_gen_to_out(data_dir):

        gen_dir = os.path.join(data_dir, 'gen')
        out_dir = os.path.join(data_dir, 'out')

        datatools.ensure_dir_exists(gen_dir)
        datatools.ensure_dir_exists(out_dir)

        gen_glob_file_path = os.path.join(gen_dir, '*.npy')
        print('Converting gen to out...')

        for npy_data_file in glob.glob(gen_glob_file_path):
            blocks = []
            filename = npy_data_file.split('/')[-1]
            wav_filename = os.path.join(out_dir, filename.replace('.npy','.wav'))

            data_as_fft = np.load(npy_data_file)

            for fft_block in data_as_fft:
                real_imag_split = fft_block.shape[0] // 2
                real = fft_block[0:real_imag_split]
                imag = fft_block[real_imag_split:]
                time_domain = np.fft.ifft(real + 1.0j * imag)
                blocks.append(time_domain)

            song_blocks = np.concatenate(blocks)
            song_blocks = song_blocks * 32767.0
            song_blocks = song_blocks.astype('int16')
            wavfile.write(wav_filename, 44100, song_blocks)
