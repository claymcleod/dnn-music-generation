# -*- coding: utf-8 -*-
#
# Copyright © 2016 Clay L. McLeod <clay.l.mcleod@gmail.com>
#
# Distributed under terms of the MIT license.

from __future__ import print_function

import os, glob
import numpy as np

from scipy.io import wavfile

from sets import Set
from nntools import nntools
from terminaltools import write_flush

from keras.callbacks import ModelCheckpoint

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
        data_glob_path = os.path.join(data_dir, '*.mp3')
	mp3_data_files = glob.glob(data_glob_path)
	if len(mp3_data_files) > 0:
            write_flush('-- Converting stereo to mono...')
            for data_filename in mp3_data_files:
                name = data_filename.split('/')[-1]
                tmp_filename = os.path.join(tmp_dir, name)
                cmd = 'lame --quiet -a -m m {} {}'.format(data_filename, tmp_filename)
                os.system(cmd)
            write_flush('finished.      \n')

        tmp_data_glob_path = os.path.join(tmp_dir, '*.mp3')
	tmp_data_files = glob.glob(tmp_data_glob_path)
	if len(tmp_data_files) > 0:
            write_flush('-- Converting to mono to wav...')
            for tmp_filename in tmp_data_files:
                name = tmp_filename.split('/')[-1].replace('.mp3','.wav')
                wav_filename = os.path.join(wav_dir, name)
                cmd = 'lame --quiet --decode {} {} --resample 44.1'.format(tmp_filename, wav_filename)
                os.system(cmd)
            write_flush('finished.      \n')

	data_folder_wav_glob_path = os.path.join(data_dir, '*.wav')
	wav_files_in_data_folder = glob.glob(data_folder_wav_glob_path)
	if len(wav_files_in_data_folder) > 0:
	    write_flush('-- Moving existing wavs...')
	    for wav_filename in wav_files_in_data_folder:
	        name = wav_filename.split('/')[-1]
	        new_wav_filename = os.path.join(wav_dir, name)
	        cmd = 'cp {} {}'.format(wav_filename, new_wav_filename)
	        os.system(cmd)
            write_flush('finished.      \n')

    @staticmethod
    def convert_wav_to_fft(data_dir, block_size, seql):

        fft_dir = os.path.join(data_dir, 'fft')
        datatools.ensure_dir_exists(fft_dir)

        # Split into chunks

        wav_dir = os.path.join(data_dir, 'wav')
        wav_glob_path = os.path.join(wav_dir, '*.wav')
        wav_files = glob.glob(wav_glob_path)
        block_lists = []

        if len(wav_files) > 0:
            write_flush('-- Converting wav to fft...')
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
                Y_train = np.zeros((training_instances, freq_bins))

                while True:
                    if start_index+seql+1 < len(block_lists):
                        progress_str = '\r-- Converting wav to fft... ({}/{})'.format(start_index, len(block_lists)-seql-1)
                        write_flush(progress_str)
                    else:
                        write_flush('\r-- Converting wav to fft...finished.      \n')
                        break

                    for i in range(seql):
                        for j in range(freq_bins):
                            X_train[start_index][i][j] = block_lists[start_index+i][j]
                    for j in range(freq_bins):
                        Y_train[start_index][j] = block_lists[start_index+seql+1][j]

                    start_index = start_index+1

                write_flush('-- Saving fft file...')
                np.save(np_data_file+'_x', X_train)
                np.save(np_data_file+'_y', Y_train)
                write_flush('finished.      \n')

        return block_lists

    @staticmethod
    def convert_gen_to_out(data_dir):

        gen_dir = os.path.join(data_dir, 'gen')
        out_dir = os.path.join(data_dir, 'out')

        datatools.ensure_dir_exists(gen_dir)
        datatools.ensure_dir_exists(out_dir)

        gen_glob_file_path = os.path.join(gen_dir, '*.npy')
        write_flush('-- Converting gen to out...')

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
            write_flush('finished.      \n')


    @staticmethod
    def train_dnn(data_dir, epochs_per_round, max_training_iterations):
        fft_dir = os.path.join(data_dir, 'fft')
        fft_glob = os.path.join(fft_dir, '*.npy')

        weights_dir = os.path.join(data_dir, 'weights')
        datatools.ensure_dir_exists(weights_dir)

        filenames = Set()

        for g in glob.glob(fft_glob):
            filenames.add(g.replace('_x.npy','').replace('_y.npy',''))

        for f in filenames:
            X_train = np.load(f+'_x.npy')
            y_train = np.load(f+'_y.npy')
            filename = f.split('/')[-1]
            weight_file = os.path.join(weights_dir, filename+'.hdf5')

            write_flush('# Training on \'{}\'\n\n'.format(filename))
            write_flush('-- Building dnn...')
            model = nntools.get_current_model(X_train.shape[2], 2048)
            write_flush('finished.       \n')
            checkpointer = ModelCheckpoint(filepath=weight_file, verbose=0, save_best_only=True)
            i = 0

            write_flush('-- Training model... ({}/{})'.format(i, max_training_iterations))
            while True:
                if i >= max_training_iterations:
	            break

    	        if os.path.exists(weight_file):
	            model.load_weights(weight_file)

                model.fit(X_train, y_train, nb_epoch=epochs_per_round, verbose=0, validation_split=0.1, batch_size=64, callbacks=[checkpointer])

	        i = i + epochs_per_round
                write_flush('\r-- Training model... ({}/{})'.format(i, max_training_iterations))

            write_flush('\r-- Training model...finished.\n')


    @staticmethod
    def generate_from_dnn(data_dir, seql, generate_x_blocks):

        fft_dir = os.path.join(data_dir, 'fft')
        fft_glob = os.path.join(fft_dir, '*.npy')

        weights_dir = os.path.join(data_dir, 'weights')
        gen_dir = os.path.join(data_dir, 'gen')

        datatools.ensure_dir_exists(weights_dir)
        datatools.ensure_dir_exists(gen_dir)

        filenames = Set()

        for g in glob.glob(fft_glob):
            filenames.add(g.replace('_x.npy','').replace('_y.npy',''))

        for f in filenames:
            X_train = np.load(f+'_x.npy')
            y_train = np.load(f+'_y.npy')
            filename = f.split('/')[-1]
            weight_file = os.path.join(weights_dir, filename+'.hdf5')
            trained_file_location = os.path.join(gen_dir, filename)

            write_flush("# Generating for '{}'\n\n".format(filename))
            write_flush("-- Preparing data...")
            output = np.zeros((generate_x_blocks+seql, X_train.shape[1], X_train.shape[2]))
            output = np.append(X_train[0:seql], output, axis=0)
            fft_output = np.zeros((generate_x_blocks, X_train.shape[2]))
            write_flush("finished.\n")

            write_flush("-- Building dnn...")
            model = nntools.get_current_model(X_train.shape[2], 2048)
            write_flush("finished.\n")

            write_flush("-- Loading weights...")
            if os.path.exists(weight_file):
	        model.load_weights(weight_file)
            write_flush("finished.\n")

            i = 0
            l = len(X_train)
            write_flush("-- Generating...")
            while True:
	        write_flush("\r-- Generating... ({}/{})".format(i, fft_output.shape[0]))
	        if i >= fft_output.shape[0]:
	            break

	        next_val = model.predict(output[i:i+seql])

                for k in range(0, seql-1):
	            for x in range(0, output.shape[2]):
		        output[i+seql+1][k][x] = next_val[k][x]
		        fft_output[i][x] = next_val[0][x]

	        i = i + 1

            write_flush("\r-- Generating... finished.\n")
            write_flush("-- Saving numpy array...")
            np.save(trained_file_location, fft_output)
            write_flush("finished.\n\n")

