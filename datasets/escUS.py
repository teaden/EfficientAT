import os
from torch.utils.data import Dataset as TorchDataset
import torch
import numpy as np
import pandas as pd
import librosa

from datasets.helpers.audiodatasets import PreprocessDataset, get_roll_func

# specify ESC50 location in 'dataset_dir'
# 3 files have to be located there:
# - FSD50K.eval_mp3.hdf
# - FSD50K.val_mp3.hdf
# - FSD50K.train_mp3.hdf
# follow the instructions here to get these 3 files:
# https://github.com/kkoutini/PaSST/tree/main/esc50


def pad_or_truncate(x, audio_length):
    """Pad all audio to specific length."""
    if len(x) <= audio_length:
        return np.concatenate((x, np.zeros(audio_length - len(x), dtype=np.float32)), axis=0)
    else:
        return x[0: audio_length]


def pydub_augment(waveform, gain_augment=0):
    if gain_augment:
        gain = torch.randint(gain_augment * 2, (1,)).item() - gain_augment
        amp = 10 ** (gain / 20)
        waveform = waveform * amp
    return waveform


class MixupDataset(TorchDataset):
    """ Mixing Up wave forms
    """

    def __init__(self, dataset, beta=2, rate=0.5):
        self.beta = beta
        self.rate = rate
        self.dataset = dataset
        print(f"Mixing up waveforms from dataset of len {len(dataset)}")

    def __getitem__(self, index):
        if torch.rand(1) < self.rate:
            x1, f1, y1 = self.dataset[index]
            idx2 = torch.randint(len(self.dataset), (1,)).item()
            x2, f2, y2 = self.dataset[idx2]
            l = np.random.beta(self.beta, self.beta)
            l = max(l, 1. - l)
            x1 = x1 - x1.mean()
            x2 = x2 - x2.mean()
            x = (x1 * l + x2 * (1. - l))
            x = x - x.mean()
            return x, f1, (y1 * l + y2 * (1. - l))
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


class AudioSetDataset(TorchDataset):
    def __init__(self, audiopath, resample_rate=32000, classes_num=50,
                 clip_length=5, gain_augment=0):
        """
        Reads the mp3 bytes from HDF file decodes using av and returns a fixed length audio wav
        """
        self.resample_rate = resample_rate

        self.clip_length = clip_length * resample_rate
        self.classes_num = classes_num
        self.gain_augment = gain_augment
        self.audiopath = audiopath
        self.filenames = []
        print("Loading files...")
        for root, _, files in os.walk(audiopath):
            for file in files:
                full_path = os.path.join(root, file)
                self.filenames.append(full_path)  # Store the full path
        print(f"{len(self.filenames)} files for inference")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):

        filename = self.filenames[index]
        waveform, _ = librosa.load(filename, sr=self.resample_rate, mono=True)

        if self.gain_augment:
            waveform = pydub_augment(waveform, self.gain_augment)
        waveform = pad_or_truncate(waveform, self.clip_length)
        return waveform.reshape(1, -1), filename


def get_base_data_set(resample_rate=32000, gain_augment=0, source_link=''):
    audiopath = source_link
    ds = AudioSetDataset(audiopath=audiopath, resample_rate=resample_rate, gain_augment=gain_augment)
    return ds

def get_data_set(resample_rate=32000, gain_augment=0, source_link=''):
    ds = get_base_data_set(resample_rate=resample_rate, gain_augment=gain_augment, source_link=source_link)
    return ds
