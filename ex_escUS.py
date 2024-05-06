import wandb
import numpy as np
import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import argparse
from sklearn import metrics
import torch.nn.functional as F
import csv

from datasets.escUS import get_data_set
from models.mn.model import get_model as get_mobilenet
from models.dymn.model import get_model as get_dymn
from models.preprocess import AugmentMelSTFT
from helpers.init import worker_init_fn
from helpers.utils import NAME_TO_WIDTH, exp_warmup_linear_down, mixup


def preprocess_esc(
        source_dir,
        model,
        n_mels=128,
        resample_rate=32000,
        window_size=800,
        hop_size=320,
        n_fft=1024,
        freqm=0,
        timem=0,
        fmin=0,
        fmax=None,
        fmin_aug_range=10,
        fmax_aug_range=2000,
        cuda=True,
        no_roll=False,
        no_wavmix=False,
        gain_augment=12,
        num_workers=12,
        mixup_alpha=None,
        batch_size=128
):
    device = torch.device('cuda') if cuda and torch.cuda.is_available() else torch.device('cpu')

    # model to preprocess waveform into mel spectrograms
    mel = AugmentMelSTFT(n_mels=n_mels,
                         sr=resample_rate,
                         win_length=window_size,
                         hopsize=hop_size,
                         n_fft=n_fft,
                         freqm=freqm,
                         timem=timem,
                         fmin=fmin,
                         fmax=fmax,
                         fmin_aug_range=fmin_aug_range,
                         fmax_aug_range=fmax_aug_range
                         )
    mel.to(device)
    model.to(device)

    dl = DataLoader(dataset=get_data_set(resample_rate=resample_rate,
                                                  gain_augment=gain_augment,
                                                  source_link=source_dir),
                    worker_init_fn=worker_init_fn,
                    num_workers=num_workers,
                    batch_size=batch_size,
                    shuffle=False)

    results_path = os.path.join('../drive/MyDrive/8321_Final_Project/', 'results.csv')
    with open(results_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['filename', 'target', 'softmax_confidence'])

            for batch in tqdm(dl, desc="Processing batches", unit="batch"):
                mel.eval()
                model.eval()
                x, filenames = batch
                x = x.to(device)

                with torch.no_grad():
                    x = _mel_forward(x, mel)
                    y_hat, _ = model(x)
                    y_hat_probs = F.softmax(y_hat, dim=1)
                    max_probs, indices = torch.max(y_hat_probs, dim=1)  # Get max confidence and indices

                    for i, filename in enumerate(filenames):
                        writer.writerow([filename, indices[i].item(), round(max_probs[i].item(), 2)])

    print(f"Results saved to {results_path}")


def _mel_forward(x, mel):
    old_shape = x.size()
    x = x.reshape(-1, old_shape[2])
    x = mel(x)
    x = x.reshape(old_shape[0], old_shape[1], x.shape[1], x.shape[2])
    return x
