"""
Author: Allen Chang
Date: January 08, 2023
"""

import os
from typing import List, Tuple

import torch
import numpy as np
import pandas as pd
from ast import literal_eval
from torch.utils.data import Dataset
import csv
import json
from PIL import Image


class AVDataset(Dataset):
    def __init__(self,
                 dir_data: str,
                 id_noise_list: List[str],
                 aux_modality: str,
                 pad_token: int,
                 max_target_length: int,
                 load_noise: bool = False):
        self.id_noise_list = id_noise_list
        self.aux_modality = aux_modality
        self.n_sets = len(id_noise_list)
        self.pad_token = pad_token
        self.max_target_length = max_target_length
        self.load_noise = load_noise

        assert aux_modality in ["resnet", "clip", "obj_count"], "aux_modality must be one of {'resnet', 'clip', 'obj_count'}"

        self.dirs_audio = []
        for id_noise in id_noise_list:
            self.dirs_audio.append(os.path.join(dir_data, 'audio', id_noise))
        self.vision = pd.read_csv(os.path.join(dir_data, f'{aux_modality}.csv'))
        self.target = pd.read_csv(os.path.join(dir_data, 'target.csv'))
        self.noise = pd.read_csv(os.path.join(dir_data, 'noise.csv'))

        self.indices = self.target[['id_assignment', 'idx_instruction']]
        self.indices = self.indices.astype({"idx_instruction": int}).astype({"idx_instruction": str})
        self.vision = self.vision.iloc[:, 2:].values
        self.target_str = self.target['transcript_str'].values
        self.target = self.target['transcript_tokens'].apply(literal_eval).values

        if self.load_noise:
            self.noise_list = []
            for id_noise in id_noise_list:
                self.noise_list.append(self.noise.loc[self.noise['id_noise'] == id_noise, 'idxs'].apply(literal_eval).values)

    def __len__(self) -> int:
        return len(self.indices) * self.n_sets

    def __getitem__(self, 
                    idx: int) -> Tuple[Tuple, Tuple]:
        index = int(idx / self.n_sets)
        idx_set = idx % self.n_sets
        f = '_'.join(self.indices.iloc[index]) + '.pt'

        x_audio = torch.load(os.path.join(self.dirs_audio[idx_set], f), map_location='cpu')
        vision = torch.tensor(self.vision[index], dtype=torch.float)
        target_str = self.target_str[index]
        target = torch.tensor(np.pad(self.target[index],
                                     (0, max(0, self.max_target_length - len(self.target[index]))),
                                     mode='constant',
                                     constant_values=self.pad_token))

        if self.load_noise:
            if self.id_noise_list[idx_set] in ['clean', 'mix_clean']:
                noise = torch.tensor([], dtype=torch.long)
            else:
                noise = torch.tensor(self.noise_list[idx_set][index], dtype=torch.long)

            return (x_audio, vision), (target, target_str, noise)
        else:
            return (x_audio, vision), (target, target_str)

class ImageAVDataset(Dataset):
    def __init__(self,
                 dir_data: str,
                 id_noise_list: List[str],
                 aux_modality: str,
                 pad_token: int,
                 max_target_length: int,
                 load_noise: bool = False):
        self.id_noise_list = id_noise_list
        self.aux_modality = aux_modality
        self.n_sets = len(id_noise_list)
        self.pad_token = pad_token
        self.max_target_length = max_target_length
        self.load_noise = load_noise
        self.dir_data = dir_data
        assert aux_modality in ["resnet", "clip", "obj_count"], "aux_modality must be one of {'resnet', 'clip', 'obj_count'}"

        self.dirs_audio = []
        for id_noise in id_noise_list:
            self.dirs_audio.append(os.path.join(dir_data, 'audio', id_noise))
        self.vision = pd.read_csv(os.path.join(dir_data, f'{aux_modality}.csv'))
        self.target = pd.read_csv(os.path.join(dir_data, 'target.csv'))
        self.noise = pd.read_csv(os.path.join(dir_data, 'noise.csv'))
        
        self.indices = self.target[['id_assignment', 'idx_instruction']]
        self.indices = self.indices.astype({"idx_instruction": int}).astype({"idx_instruction": str})
        self.vision = self.vision.iloc[:, 2:].values
        self.target_str = self.target['transcript_str'].values
        self.target = self.target['transcript_tokens'].apply(literal_eval).values

        if self.load_noise:
            self.noise_list = []
            for id_noise in id_noise_list:
                self.noise_list.append(self.noise.loc[self.noise['id_noise'] == id_noise, 'idxs'].apply(literal_eval).values)

    def __len__(self) -> int:
        return len(self.indices) * self.n_sets

    def __getitem__(self, 
                    idx: int) -> Tuple[Tuple, Tuple]:
        index = int(idx / self.n_sets)
        idx_set = idx % self.n_sets
        f = '_'.join(self.indices.iloc[index]) + '.pt'

        x_audio = torch.load(os.path.join(self.dirs_audio[idx_set], f), map_location='cpu')
        vision = torch.tensor(self.vision[index], dtype=torch.float)
        target_str = self.target_str[index]
        target = torch.tensor(np.pad(self.target[index],
                                     (0, max(0, self.max_target_length - len(self.target[index]))),
                                     mode='constant',
                                     constant_values=self.pad_token))
        targ = pd.read_csv(os.path.join(self.dir_data, 'target.csv'))
        x = targ.f_json[index]
        parentdir = x.replace('/traj_data.json', '')
        with open(x, 'r') as f:
            traj = json.load(f)
            images = traj['images']
            f_image = [image for image in images if image['high_idx'] == targ.idx_instruction[index]][0]['image_name']
            f_image = os.path.join(parentdir, 'raw_images', f_image.replace('png', 'jpg'))
            #image = Image.open(f_image)

        if self.load_noise:
            if self.id_noise_list[idx_set] in ['clean', 'mix_clean']:
                noise = torch.tensor([], dtype=torch.long)
            else:
                noise = torch.tensor(self.noise_list[idx_set][index], dtype=torch.long)

            return (x_audio, vision), (target, target_str, noise), f_image
        else:
            return (x_audio, vision), (target, target_str), f_image