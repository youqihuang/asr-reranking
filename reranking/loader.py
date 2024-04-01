import os
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
from lib.out import bcolors
from lib.dataset import AVDataset
from lib.eval import WordErrorRate, RecoveryRate
from lib.reranker.rerankpipeline import RerankPipeline
import math

import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Subset

from lib.wav2vec2.bundles import WAV2VEC2_ASR_BASE_960H
import torch.nn.functional as F

from lib.models import TransformerPipeline, MultimodalTransformerPipeline
from sklearn.preprocessing import LabelEncoder

from lib.eval import WordErrorRate, RecoveryRate
from typing import Tuple, List

wer = WordErrorRate()

def init_word_to_token(dir_source: str) -> Tuple[LabelEncoder, int, int, int, int]:
    word_to_token = LabelEncoder()
    word_to_token.classes_ = np.load(f'{dir_source}/train/word_to_token.npy')

    BOS_token, EOS_token, PAD_token = word_to_token.transform(
        ['<BOS>', '<EOS>', '<PAD>'])
    n_tokens = len(word_to_token.classes_)

    return word_to_token, BOS_token, EOS_token, PAD_token, n_tokens

class RerankDataset(Dataset):
  
  def __init__(self, asrdataset, pipeline, scorer, args, pipeline_type):
      self.asrdataset = asrdataset
      self.pipeline = pipeline
      self.scorer = scorer
      self.args = args
      self.pipeline_type = pipeline_type
      self.word_to_token, self.BOS_token, self.EOS_token, self.PAD_token, self.n_tokens = init_word_to_token(self.args.dir_source)
      self.n_best = 5
      
  def __len__(self):
      return len(self.asrdataset)
  def __getitem__(self, idx):
      sub = Subset(self.asrdataset, [idx])
      load = DataLoader(sub, batch_size=1, shuffle=True)
      pred_strs = []
      probs = []
      targets_str = []
      noise_indices = []
      for item in iter(load):
        (feats, targets, image) = item
        x_audio = feats[0].to(self.args.device)
        x_vision = feats[1].unsqueeze(1).to(self.args.device)
        targets_str = targets[1][0]
        noise_indices.append(targets[2][0])
        memory = x_audio.repeat_interleave(self.args.top_k, dim=0)
        if self.pipeline_type == 'multimodal':
            x_vision = x_vision.repeat_interleave(self.args.top_k, dim=0)
        # Start decoder
        top_input = torch.tensor([[self.BOS_token]], device=self.args.device).repeat_interleave(self.args.top_k, dim=0)
        top_sequences = [([self.BOS_token], 0)]
        with torch.no_grad():
            for _ in range(self.args.max_target_len):
                tgt_mask = self.pipeline.module.get_tgt_mask(top_input.shape[1]).to(self.args.device)
                    # Standard training
                if self.pipeline_type == 'unimodal':
                    pred = self.pipeline(memory, top_input, tgt_mask)
                elif self.pipeline_type == 'multimodal':
                    pred = self.pipeline(memory, x_vision, top_input, tgt_mask)
                pred = F.log_softmax(pred[:, -1, :], dim=-1)
                new_sequences = []
                for i in range(len(top_sequences)):
                    old_seq, old_score = top_sequences[i]
                    p_word, tok_word = pred[i].topk(self.args.top_k)
                    for idx_word in range(self.args.top_k):
                        new_seq = old_seq + [tok_word[idx_word].item()]
                        new_score = old_score + p_word[idx_word].item()
                        new_sequences.append((new_seq, new_score))
                top_sequences = sorted(new_sequences, key=lambda val: val[1], reverse=True)
                top_sequences = top_sequences[:self.args.top_k]
                top_input = torch.tensor([seq[0] for seq in top_sequences], device=self.args.device)  
            pred_seqs = []
            pred_probs = []
            pred_strs = []   
            for i in range(self.args.top_k):       
                top_sequence = [token for token in top_sequences[i][0] if token not in [self.BOS_token, self.EOS_token, self.PAD_token]]
                pred_strs.append(" ".join(self.word_to_token.inverse_transform(top_sequence)))
                pred_probs.append(top_sequences[i][1])
            lowest_wer = 1.5
            lowest_idx = 0
            wers = []
            idcs = []
            lab = 0
            labels = [0 for m in range(len(pred_strs))]
            for k in range(len(pred_strs)):
              wers.append(wer(targets_str, pred_strs[k]))
              idcs.append(k)
            wers_copy = [wers[i] for i in range(len(wers))]
            while len(wers)>0:
              min_idx = wers.index(min(wers))
              actual_idx = idcs[min_idx]
              labels[actual_idx] = lab
              wers.pop(min_idx)
              idcs.pop(min_idx)
              lab+=1
            assert len(labels) == len(pred_strs)
            n_best_hyps = [(pred_strs[k], pred_probs[k]) for k in range(self.n_best)]
            scores = self.scorer(n_best_hyps, feats[1].unsqueeze(1), labels)
            return (n_best_hyps, labels, targets_str, wers_copy, image) 
      
              
class GPT2_dataset(Dataset):
  def __init__(self, asrdataset):
      self.asrdataset = asrdataset
  def __len__(self):
      return len(self.asrdataset)
  def __getitem__(self, idx):
      sub = Subset(self.asrdataset, [idx])
      load = DataLoader(self.asrdataset, batch_size=1, shuffle=True)
      for id_batch, (feats, targets) in tqdm(enumerate(load),
                                           position=0,
                                           leave=True):
          targets_str = targets[1][0]
      return targets_str
        
               
        
      
     