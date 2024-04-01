import os
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
from lib.out import bcolors
from lib.dataset import AVDataset, ImageAVDataset
from lib.eval import WordErrorRate, RecoveryRate
from lib.reranker.rerankpipeline import RerankPipeline, RerankLayer

import math

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from lib.wav2vec2.bundles import WAV2VEC2_ASR_BASE_960H
import torch.nn.functional as F
from lib.reranker.loader import RerankDataset, GPT2_dataset
from lib.models import TransformerPipeline, MultimodalTransformerPipeline
from main import init_word_to_token
from transformers import GPT2Tokenizer, GPT2Model, AutoModelWithLMHead, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import csv

from PIL import Image



def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    # Data Extraction Parameters
    parser.add_argument('--run', type=str,
                        help='ID of the run to load weights from.')
    parser.add_argument('--id_noise', type=str, nargs="*",
                        help='Noise id. Choices are: ["clean", "mask_{rho}", "swap_{rho}"].')
    parser.add_argument('--eval_set', type=str, default='valid',
                        help='Evaluation set to use.')
    parser.add_argument('--train_set', type=str, default='train',
                        help='Evaluation set to use.')
    
    parser.add_argument('--top_k', type=int, default=5,
                        help='Beam size for beam search decoder.')
    parser.add_argument('--out', type=str, default='models/results.csv',
                        help='Output file path.')
    parser.add_argument('--dir_source', type=str, default="data",
                        help='Name of directory to load the data from.')

    # Pipeline Parameters
    parser.add_argument('--d_audio', type=int, default=[312, 768], nargs=2,
                        help='Dimension of the audio embedding.')
    parser.add_argument('--d_vision', type=int, default=512,
                        help='Dimension of the vision embedding.')
    parser.add_argument('--d_obj_count', type=int, default=59,
                        help='Dimension of the object counts (the number of objects possible).')
    parser.add_argument('--max_target_len', type=int, default=25,
                        help='Maximum sequence length of a target transcript.')
    parser.add_argument('--depth', type=int, default=4,
                        help='Depth of the TransformerDecoder')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout of the TransformerDecoderLayer')

    # Torch Parameters
    parser.add_argument('--device', type=str, default=("cuda" if torch.cuda.is_available() else "cpu"),
                        help='Device to use for torch.')

    args = parser.parse_args()

    # Assertions, to make sure params are valid.
    assert len(args.d_audio) == 2, "d_audio must have length 2"

    assert args.run == 'baseline' or os.path.exists(f"models/{args.run}.pt"), f"run '{args.run}' not found"
    assert len(args.id_noise) > 0, "id_noise must have at least one value"
    for id_noise in args.id_noise:
        assert os.path.isdir(f"{args.dir_source}/train/audio/{id_noise}"), f"id_noise '{id_noise}' not found"

    return args

def init_pipeline(device: str):
    bundle = WAV2VEC2_ASR_BASE_960H
    if args.run[:8] == 'unimodal':
        pipeline = 'unimodal'
    else:
        pipeline = 'multi[clip]'
    if pipeline == 'unimodal':
        pipeline = nn.DataParallel(TransformerPipeline(
            args.d_audio, n_tokens, args.depth, args.max_target_len, args.dropout))
    elif pipeline in ['multi[resnet]', 'multi[clip]']:
        pipeline = nn.DataParallel(MultimodalTransformerPipeline(
            args.d_audio, args.d_vision, n_tokens, args.depth, args.max_target_len, args.dropout))
    elif args.pipeline == 'multi[obj_count]':
        pipeline = nn.DataParallel(MultimodalTransformerPipeline(
            args.d_audio, args.d_obj_count, n_tokens, args.depth, args.max_target_len, args.dropout))
    else:
        assert False, f"Pipeline {args.pipeline} not implemented."

    pipeline.module.load_state_dict(torch.load(f'models/{args.run}.pt', map_location=torch.device('cpu')))
    pipeline.to(device)
    return pipeline




def init_dataloader(dataset):
    aux_modality = "clip"
    if args.pipeline == 'multi[clip]':
        aux_modality = "clip"
    elif args.pipeline == "multi[obj_count]":
        aux_modality = "obj_count"
    dataset = AVDataset(f'{args.dir_source}/{dataset}',
                        list(args.id_noise),
                        aux_modality=aux_modality,
                        pad_token=PAD_token,
                        max_target_length=args.max_target_len,
                        load_noise=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    return dataset, loader

def init_dataloader_image(dataset):
    aux_modality = "clip"
    if args.pipeline == 'multi[clip]':
        aux_modality = "clip"
    elif args.pipeline == "multi[obj_count]":
        aux_modality = "obj_count"
    dataset = ImageAVDataset(f'{args.dir_source}/{dataset}',
                        list(args.id_noise),
                        aux_modality=aux_modality,
                        pad_token=PAD_token,
                        max_target_length=args.max_target_len,
                        load_noise=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    return dataset, loader


def get_scores(loader, pipeline, rerankpipeline):
    n_batches = len(loader)
    pipeline.eval()
    action_types = ["move", "put", "place", "turn", "go", "pick", "take", "get", "grab"]
    return_dict = {
        "move": 0,
        "put": 0,
        "place": 0,
        "turn": 0,
        "go": 0,
        "pick": 0,
        "take": 0,
        "get": 0,
        "grab": 0
    }
    count_dict = {
        "move": 0,
        "put": 0,
        "place": 0,
        "turn": 0,
        "go": 0,
        "pick": 0,
        "take": 0,
        "get": 0,
        "grab": 0
    }
    pred_strs = []
    probs = []
    targets_str = []
    noise_indices = []
    
    count = 0

    for id_batch, (feats, targets) in tqdm(enumerate(loader),
                                           position=0,
                                           total=n_batches,
                                           leave=True):
        
        
        x_audio = feats[0].to(args.device)
        
        x_vision = feats[1].unsqueeze(1).to(args.device)
        
        targets_str = targets[1][0]
        noise_indices = [targets[2][0]]

        memory = x_audio.repeat_interleave(args.top_k, dim=0)
        if pipeline_type == 'multimodal':
            x_vision = x_vision.repeat_interleave(args.top_k, dim=0)
        # Start decoder
        top_input = torch.tensor([[BOS_token]], device=args.device).repeat_interleave(args.top_k, dim=0)
        top_sequences = [([BOS_token], 0)]

        with torch.no_grad():
            for _ in range(args.max_target_len):
                tgt_mask = pipeline.module.get_tgt_mask(top_input.shape[1]).to(args.device)
                    # Standard training
                if pipeline_type == 'unimodal':
                    pred = pipeline(memory, top_input, tgt_mask)
                elif pipeline_type == 'multimodal':
                    pred = pipeline(memory, x_vision, top_input, tgt_mask)
                pred = F.log_softmax(pred[:, -1, :], dim=-1)


                new_sequences = []


                for i in range(len(top_sequences)):
                    old_seq, old_score = top_sequences[i]
                    p_word, tok_word = pred[i].topk(args.top_k)
                    for idx_word in range(args.top_k):
                        new_seq = old_seq + [tok_word[idx_word].item()]
                        new_score = old_score + p_word[idx_word].item()
                        new_sequences.append((new_seq, new_score))
                
                top_sequences = sorted(new_sequences, key=lambda val: val[1], reverse=True)
                top_sequences = top_sequences[:args.top_k]
                top_input = torch.tensor([seq[0] for seq in top_sequences], device=args.device)    
            
            
            pred_seqs = []
            pred_probs = []
            pred_strs = []
            
                
            for i in range(args.top_k):       
                top_sequence = [token for token in top_sequences[i][0] if token not in [BOS_token, EOS_token, PAD_token]]
                pred_strs.append(" ".join(word_to_token.inverse_transform(top_sequence)))
                pred_probs.append(top_sequences[i][1])
            
            top_rr = rr(pred_strs[0], targets_str, noise_indices)
            
            toadd = 0
            for i in range(1,5):
                topk_rr = rr(pred_strs[i], targets_str, noise_indices)
                
                if topk_rr > top_rr:
                    
                    toadd = 1
                    break
            first_word = ""
            for word in targets_str.lower().split():
                
                if word in action_types:
                    
                    return_dict[word] += toadd
                    count_dict[word] += 1
                    break
            count += 1
            if count%20 == 0:
                for word in action_types:
                    if count_dict[word] != 0:
                        print(word, return_dict[word]/count_dict[word])
                        print(count_dict[word])

def train_reranker(asrtraindataset, asrtestdataset, pipeline, scorer, rerankmodel, pipeline_type, args, num_epochs):
    count = 0
    celoss = torch.nn.CrossEntropyLoss()
    running_loss = 0
    rerankdataset = RerankDataset(asrtraindataset, pipeline, scorer, args,  pipeline_type)
    rrloader = DataLoader(rerankdataset, batch_size = 10, shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(rerankmodel.parameters())
    for i in range(num_epochs):
        intra_count = 0
        running_loss = 0
        
        for vals, labs in iter(rrloader):
            
            output = rerankmodel(vals)
            print(output)
            loss = loss_fn(output, labs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            intra_count+=1
            if intra_count%20 == 0:
                print("epoch {}, batch {} avg loss {}".format(i, intra_count, running_loss/20))
                running_loss = 0
            
            
                        
    

def finetune_gpt2(asrtraindataset, asrtestdataset, pipeline):
    
    
    pipeline.eval()
    targets_str = []
    noise_indices = []
    
    train_dataset = GPT2_dataset(asrtraindataset)
    valid_dataset = GPT2_dataset(asrtestdataset)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = AutoModelWithLMHead.from_pretrained('gpt2')
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )
    training_args = TrainingArguments(
    output_dir="./gpt2-gerchef", #The output directory
    overwrite_output_dir=True, #overwrite the content of the output directory
    num_train_epochs=3, # number of training epochs
    per_device_train_batch_size=32, # batch size for training
    per_device_eval_batch_size=64,  # batch size for evaluation
    evaluation_strategy="steps",
    eval_steps = 100,
    warmup_steps=500,# number of warmup steps for learning rate scheduler
    prediction_loss_only=True,
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer = tokenizer,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
    )
    
    trainer.train()

if __name__ == "__main__":
    args = parse_args()
    # Determine unimodal/multimodal
    if args.run == 'baseline' or args.run[:8] == 'unimodal':
        args.pipeline = 'unimodal'
        pipeline_type = 'unimodal'
    else:
        pipeline_type = 'multimodal'
        args.pipeline = args.run[:args.run.find('_[')]

        if args.pipeline not in ['multi[resnet]', 'multi[clip]', 'multi[obj_count]']:
            assert False, f"Pipeline {args.pipeline} not found."

    # Print device
    print(f"> Using device: {bcolors.OKGREEN}{args.device}{bcolors.ENDC}")
    print(f"> Testing run {bcolors.OKCYAN}{args.run}{bcolors.ENDC}")

    # Initialize pipeline and logger
    word_to_token, BOS_token, EOS_token, PAD_token, n_tokens = init_word_to_token(args.dir_source)
    pipeline = init_pipeline(args.device)
    asrtraindataset, _ = init_dataloader_image(args.train_set)
    #asrtestdataset, asrtestloader = init_dataloader(args.eval_set)
    scorer = RerankPipeline(5, train=False) 
    rerankdataset = RerankDataset(asrtraindataset, pipeline, scorer, args, pipeline_type)
    rrloader = DataLoader(rerankdataset, batch_size = 1, shuffle=True)
    filepath = os.path.join(os.getcwd(), 'reranking/analysis_mask_1.0_nouns.csv')
    
    try:
        os.remove(filepath)
    except OSError:
        pass
    count = 0
    with open(filepath, 'w+') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["target", "Hypotheses", "WERs", "Ranks"])
        for item in iter(rrloader):
            (n_best_hyps, labels, target, wers, img_path) = item
            if labels[0] == 1: continue
            if count > 25: break
            writer.writerow([target])
            for i in range(args.top_k):
                row = ["", n_best_hyps[i][0], wers[i].item(), labels[i].item()]
                writer.writerow(row)
            print(img_path)
            img = Image.open(img_path[0][0])
            img.save(os.path.join(os.getcwd(), "analysis_mask_1.0_nouns_images/image_{}.jpg".format(count)))
            count += 1
    """
    asrtraindataset, _ = init_dataloader(args.train_set)
    asrtestdataset, asrtestloader = init_dataloader(args.eval_set)
    scorer = RerankPipeline(5, train=True)
    wer = WordErrorRate()
    rr = RecoveryRate()
    rrmodel = RerankLayer(5)
    #get_scores(asrtestloader, pipeline, scorer)
    train_reranker(asrtraindataset, asrtestdataset, pipeline, scorer, rrmodel, pipeline_type, args, 1)
    
    """