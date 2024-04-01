import torch
import huggingface_hub
from transformers import BertTokenizer, BertModel
import clip
from PIL import Image
import random
import string
import nltk 
import torch
from transformers import OpenAIGPTTokenizer, OpenAIGPTLMHeadModel
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np
import math


class RerankPipeline(torch.nn.Module):
    def __init__(self, n_best, train):
        super(RerankPipeline,self).__init__()
        self.n_best = n_best
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.CLIP_model, self.CLIP_preprocess = clip.load("ViT-B/32", device=self.device)
        self.GPT2Tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.weight_matrix = [0, 1, 2, 5, 10]
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=0)
        
        self.train = train
    
    def process_nbest(self, n_best_list) -> list:
        for hypothesis in n_best_list:
            processed_hypotheses = []
            hypothesis = hypothesis.strip().lower()
            hypothesis = hypothesis.translate(str.maketrans('', ''. string.punctuation))
            processed_hypotheses.append(hypothesis)
        return processed_hypotheses
    
    def compute_sentence_probabilities(self, n_best_list):
        probabilities = []
        for text in n_best_list:
            input_ids = torch.tensor(self.GPT2Tokenizer.encode(text)).unsqueeze(0)
            
            with torch.no_grad():
                loss = self.model(input_ids, labels=input_ids)[0].item()
                pval = np.exp(-1*loss)
            probabilities.append(pval)
        
        probabilities = torch.tensor(probabilities)
        probabilities = torch.nn.functional.log_softmax(probabilities, dim=0)
        return probabilities.tolist()
    
    def compute_CLIP_scores(self, n_best_list, image_features):
        CLIP_scores = []
        image_features = torch.squeeze(image_features, dim=0)
        image_features = torch.squeeze(image_features, dim=0)
           
        for text in n_best_list:
            with torch.no_grad():
                tokenized_text = clip.tokenize(text)
                text_features = self.CLIP_model.encode_text(tokenized_text).squeeze()
                text_features /= text_features.norm(dim=-1, keepdim=True)
                
                similarity = self.cosine_similarity(image_features, text_features.T)
                
                CLIP_scores.append(similarity)
        #print(CLIP_scores)
        CLIP_scores = torch.Tensor(CLIP_scores)
        
        CLIP_scores = torch.nn.functional.log_softmax(CLIP_scores, dim=0)
        return CLIP_scores.tolist()
    
    def forward(self, n_best_hypotheses, image, labels = []):
        n_best_list = [_[0] for _ in n_best_hypotheses]
        acoustic_scores = [_[1] for _ in n_best_hypotheses]
        #print(acoustic_scores)
        acoustic_scores = np.exp(acoustic_scores)
        
        acoustic_scores = torch.FloatTensor(acoustic_scores)
        
        acoustic_scores = torch.nn.functional.log_softmax(acoustic_scores, dim=0)
        language_scores = self.compute_sentence_probabilities(n_best_list)
        CLIP_scores = self.compute_CLIP_scores(n_best_list, image)
        if self.train:
            n_best_list_with_scores_and_labels = [(n_best_list[_], acoustic_scores[_], language_scores[_], CLIP_scores[_], labels[_]) for _ in range(self.n_best)]
            random.shuffle(n_best_list_with_scores_and_labels)
            n_best_list = [n_b[0] for n_b in n_best_list_with_scores_and_labels]
            acoustic_scores = [n_b[1] for n_b in n_best_list_with_scores_and_labels]
            language_scores = [n_b[2] for n_b in n_best_list_with_scores_and_labels]
            CLIP_scores = [n_b[3] for n_b in n_best_list_with_scores_and_labels]
            labels = [n_b[4] for n_b in n_best_list_with_scores_and_labels]
        
        
        #print(CLIP_scores)
        h_scores = torch.unsqueeze(torch.tensor([acoustic_scores[0], language_scores[0], CLIP_scores[0]]), 0)
        for i in range(1,5):
            h_scores = torch.cat((h_scores, torch.unsqueeze(torch.tensor([acoustic_scores[i], language_scores[i], CLIP_scores[i]]),0)), 1)
        #return h_scores
        
        
        
        return torch.squeeze(h_scores), torch.tensor(labels)
        #preds = []
        
        """
        for i in self.weight_matrix:
            for j in self.weight_matrix:
                for k in self.weight_matrix:
                    
                    weighted_sum = (i*acoustic_scores + j*language_scores + k*CLIP_scores)
                    top_index = torch.argmax(weighted_sum)
                    preds.append(n_best_list[top_index])
        return preds
        """
class RerankLayer(torch.nn.Module):
    def __init__(self, n_best):
        super(RerankLayer,self).__init__()
        self.n_best = n_best
        self.lin = torch.nn.Linear(15, 30)
        self.relu = torch.nn.ReLU()
        self.lin2 = torch.nn.Linear(30, 15)
        self.lin3 = torch.nn.Linear(15, 5)

    def forward(self, input):
        
        x = self.lin(input)
        x = self.relu(x)
        x = self.lin2(x)
        x = self.relu(x)
        x = self.lin3(x)
        return torch.FloatTensor(x)
        
        
    