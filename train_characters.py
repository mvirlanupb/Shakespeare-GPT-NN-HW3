from torch.utils.data import Dataset,DataLoader
from transformers import GPT2Tokenizer
import pandas as pd
import torch
import lightning as pl
from gpt_components import GPTTrain,GPT_HW3,device,Perplexity
from aim.pytorch_lightning import AimLogger
from aim import Run
from tqdm import tqdm
import torch.nn as nn
from gpt_config import *
import argparse

class PlayDataset(Dataset):
    def __init__(self,text_file_name):
        file = open(text_file_name,"r")
        lines = file.readlines()
        united_lines = "".join(lines)
        n = len(united_lines)
        self.divided_corpus = []
        step_size = 64
        for i in tqdm(range(0,n,step_size-2)):
            self.divided_corpus.append(" ".join(united_lines[i:i+step_size-2]))
        self.divided_corpus_size = len(self.divided_corpus)
        file.close()
    def __getitem__(self,idx):
        return self.divided_corpus[idx]
    def __len__(self):
        return self.divided_corpus_size


with open("full.txt","r") as f:
    all_lines = f.readlines()
    united_lines="".join(all_lines)
    all_chars = set(united_lines)
all_chars=sorted(all_chars)
char_to_idx={
    "<PAD>":0,
    "<BOS>":1,
    "<EOS>":2
}
char_to_idx.update({c:3+i for i,c in enumerate(all_chars)})
idx_to_char={v:k for k,v in char_to_idx.items()}
#,bos_token,eos_token,pad_token
def char_tokenizer(batch_texts):
    #batch_texts is a list of texts
    splitted_texts=[]
    tokenized_texts=[]
    max_seq_len=0
    for text in batch_texts:
        #we take each text and split it into chars
        text_split_chars = list(text)
        new_split = ["<BOS>"]+text_split_chars+["<EOS>"]
        max_seq_len=max(len(new_split),max_seq_len)
        splitted_texts.append(new_split)
    #now, splitted texts is a list where each element is a list of characters
    for text in splitted_texts:
        padded_text=text+["<PAD>"]*(max_seq_len-len(text))
        tokenized_texts.append([char_to_idx[c] for c in padded_text])
    return torch.Tensor(tokenized_texts).to(torch.long)

if __name__=='__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--gpt_type",choices=["small","large"])
    args_list = argparser.parse_args()
    gpt_type=args_list.gpt_type
    param_dict = big_gpt if gpt_type=="large" else small_gpt
    trainData = PlayDataset("train.txt")
    testData = PlayDataset("test.txt")
    pl.seed_everything(42)
    trainLoader = DataLoader(trainData,batch_size=32,shuffle=False,collate_fn=char_tokenizer)
    testLoader = DataLoader(testData,batch_size=32,shuffle=False,collate_fn=char_tokenizer)
    num_epochs = 8
    steps_per_epoch = len(trainLoader)
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = 500
    aim_logger = AimLogger(experiment="NN_HW3_chars")
    
    model = GPTTrain(param_dict,len(all_chars)+3,char_to_idx["<PAD>"],warmup_steps,total_steps)
    aim_logger = AimLogger(experiment="NN_HW3_chars")
    #csv_logger = pl.pytorch.loggers.csv_logs.CSVLogger("raw_SW_logs","raw_log")
    check_callback = pl.pytorch.callbacks.ModelCheckpoint(f"GPT_char_{gpt_type}",filename="model_chars")
    trainer = pl.Trainer(logger=[aim_logger],callbacks=[check_callback],max_epochs=num_epochs)
    trainer.fit(model,trainLoader,testLoader)