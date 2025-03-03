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
import argparse
from gpt_config import *

class PlayDataset(Dataset):
    def __init__(self,text_file_name):
        file = open(text_file_name,"r")
        lines = file.readlines()
        united_lines = "".join(lines)
        division_in_words = united_lines.split(" ")
        n = len(division_in_words)
        self.divided_corpus = []
        step_size = 128
        for i in tqdm(range(0,n,step_size)):
            self.divided_corpus.append(" ".join(division_in_words[i:i+step_size]))
        self.divided_corpus_size = len(self.divided_corpus)
    def __getitem__(self,idx):
        return "<|bos|>"+self.divided_corpus[idx]+"<|endoftext|>"
    def __len__(self):
        return self.divided_corpus_size



gpt_tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
special_tokens={
    "pad_token":"<|pad|>",
    "bos_token":"<|bos|>"
}
#gpt_tokenizer.pad_token = gpt_tokenizer.eos_token
added_tokens=gpt_tokenizer.add_special_tokens(special_tokens)

def subword_tokenizer(batch_texts):
    result_dict=gpt_tokenizer(batch_texts,padding='longest',return_tensors="pt")
    return result_dict["input_ids"]

if __name__=='__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--gpt_type",choices=["small","large"])
    args_list = argparser.parse_args()
    gpt_type=args_list.gpt_type
    param_dict = big_gpt if gpt_type=="large" else small_gpt
    trainData = PlayDataset("train.txt")
    testData = PlayDataset("test.txt")
    pl.seed_everything(42)
    trainLoader = DataLoader(trainData,batch_size=32,shuffle=False,collate_fn=subword_tokenizer)
    testLoader = DataLoader(testData,batch_size=32,shuffle=False,collate_fn=subword_tokenizer)
    num_epochs = 8
    steps_per_epoch = len(trainLoader)
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = 50
    aim_logger = AimLogger(experiment="NN_HW3_subword")
    model = GPTTrain(param_dict,len(gpt_tokenizer),gpt_tokenizer.pad_token_id,warmup_steps,total_steps)
    #csv_logger = pl.pytorch.loggers.csv_logs.CSVLogger("raw_SW_logs","raw_log")
    check_callback = pl.pytorch.callbacks.ModelCheckpoint(f"GPT_subword_{gpt_type}",filename="model_subword")
    trainer = pl.Trainer(logger=[aim_logger],callbacks=[check_callback],max_epochs=num_epochs)
    trainer.fit(model,trainLoader,testLoader)