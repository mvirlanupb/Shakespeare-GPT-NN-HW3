from gpt_components import GPTTrain
from transformers import GPT2Tokenizer
from train_characters import all_chars,char_to_idx,idx_to_char
import argparse
import torch
import lightning as pl
import numpy as np
from torchmetrics.functional.text import bleu_score,rouge
gpt_tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
special_tokens={
    "pad_token":"<|pad|>",
    "bos_token":"<|bos|>"
}
#gpt_tokenizer.pad_token = gpt_tokenizer.eos_token
added_tokens=gpt_tokenizer.add_special_tokens(special_tokens)

with open("test.txt") as f:
    test_samples = f.readlines()

chosen_samples = test_samples[:10]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def unroll_subword(model,sample,SUBWORD_CONTEXT=128):
    sample_words = sample.split(" ")
    start = "".join(sample_words[:10])
    start_tokens = gpt_tokenizer(start,return_tensors="pt")
    tokenized_seq = start_tokens["input_ids"]
    tokenized_seq = tokenized_seq.to(device)
    input_seq = torch.clone(tokenized_seq)
    model = model.to(device)
    model.eval()
    while True:
        out_logits = model(input_seq)#(1,seq_len,vocab_size)
        logit_last_token = out_logits[:,-1,:]#(1,vocab_size)
        out_probabilities = torch.softmax(logit_last_token,dim=1)
        predicted_token = torch.argmax(out_probabilities,dim=1)

        if(predicted_token.item()==gpt_tokenizer.eos_token_id):
            break
        else:
            new_addition = torch.tensor(predicted_token).to(device)
            new_addition = new_addition.reshape(1,-1)
            if(input_seq.shape[0]==SUBWORD_CONTEXT):
                input_seq = input_seq[:,1:]
            else:
                input_seq = torch.concatenate((input_seq,new_addition),dim=1)
            tokenized_seq = torch.concatenate((tokenized_seq,new_addition),dim=1)
        if(tokenized_seq.shape[1]==200):
            break
    tokenized_seq = tokenized_seq.squeeze(0)
    if(tokenized_seq[-1]==gpt_tokenizer.eos_token_id):
        tokenized_seq=tokenized_seq[:-1]
    tokenized_seq = tokenized_seq.cpu()
    tokenized_seq = tokenized_seq.tolist()
    return gpt_tokenizer.decode(tokenized_seq)

def unroll_char(model,sample,CHAR_CONTEXT=256):
    start = "".join(sample.split(" ")[:10])
    start = list(start)
    start = ["<BOS>"]+start
    tokenized_start = [char_to_idx[c] for c in start]
    tokenized_seq = torch.tensor(tokenized_start).to(device)
    tokenized_seq = tokenized_seq.unsqueeze(0)
    input_seq = torch.clone(tokenized_seq)
    model=model.to(device)
    model.eval()
    while True:
        out_logits = model(input_seq)#(1,seq_len,vocab_size)
        logit_last_token = out_logits[:,-1,:]#(1,vocab_size)
        out_probabilities = torch.softmax(logit_last_token,dim=1)
        predicted_token = torch.argmax(out_probabilities,dim=1)

        if(predicted_token.item()==char_to_idx["<EOS>"]):
            break
        else:
            new_addition = torch.tensor(predicted_token).to(device)
            new_addition = new_addition.reshape(1,-1)
            if(input_seq.shape[0]==CHAR_CONTEXT):
                input_seq = input_seq[:,1:]
            else:
                input_seq = torch.concatenate((input_seq,new_addition),dim=1)
            tokenized_seq = torch.concatenate((tokenized_seq,new_addition),dim=1)
    tokenized_seq = tokenized_seq.squeeze(0)
    if(tokenized_seq[-1]==char_to_idx["<EOS>"]):
        tokenized_seq = tokenized_seq[:-1]
    tokenized_seq = tokenized_seq.cpu()
    tokenized_seq = tokenized_seq.tolist()
    return "".join([idx_to_char[c] for c in tokenized_seq])



if __name__=="__main__":
    pl.seed_everything(42)
    np.random.seed(42)
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--gpt_mode",choices=["subword","char"])
    argument_parser.add_argument("--gpt_dim",choices=["large","small"])
    parsed_args = argument_parser.parse_args()
    gpt_mode = parsed_args.gpt_mode
    gpt_dim = parsed_args.gpt_dim
    if(gpt_mode=="subword"):
        if(gpt_dim=="large"):
            pytorch_lightning_model = GPTTrain.load_from_checkpoint("GPT_subword_large/model_subword.ckpt")
        else:
            pytorch_lightning_model = GPTTrain.load_from_checkpoint("GPT_subword_small/model_subword.ckpt")
        predictions,targets=[],[]
        for i,sample in enumerate(chosen_samples):
            continuation = unroll_subword(pytorch_lightning_model.gpt_model,sample)
            predictions.append(continuation)
            targets.append([sample])
        for n_gram in [1,2,3,4]:
            print(f"BLEU-{n_gram}={bleu_score(predictions,targets,n_gram=n_gram)}")
        print(rouge.rouge_score(predictions,targets))
    else:
        if(gpt_dim=="large"):
            pytorch_lightning_model = GPTTrain.load_from_checkpoint("GPT_char_large/model_chars.ckpt")
        else:
            pytorch_lightning_model = GPTTrain.load_from_checkpoint("GPT_char_small/model_chars.ckpt")
        #print(unroll_char(pytorch_lightning_model.gpt_model,chosen_samples[0],decoding_style="topk"))
        predictions,targets=[],[]
        for i,sample in enumerate(chosen_samples):
            continuation = unroll_char(pytorch_lightning_model.gpt_model,sample)
            predictions.append(continuation)
            targets.append([sample])
        for n_gram in [1,2,3,4]:
            print(f"BLEU-{n_gram}={bleu_score(predictions,targets,n_gram=n_gram)}")
        print(rouge.rouge_score(predictions,targets))
    