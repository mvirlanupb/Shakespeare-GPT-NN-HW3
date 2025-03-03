import torch.nn as nn
import torch
import math
import lightning as pl
from torchmetrics.text import Perplexity

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16
class OneLayerAttention(nn.Module):
    def __init__(self,QK_dim,V_dim,d_model):
        #X has the dims (N_seq,seq_len,d_model)
        super().__init__()
        self.W_q = nn.Parameter(torch.randn(d_model,QK_dim,dtype=dtype,device=device),requires_grad=True)#(d_model,QK_dim)
        nn.init.xavier_normal_(self.W_q)
        self.W_k = nn.Parameter(torch.randn(d_model,QK_dim,dtype=dtype,device=device),requires_grad=True)
        nn.init.xavier_normal_(self.W_k)
        self.W_v = nn.Parameter(torch.randn(d_model,V_dim,dtype=dtype,device=device),requires_grad=True)
        nn.init.xavier_normal_(self.W_v)
        self.QK_dim=QK_dim
        self.V_dim=V_dim
    def forward(self,X):
        
        Q = torch.matmul(X,self.W_q)#(N_seq,seq_len,QK_dim)
        K = torch.matmul(X,self.W_k)#(N_seq,seq_len,QK_dim)
        V = torch.matmul(X,self.W_v)#(N_seq,seq_len,V_dim)
        N = X.shape[0]
        seq_len = X.shape[1]
        output = torch.matmul(Q,K.permute((0,2,1)))/math.sqrt(self.QK_dim)
        index_upper = torch.triu_indices(seq_len,seq_len,offset=1)
        output[:,index_upper[0],index_upper[1]]=float('-inf')
        softmax_weights = torch.softmax(output,dim=2,dtype=dtype)
        #softmax_weights = self.attn_drop(softmax_weights)
        #(N,seq_len,seq_len) x (seq_len,V_dim) => (N,seq_len,V_dim)
        return torch.matmul(softmax_weights,V)

class MultiHeadAttention(nn.Module):
    def __init__(self,no_attn_heads=8,d_model=512):
        super().__init__()
        d_k = d_model//no_attn_heads
        d_v = d_k
        self.no_attn_heads=no_attn_heads
        self.attn_heads = [OneLayerAttention(d_k,d_v,d_model) for _ in range(no_attn_heads)]
        self.W_o = nn.Parameter(torch.randn(no_attn_heads*d_v,d_model,dtype=dtype,device=device),requires_grad=True)
        nn.init.xavier_normal_(self.W_o)
    def forward(self,X):
        attn_results=[head(X) for head in self.attn_heads]
        #no_attn_heads times (N,seq_len,V_dim_head)
        stacked_results = torch.stack(attn_results,dim=1) #(N,no_attn_heads,seq_len,V_dim_head)
        stacked_results = torch.permute(stacked_results,(0,2,3,1))#(N,seq_len,no_attn_heads,V_dim_head)
        stacked_results = torch.flatten(stacked_results,start_dim=2) #(N,seq_len,V_dim)
        return torch.matmul(stacked_results,self.W_o)#(N,seq_len,d_model)

class LayerNorm(nn.Module):
    def __init__(self,d_model):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones((1,1,d_model),dtype=dtype,device=device),requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1,1,d_model),dtype=dtype,device=device),requires_grad=True)
    def forward(self,X): #X has dim (N,seq_len,d_model)
        mean_X = torch.mean(X,dim=2).unsqueeze(2)
        std_X = torch.std(X,dim=2).unsqueeze(2)+1e-10
        normalized_X=(X-mean_X)/std_X
        return normalized_X*self.gamma+self.beta

class InputEmbedding(nn.Module):
    def __init__(self,emb_size,vocab_size,pad_id):
        super().__init__()
        self.emb_layer = nn.Embedding(num_embeddings=vocab_size,embedding_dim=emb_size,padding_idx=pad_id,dtype=dtype)
        #nn.init.xavier_normal_(self.emb_layer.weight)
    def forward(self,X):
        #X has size (N,seq_len)
        Y = self.emb_layer(X) # Y has size (N,seq_len,emb_size)
        PE_matrix = torch.zeros_like(Y,dtype=dtype)
        seq_index = torch.arange(Y.shape[1],device=device).reshape(-1,1)
        emb_index = torch.arange(Y.shape[2],device=device)
        even_index = emb_index[emb_index%2==0]
        odd_index = emb_index[emb_index%2==1]
        denominator = 10000**(emb_index[emb_index%2==0]/Y.shape[2])
        args_sin_cos = seq_index/denominator
        sin_result = torch.sin(args_sin_cos)
        cos_result = torch.cos(args_sin_cos)
        PE_matrix[:,:,even_index] = sin_result.to(dtype=dtype)
        PE_matrix[:,:,odd_index] = cos_result.to(dtype=dtype)
        PE_matrix = PE_matrix.to(device)
        return Y+PE_matrix


class FFN(nn.Module):
    def __init__(self,d_model,d_ff,act_fn=nn.GELU()):
        super().__init__()
        self.ln1 = nn.Linear(d_model,d_ff,dtype=dtype)
        self.ln2 = nn.Linear(d_ff,d_model,dtype=dtype)
        self.drop = nn.Dropout(p=0.01)
        self.act_fn = act_fn
        nn.init.xavier_normal_(self.ln1.weight)
        nn.init.xavier_normal_(self.ln2.weight)
        nn.init.zeros_(self.ln1.bias)
        nn.init.zeros_(self.ln2.bias)
    def forward(self,X):
        #X has (B,N,seq_len,d_model)
        Y = self.ln1(X)
        Y = self.drop(Y)
        Y = self.act_fn(Y)
        return self.ln2(Y)

class TransformerLayer(nn.Module):
    def __init__(self,no_attn_heads=8,d_model=512):
        super().__init__()
        self.multi_head = MultiHeadAttention(no_attn_heads,d_model)
        self.layer_norm1 = LayerNorm(d_model)
        self.layer_norm2 = LayerNorm(d_model)
        self.ffn = FFN(d_model,2048)
        self.drop = nn.Dropout(p=0.01)
    def forward(self,X):
        #PRE-LN FLOW
        #STEP 1 Layer Normalization
        y = self.layer_norm1(X)
        y = self.drop(y)
        #STEP 2 y goes into multi_head attention
        after_attn = self.multi_head(y)
        #STEP 3 first addition
        first_addition = X+after_attn
        #STEP 4 layer normalize the first_addition result
        first_addition = self.drop(first_addition)
        tmp = self.layer_norm2(first_addition)
        tmp = self.drop(tmp)
        #STEP 5 pass tmp through the FFN
        tmp = self.ffn(tmp)
        #STEP 6 We add the tmp to the first addition and return
        return self.drop(first_addition+tmp)

class GPT_HW3(nn.Module):
    def __init__(self,no_attn_heads,number_transf_layers,d_model,vocab_size,pad_id):
        super().__init__()
        self.input_embedding = InputEmbedding(d_model,vocab_size,pad_id)
        self.transf_layers=nn.ModuleList([TransformerLayer(no_attn_heads=no_attn_heads,d_model=d_model) for _ in range(number_transf_layers)])
        self.pre_softmax = nn.Linear(d_model,vocab_size,dtype=dtype)
        nn.init.xavier_normal_(self.pre_softmax.weight)
        nn.init.zeros_(self.pre_softmax.bias)
    def forward(self,X):
        #this time X has size (N,seq_len)
        y = self.input_embedding(X)
        for layer in self.transf_layers:
            y=layer(y)
        return self.pre_softmax(y)

GEN_CONTEXT=512
class GPTTrain(pl.LightningModule):
    def __init__(self,param_dict,vocab_size,pad_id,warmup_steps,total_steps):
        super().__init__()
        self.save_hyperparameters()
        no_attn_heads=param_dict["num_attn_heads"]
        number_transf_layers=param_dict["num_transformers_layers"]
        d_model = param_dict["emb_dim"]
        self.gpt_model = GPT_HW3(no_attn_heads,number_transf_layers,d_model,vocab_size,pad_id)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)
        self.warmup_steps=warmup_steps
        self.total_steps = total_steps
        self.automatic_optimization=False
        self.perplex_score = Perplexity(ignore_index=pad_id)
    def compute_loss(self,batch):
        input_tokens = batch[:,:-1]
        targets = batch[:,1:]
        out_logits = self.gpt_model(input_tokens)#(N,seq_size,vocab_size)
        loss_value = self.loss_fn(out_logits.permute((0,2,1)),targets)
        return loss_value,out_logits,targets

    def training_step(self,batch,batch_idx):
        opt = self.optimizers()
        sch = self.lr_schedulers()
        opt.zero_grad()
        loss_value,_,_=self.compute_loss(batch)
        self.log("Train_loss",loss_value,on_step=False,on_epoch=True,logger=True)
        self.manual_backward(loss_value)
        opt.step()
        sch.step()
        return loss_value
    def validation_step(self,batch,batch_idx):
        loss_value,out_logits,targets = self.compute_loss(batch)
        perplexity_value = self.perplex_score(out_logits,targets)
        self.log("Validation_loss",loss_value,on_step=False,on_epoch=True,logger=True)
        self.log("Validation_Perplexity",perplexity_value,on_step=False,on_epoch=True,logger=True)
        return perplexity_value
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),lr=1e-3)
        lin_sched=torch.optim.lr_scheduler.LinearLR(optimizer,1e-5,1,total_iters=self.warmup_steps)
        aux_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=self.total_steps-self.warmup_steps)
        lr_sched=torch.optim.lr_scheduler.SequentialLR(optimizer,
                                                       [lin_sched,aux_sched],
                                                       milestones=[self.warmup_steps])
        
        return [optimizer],[lr_sched]