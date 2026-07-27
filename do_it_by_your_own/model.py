import torch
import torch.nn as nn
import torch.nn.functional as F

class MinimindModel(nn.Module):
    def __init__(self,vocab_size,max_seq_len,d_model,dropout,
                 nhead,dim_feedforward,num_layers):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size,d_model)
        self.position_embedding = nn.Embedding(max_seq_len,d_model)
        self.dropout = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,nhead=nhead,
            dim_feedforward=dim_feedforward,dropout=dropout,batch_first=True,norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer,num_layers)
        self.norm = nn.LayerNorm(d_model)
    def forward(self,input_ids):
        B,T = input_ids.shape
        x = self.embeddings(input_ids)
        x = self.dropout(x)
        positions = torch.arange(T, device=x.device)
        positions = self.position_embedding(positions)
        x = x + positions
        causial_mask = nn.Transformer.generate_square_subsequent_mask(T,x.device)
        x = self.transformer(x,causial_mask)
        x = self.norm(x)
        return x

class MinimindForCausalLM(nn.Module):
    def __init__(self,vocab_size,max_seq_len,d_model,dropout,
                 nhead,dim_feedforward,num_layers):
        super().__init__()
        self.model = MinimindModel(vocab_size,max_seq_len,d_model,dropout,
                 nhead,dim_feedforward,num_layers)
        self.lm_head = nn.Linear(d_model,vocab_size)
        self.lm_head.weight = self.model.embeddings.weight
    def forward(self,input_ids,labels=None):
        x = self.model(input_ids)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            shift_logits = logits[:,:-1,:]
            shift_lables = labels[:,1:]
            shift_logits = shift_logits.reshape(-1,shift_logits.shape[-1])
            shift_lables = shift_lables.reshape(-1)
            loss = F.cross_entropy(shift_logits,shift_lables)
        return logits,loss
    def generate(self,batch_size,device,max_new_tokens,input_ids,repetition_penalty,temperature,topk,top_p,do_sample,eos_token_id):
        with torch.no_grad():
            finished = torch.zeros(batch_size,dtype=bool,device=device)
            for _ in range(max_new_tokens):
                logits,loss = self(input_ids,labels=None)
                logits = logits[:,-1,:]
                repeated_scores = torch.gather(logits,dim=1,index=input_ids)
                repeated_scores = torch.where(repeated_scores<0,repeated_scores*repetition_penalty,repeated_scores/repetition_penalty)
                logits.scatter_(dim=1,index=input_ids,src=repeated_scores)
                logits = logits / temperature
                topk_values,topk_indices = torch.topk(logits,k=topk,dim=1)
                filtered_logits = torch.full_like(logits,float("-inf"))
                filtered_logits.scatter_(dim=1,index=topk_indices,src=topk_values)
                logits = filtered_logits
                sorted_logits,sorted_indices = torch.sort(logits,dim=1,descending=True)
                sorted_probs = torch.softmax(sorted_logits,dim=1)
                cumulative_probs = torch.cumsum(sorted_probs,dim=1)
                sorted_indices_to_remove = (cumulative_probs > top_p)
                sorted_indices_to_remove[...,1:] = sorted_indices_to_remove[...,:-1].clone()
                sorted_indices_to_remove[...,0] = False
                sorted_logits = sorted_logits.masked_fill(sorted_indices_to_remove,float("-inf"))
                logits.scatter_(dim=1,index=sorted_indices,src=sorted_logits)
                probs = torch.softmax(logits,dim=1)
                if do_sample:
                    next_token = torch.multinomial(probs,num_samples=1)
                else:
                    next_token = torch.argmax(probs,dim=1,keepdim=True)
                next_token = torch.where(finished.unsqueeze(1),torch.full_like(next_token,eos_token_id),next_token)
                input_ids = torch.cat([input_ids,next_token],dim=1)
                next_token = next_token.squeeze(-1)
                is_eos = torch.where(next_token==eos_token_id,True,False)
                finished = finished|is_eos
                if finished.all():
                    break
            return input_ids



# Minimind = MinimindForCausalLM(100,100,
#         6,0.1,3,64,1)
# input_ids = torch.randint(0,100,(2,8))
# labels = torch.randint(0,1,(2,8))
# optimizer = torch.optim.AdamW(Minimind.parameters(),lr=0.001)
# optimizer.zero_grad()
# logits,loss = Minimind(input_ids,labels)
# loss.backward()
# optimizer.step()

# print(Minimind.lm_head.weight.grad is not None)
# print(Minimind.lm_head.weight.grad.shape)
# print(f"logits_shape={logits.shape}")
# print(f"loss_shape={loss.shape}")



