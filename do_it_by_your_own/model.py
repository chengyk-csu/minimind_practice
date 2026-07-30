import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.bias import causal_lower_right
from transformers.modeling_outputs import CausalLMOutputWithPast

def precompute_freqs_cis(head_dim, max_seq_len, theta=1e6):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(max_seq_len).float()
    freqs = torch.outer(positions, inv_freq)
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)],dim=-1)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)],dim=-1)
    return freqs_cos, freqs_sin

def apply_rotary_pos_emb(q, k, cos, sin):
    def rotate_half(x):
        half_dim = x.shape[-1] // 2
        return torch.cat([-x[..., half_dim:], x[..., :half_dim]],dim=-1)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed, k_embed

class Attention(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads,dropout=0.0):
        super().__init__()
        assert d_model % num_heads == 0
        assert num_heads % num_kv_heads == 0
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model,num_heads * self.head_dim,bias=False)
        self.k_proj = nn.Linear(d_model,num_kv_heads * self.head_dim,bias=False)
        self.v_proj = nn.Linear(d_model,num_kv_heads * self.head_dim,bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim,d_model,bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim,eps=1e-6)
        self.k_norm = nn.RMSNorm(self.head_dim,eps=1e-6)
        self.dropout = dropout
        self.resid_dropout = nn.Dropout(dropout)
    def forward(self, x, position_embeddings,past_key_value=None,use_cache=False):
            batch_size, seq_len, _ = x.shape
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)
            q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
            k = k.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
            v = v.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
            q = self.q_norm(q)
            k = self.k_norm(k)
            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb(q,k,cos,sin)
            if past_key_value is not None:
                past_k, past_v = past_key_value
                k = torch.cat([past_k, k], dim=1)
                v = torch.cat([past_v, v], dim=1)
            present_key_value = (k, v) if use_cache else None
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            if past_key_value is None:
                attn_mask = None
                use_causal_mask = True
            else:
                attn_mask = causal_lower_right(
                    q.size(-2),
                    k.size(-2)
                )
                use_causal_mask = False
            attn_output = F.scaled_dot_product_attention(q,k,v,dropout_p=self.dropout if self.training else 0.0,attn_mask=attn_mask,is_causal=use_causal_mask,enable_gqa=True)
            attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len,self.num_heads * self.head_dim)
            attn_output = self.o_proj(attn_output)
            attn_output = self.resid_dropout(attn_output)
            return attn_output, present_key_value

class FeedForward(nn.Module):
    def __init__(self, d_model, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(d_model,intermediate_size,bias=False)
        self.up_proj = nn.Linear(d_model,intermediate_size,bias=False)
        self.down_proj = nn.Linear(intermediate_size,d_model,bias=False)
        self.act_fn = nn.SiLU()
    def forward(self, x):
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        hidden_states = gate * up
        output = self.down_proj(hidden_states)
        return output

class MiniMindBlock(nn.Module):
    def __init__(self,d_model,num_heads,num_kv_heads,intermediate_size,dropout=0.0):
        super().__init__()
        self.self_attn = Attention(d_model=d_model,num_heads=num_heads,num_kv_heads=num_kv_heads,dropout=dropout)
        self.input_layernorm = nn.RMSNorm(d_model,eps=1e-6)
        self.post_attention_layernorm = nn.RMSNorm(d_model,eps=1e-6)
        self.mlp = FeedForward(d_model=d_model,intermediate_size=intermediate_size)
    def forward(self, hidden_states, position_embeddings,past_key_value=None,use_cache=False):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states,present_key_value = self.self_attn(hidden_states,position_embeddings, past_key_value=past_key_value,use_cache=use_cache)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states,present_key_value

class MinimindModel(nn.Module):
    def __init__(self,vocab_size,max_seq_len,d_model,dropout,
                 nhead,num_kv_heads,dim_feedforward,num_layers):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size,d_model)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            MiniMindBlock(d_model=d_model,num_heads=nhead,num_kv_heads=num_kv_heads,intermediate_size=dim_feedforward,dropout=dropout)
            for _ in range(num_layers)
        ])
        head_dim = d_model // nhead
        freqs_cos, freqs_sin = precompute_freqs_cis(head_dim=head_dim,max_seq_len=max_seq_len,theta=1e6)
        self.register_buffer("freqs_cos",freqs_cos,persistent=False)
        self.register_buffer("freqs_sin",freqs_sin,persistent=False)
        self.norm = nn.RMSNorm(d_model,eps=1e-6)
    def forward(self,input_ids,past_key_values=None,use_cache=False):
        B,T = input_ids.shape
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)
        start_pos = (past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0)
        x = self.embeddings(input_ids)
        x = self.dropout(x)
        position_embeddings = (self.freqs_cos[start_pos:start_pos+T],self.freqs_sin[start_pos:start_pos+T])
        presents = []
        for layer, past_key_value in zip(self.layers,past_key_values):
            x, present_key_value = layer(
                x,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache
            )
            presents.append(present_key_value)
        x = self.norm(x)
        return x,presents

class MinimindForCausalLM(nn.Module):
    def __init__(self,vocab_size,max_seq_len,d_model,dropout,
                 nhead,num_kv_heads,dim_feedforward,num_layers):
        super().__init__()
        self.model = MinimindModel(vocab_size,max_seq_len,d_model,dropout,
                 nhead,num_kv_heads,dim_feedforward,num_layers)
        self.lm_head = nn.Linear(d_model,vocab_size,bias=False)
        self.lm_head.weight = self.model.embeddings.weight
    def forward(self,input_ids,labels=None,past_key_values=None,use_cache=False):
        x, presents = self.model(input_ids,past_key_values=past_key_values,use_cache=use_cache)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            shift_logits = logits[:,:-1,:]
            shift_lables = labels[:,1:]
            shift_logits = shift_logits.reshape(-1,shift_logits.shape[-1])
            shift_lables = shift_lables.reshape(-1)
            loss = F.cross_entropy(shift_logits,shift_lables)
        return CausalLMOutputWithPast(loss=loss,logits=logits,past_key_values=tuple(presents) if use_cache else None)
    def generate(self,batch_size,device,max_new_tokens,input_ids,repetition_penalty,temperature,topk,top_p,do_sample,eos_token_id):
        with torch.no_grad():
            finished = torch.zeros(batch_size,dtype=bool,device=device)
            past_key_values = None
            for _ in range(max_new_tokens):
                past_len = (past_key_values[0][0].shape[1]if past_key_values is not None else 0)
                model_input_ids = input_ids[:, past_len:]
                outputs = self.forward(input_ids=model_input_ids,past_key_values=past_key_values,use_cache=True)
                logits = outputs.logits[:, -1, :]
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
                past_key_values = outputs.past_key_values
                next_token = next_token.squeeze(-1)
                is_eos = torch.where(next_token==eos_token_id,True,False)
                finished = finished|is_eos
                if finished.all():
                    break
            return input_ids


