import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import MoeCausalLMOutputWithPast
from transformers import PretrainedConfig
import math
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin

class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"
    def __init__(self, d_model=768, num_layers=8, use_moe=False, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_layers = num_layers
        self.use_moe = use_moe
        self.dropout = kwargs.get("dropout", 0.0)
        self.vocab_size = kwargs.get("vocab_size", 6400)
        self.bos_token_id = kwargs.get("bos_token_id", 1)
        self.eos_token_id = kwargs.get("eos_token_id", 2)
        self.flash_attn = kwargs.get("flash_attn", True)
        self.num_heads = kwargs.get("num_heads", 8)
        self.num_kv_heads = kwargs.get("num_kv_heads", 4)
        self.head_dim = kwargs.get("head_dim", self.d_model // self.num_heads)
        self.hidden_act = kwargs.get("hidden_act", 'silu')
        self.dim_feedforward = kwargs.get("dim_feedforward", math.ceil(d_model * math.pi / 64) * 64)
        self.max_seq_len = kwargs.get("max_seq_len", 32768)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        self.rope_theta = kwargs.get("rope_theta", 1e6)
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)
        self.inference_rope_scaling = kwargs.get("inference_rope_scaling", False)
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None
        self.num_experts = kwargs.get("num_experts", 4)
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 1)
        self.moe_intermediate_size = kwargs.get("moe_intermediate_size", self.dim_feedforward)
        self.norm_topk_prob = kwargs.get("norm_topk_prob", True)
        self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef", 5e-4)

def precompute_freqs_cis(head_dim,max_seq_len,theta=1e6,rope_scaling=None):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float()/ head_dim))
    attention_factor = 1.0
    if rope_scaling is not None:
        original_max_seq_len = rope_scaling.get("original_max_position_embeddings",2048)
        factor = rope_scaling.get("factor", 16)
        beta_fast = rope_scaling.get("beta_fast", 32.0)
        beta_slow = rope_scaling.get("beta_slow", 1.0)
        attention_factor = rope_scaling.get("attention_factor",1.0)
        if max_seq_len / original_max_seq_len > 1.0:
            def find_correction_dim(beta):
                return (head_dim* math.log(original_max_seq_len/ (beta * 2 * math.pi))/ (2 * math.log(theta)))
            low = max(math.floor(find_correction_dim(beta_fast)),0)
            high = min(math.ceil(find_correction_dim(beta_slow)),head_dim // 2 - 1)
            ramp = torch.clamp(
                (torch.arange(head_dim // 2,device=inv_freq.device).float()- low)/ max(high - low, 0.001),
                min=0,
                max=1
            )
            inv_freq = inv_freq * (1 - ramp + ramp / factor)
    positions = torch.arange(max_seq_len,device=inv_freq.device).float()
    freqs = torch.outer(positions, inv_freq)
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)],dim=-1) * attention_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)],dim=-1) * attention_factor
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
    def __init__(self, config:MiniMindConfig):
        super().__init__()
        self.flash_attn = config.flash_attn
        assert config.d_model % config.num_heads == 0
        assert config.num_heads % config.num_kv_heads == 0
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.d_model // config.num_heads
        self.q_proj = nn.Linear(config.d_model,config.num_heads * self.head_dim,bias=False)
        self.k_proj = nn.Linear(config.d_model,config.num_kv_heads * self.head_dim,bias=False)
        self.v_proj = nn.Linear(config.d_model,config.num_kv_heads * self.head_dim,bias=False)
        self.o_proj = nn.Linear(config.num_heads * self.head_dim,config.d_model,bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim,eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim,eps=config.rms_norm_eps)
        self.dropout = config.dropout
        self.resid_dropout = nn.Dropout(config.dropout)
    def forward(self, x, position_embeddings,past_key_value=None,use_cache=False,attention_mask=None):
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
            can_use_fast_path = (self.flash_attn and past_key_value is None and (attention_mask is None or torch.all(attention_mask == 1)))
            if can_use_fast_path:
                attn_output = F.scaled_dot_product_attention(q,k,v,attn_mask=None,dropout_p=self.dropout if self.training else 0.0,is_causal=True,enable_gqa=True)
            else:
                key_padding_mask = None
                if attention_mask is not None:
                    key_padding_mask = attention_mask[:, None, None, :].to(
                        device=q.device,
                        dtype=torch.bool
                    )
                query_len = q.size(-2)
                key_len = k.size(-2)
                past_len = key_len - query_len
                query_positions = (torch.arange(query_len, device=q.device)+past_len)
                key_positions = torch.arange(key_len,device=q.device)
                causal_mask = (key_positions.unsqueeze(0) <= query_positions.unsqueeze(1))
                attn_mask = causal_mask[None, None, :, :]
                if key_padding_mask is not None:
                    attn_mask = attn_mask & key_padding_mask
                attn_output = F.scaled_dot_product_attention(q,k,v,dropout_p=self.dropout if self.training else 0.0,attn_mask=attn_mask,enable_gqa=True)
            attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len,self.num_heads * self.head_dim)
            attn_output = self.o_proj(attn_output)
            attn_output = self.resid_dropout(attn_output)
            return attn_output, present_key_value

class FeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig,dim_feedforward=None):
        super().__init__()
        # 修复：Config 中的 FFN 宽度字段名是 dim_feedforward。
        intermediate_size = (dim_feedforward if dim_feedforward is not None else config.dim_feedforward)
        self.gate_proj = nn.Linear(config.d_model,intermediate_size,bias=False)
        self.up_proj = nn.Linear(config.d_model,intermediate_size,bias=False)
        self.down_proj = nn.Linear(intermediate_size,config.d_model,bias=False)
        self.act_fn = ACT2FN[config.hidden_act]
    def forward(self, x):
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        hidden_states = gate * up
        output = self.down_proj(hidden_states)
        return output

class MOEFeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(config.d_model,config.num_experts,bias=False)
        self.experts = nn.ModuleList([
            FeedForward(config,config.moe_intermediate_size)
            for _ in range(config.num_experts)
        ])
    def _route_tokens(self, x):
        batch_size, seq_len, d_model = x.shape
        x_flat = x.reshape(batch_size * seq_len,d_model)
        router_logits = self.gate(x_flat)
        routing_probs = F.softmax(router_logits,dim=-1,dtype=torch.float32)
        topk_weights, topk_indices = torch.topk(routing_probs,k=self.config.num_experts_per_tok,dim=-1)
        if self.config.norm_topk_prob:
            topk_weights = topk_weights / topk_weights.sum(dim=-1,keepdim=True)
        topk_weights = topk_weights.to(x.dtype)
        return (x_flat,routing_probs,topk_indices,topk_weights,(batch_size, seq_len, d_model))
    def forward(self, x):
        (x_flat,routing_probs,topk_indices,topk_weights,original_shape) = self._route_tokens(x)
        output = torch.zeros_like(x_flat)
        for expert_id, expert in enumerate(self.experts):
            expert_mask = topk_indices == expert_id
            if not expert_mask.any():
                if self.training:
                    output[0, 0] += (0 * sum(p.sum() for p in expert.parameters()))
                continue
            token_indices = (
                expert_mask.any(dim=-1)
                .nonzero(as_tuple=False)
                .flatten()
            )
            expert_input = x_flat[token_indices]
            expert_output = expert(expert_input)
            expert_weights = topk_weights[expert_mask].unsqueeze(-1)
            weighted_output = expert_output * expert_weights
            output.index_add_(dim=0,index=token_indices,source=weighted_output.to(output.dtype))
        if self.training and self.config.router_aux_loss_coef > 0:
            load = F.one_hot(topk_indices,num_classes=self.config.num_experts).float().mean(dim=0)
            mean_routing_probs = routing_probs.mean(dim=0)
            self.aux_loss = ((load * mean_routing_probs).sum() * self.config.num_experts * self.config.router_aux_loss_coef)
        else:
            self.aux_loss = routing_probs.new_zeros(1).squeeze()
        batch_size, seq_len, d_model = original_shape
        return output.reshape(batch_size,seq_len,d_model)

class MiniMindBlock(nn.Module):
    def __init__(self,layer_id:int,config: MiniMindConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.input_layernorm = nn.RMSNorm(config.d_model,eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.d_model,eps=config.rms_norm_eps)
        self.mlp = MOEFeedForward(config) if config.use_moe else FeedForward(config)
    def forward(self, hidden_states, position_embeddings,past_key_value=None,use_cache=False,attention_mask=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states,present_key_value = self.self_attn(hidden_states,position_embeddings, past_key_value=past_key_value,use_cache=use_cache,attention_mask=attention_mask)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states,present_key_value

class MinimindModel(nn.Module):
    def __init__(self,config: MiniMindConfig):
        super().__init__()
        self.embeddings = nn.Embedding(config.vocab_size,config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([
            MiniMindBlock(layer_id,config)
            for layer_id in range(config.num_layers)
        ])
        head_dim = config.d_model // config.num_heads
        # 修复：将 Config 中的 YaRN/RoPE scaling 配置传入预计算函数。
        freqs_cos, freqs_sin = precompute_freqs_cis(head_dim=head_dim,max_seq_len=config.max_seq_len,theta=config.rope_theta,rope_scaling=config.rope_scaling)
        self.register_buffer("freqs_cos",freqs_cos,persistent=False)
        self.register_buffer("freqs_sin",freqs_sin,persistent=False)
        self.norm = nn.RMSNorm(config.d_model,eps=config.rms_norm_eps)
    def forward(self,input_ids,attention_mask=None,past_key_values=None,use_cache=False):
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
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present_key_value)
        x = self.norm(x)
        aux_loss = x.new_zeros(())
        for layer in self.layers:
            if isinstance(layer.mlp, MOEFeedForward):
                aux_loss = aux_loss + layer.mlp.aux_loss
        return x,presents,aux_loss

class MinimindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MiniMindConfig
    _tied_weights_keys = ["lm_head.weight"]
    def __init__(self,config: MiniMindConfig):
        super().__init__(config)
        self.model = MinimindModel(config)
        self.lm_head = nn.Linear(config.d_model,config.vocab_size,bias=False)
        self.post_init()
    def get_input_embeddings(self):
        return self.model.embeddings
    def set_input_embeddings(self, value):
        self.model.embeddings = value
    def get_output_embeddings(self):
        return self.lm_head
    def set_output_embeddings(self, value):
        self.lm_head = value
    def forward(self,input_ids,attention_mask=None,labels=None,past_key_values=None,use_cache=False,logits_to_keep=0):
        x, presents,aux_loss = self.model(input_ids,attention_mask=attention_mask,past_key_values=past_key_values,use_cache=use_cache)
        if logits_to_keep == 0:
            hidden_for_logits = x
        else:
            hidden_for_logits = x[:, -logits_to_keep:, :]
        logits = self.lm_head(hidden_for_logits)
        loss = None
        if labels is not None:
            shift_logits = logits[:,:-1,:]
            shift_lables = labels[:,1:]
            shift_logits = shift_logits.reshape(-1,shift_logits.shape[-1])
            shift_lables = shift_lables.reshape(-1)
            loss = F.cross_entropy(shift_logits,shift_lables)
        # 修复：使用支持 aux_loss 字段的 Transformers 标准 MoE 输出类。
        return MoeCausalLMOutputWithPast(loss=loss,aux_loss=aux_loss,logits=logits,past_key_values=tuple(presents) if use_cache else None)
    def generate(self,batch_size,device,max_new_tokens,input_ids, repetition_penalty,temperature,topk,top_p,do_sample,eos_token_id, attention_mask=None):
        with torch.no_grad():
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids,dtype=torch.long)
            finished = torch.zeros(batch_size,dtype=bool,device=device)
            past_key_values = None
            for _ in range(max_new_tokens):
                past_len = (past_key_values[0][0].shape[1]if past_key_values is not None else 0)
                model_input_ids = input_ids[:, past_len:]
                outputs = self.forward(input_ids=model_input_ids,attention_mask=attention_mask,past_key_values=past_key_values,use_cache=True,logits_to_keep=1)
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
                new_mask = attention_mask.new_ones(attention_mask.shape[0],1)
                attention_mask = torch.cat([attention_mask, new_mask],dim=-1 )
                past_key_values = outputs.past_key_values
                next_token = next_token.squeeze(-1)
                is_eos = torch.where(next_token==eos_token_id,True,False)
                finished = finished|is_eos
                if finished.all():
                    break
            return input_ids
