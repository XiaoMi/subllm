# Copyright (C) 2024 Xiaomi Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, 
# software distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and limitations under the License.
#

from collections import namedtuple
import torch, math, logging
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple, Any
from torch import Tensor
from torch.nn import Module, CrossEntropyLoss
from torch.nn.parameter import Parameter
from transformers import PreTrainedModel
import torch.utils.checkpoint
from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass
from transformers import PretrainedConfig
import logging
from packaging import version
from typing import Optional, Tuple, Union
import random
import copy
from xformers.components.attention.core import scaled_dot_product_attention
from flash_attn.layers.rotary import ApplyRotaryEmbQKV_ as LegacyApplyRotaryEmbQKV_
from flash_attn.flash_attn_interface import flash_attn_func, flash_attn_qkvpacked_func
import ast 

import rotary_emb
from einops import rearrange
try:
    from xformers.ops import fmha
    XFORMERS_AVAILABLE = True
except ImportError:
    XFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)


class SubllmConfig(PretrainedConfig):
    model_type = "subllm"

    attribute_map = {
        "vocab_size": "num_embeddings",
        "hidden_size": "decoder_embed_dim",
        "n_layer": "decoder_layers",
        "n_head": "decoder_attention_heads"
    }

    def __init__(
            self,
            num_embeddings=32000,
            decoder_embed_dim=2048,
            decoder_ffn_embed_dim=2048*4,
            decoder_layers=24,
            decoder_attention_heads=32,
            norm_name="rmsnorm",
            mlp_name="swiglu",
            qk_bias=True,
            use_fused=False,
            use_flash=False,
            attn_type='vanilla',
            tokens_per_sample=2048,
            n_positions=2408,
            padding_idx=0,
            bos_token_id=1,
            eos_token_id=2,
            downsampling_factor_denominator="0.5",
            scale_token_embedding=False,
            qkv_pack = False,
            tie_weights = True,
            fix_zh = False,
            structure = None,
            **kwargs
    ):
        self.num_embeddings = num_embeddings
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_ffn_embed_dim = decoder_ffn_embed_dim
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.norm_name = norm_name
        self.mlp_name = mlp_name
        self.qk_bias = qk_bias
        self.use_fused = use_fused
        self.use_flash = use_flash
        self.attn_type = attn_type
        self.fix_zh = fix_zh
        self.tokens_per_sample = tokens_per_sample
        self.n_positions = n_positions
        self.padding_idx = padding_idx
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.scale_token_embedding = scale_token_embedding
        self.qkv_pack = qkv_pack
        self.tie_weights = tie_weights
        self.structure = structure
        # self.downsampling_factor = ast.literal_eval(downsampling_factor) 
        self.downsampling_factor = (1.0/ast.literal_eval(downsampling_factor_denominator),)
        
        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)


@dataclass
class CausalLMOutputs(ModelOutput):
    logits: torch.FloatTensor = None
    hidden_states: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    attn: List[Optional[Tensor]] = None
    inner_states: Optional[List[Optional[Tensor]]] = None


@dataclass
class CausalLMOutputsWithLoss(ModelOutput):
    loss: torch.FloatTensor = None
    logits: torch.FloatTensor = None
    hidden_states: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    attn: List[Optional[Tensor]] = None
    inner_states: Optional[List[Optional[Tensor]]] = None


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


try:
    from apex.normalization import FusedLayerNorm as _FusedLayerNorm
    from apex.normalization import FusedRMSNorm as _FusedRMSNorm

    has_fused_layernorm = True


    class FusedLayerNorm(_FusedLayerNorm):
        @torch.jit.unused
        def forward(self, x):
            if not x.is_cuda:
                return super().forward(x)
            else:
                with torch.cuda.device(x.device):
                    return super().forward(x)


    class FusedRMSNorm(_FusedRMSNorm):
        @torch.jit.unused
        def forward(self, x):
            if not x.is_cuda:
                return super().forward(x)
            else:
                with torch.cuda.device(x.device):
                    return super().forward(x)
except ImportError:
    has_fused_layernorm = False


def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True, export=False, norm_name="layernorm"):
    if norm_name == "layernorm":
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            export = True
        if not export and torch.cuda.is_available() and has_fused_layernorm:
            return FusedLayerNorm(normalized_shape, eps, elementwise_affine)
        return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)
    elif norm_name == "rmsnorm":
        if not export and torch.cuda.is_available() and has_fused_layernorm:
            return FusedRMSNorm(normalized_shape, eps, elementwise_affine)
        return RMSNorm(normalized_shape, eps)
    else:
        raise NotImplementedError
    


def utils_fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float("-inf")).type_as(t)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def utils_softmax(x, dim: int, onnx_trace: bool = False):
    if onnx_trace:
        return F.softmax(x.float(), dim=dim)
    else:
        return F.softmax(x, dim=dim, dtype=torch.bfloat16)



def TokenEmbedding(
        num_embeddings,
        embedding_dim,
        padding_idx,
        initialize_params_on_gpu=False,
        dtype: Optional[torch.dtype] = None,
        model_parallel=False
):
    """
    Returns an embedding initialized to normal(0, 1/sqrt(embedding_dim))
    with the padding token embedding initialized to 0.
    """
    # if not model_parallel:
    # Passing weights initialized on GPU.
    device = torch.cuda.current_device() if initialize_params_on_gpu else None
    if dtype is None:
        dtype = torch.float
    weight = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
    nn.init.normal_(weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(weight[padding_idx], 0)
    m = nn.Embedding(
        num_embeddings, embedding_dim, padding_idx=padding_idx, _weight=weight
    )
    return m


class VanillaAttn(nn.Module):
    def __init__(self, num_heads, dropout=0., embed_dim=None) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim
        self.scaling = self.head_dim ** -0.5
        self.onnx_trace = False
        self.use_flash2 = False

    def attend(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            mask=None,
            key_padding_mask=None,
            use_cache=False,
            layer_past=None,
    ):
        if layer_past is not None:
            # k,v: seqlen, batch, numheads * headdim
            past_key, past_value = layer_past[0], layer_past[1]
            k = torch.cat((past_key, k), dim=0)
            v = torch.cat((past_value, v), dim=0)
        if use_cache:
            present = (k, v)
        else:
            present = None
        
        # q-> seq_length, bsz, (num_heads * head_dim)
        tgt_len, bsz, embed_dim = q.size()
        q *= self.scaling
        q = (
            q.contiguous()
            .view(tgt_len, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        kv_bsz = bsz  # need default value for scripting
        if k is not None:
            kv_bsz = k.size(1)
            k = (
                k.contiguous()
                .view(-1, kv_bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )
        if v is not None:
            v = (
                v.contiguous()
                .view(-1, kv_bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        if mask is not None:
            if key_padding_mask is not None:
                key_padding_mask = (
                    key_padding_mask.view(bsz, 1, 1, tgt_len)
                    .expand(-1, self.num_heads, -1, -1)
                    .reshape(bsz * self.num_heads, 1, tgt_len)
                    .expand(-1, tgt_len, -1)
                )
                mask = mask.unsqueeze(0).expand(bsz * self.num_heads, -1, -1).clone()
                mask.masked_fill_(key_padding_mask, float('-inf'))
            else:
                mask = mask.unsqueeze(0)
           
            if self.onnx_trace:
                mask = mask.repeat(attn_weights.size(0), 1, 1)
            
            attn_weights += mask
            
        attn_weights_float = utils_softmax(
            attn_weights, dim=-1, onnx_trace=self.onnx_trace
        )
        attn_weights_float = attn_weights_float.masked_fill(mask.eq(float('-inf')), 0.)
        
        attn_probs = attn_weights_float.type_as(attn_weights)
        attn = torch.bmm(attn_probs, v)
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, self.embed_dim)
     
        return attn, present


class FlashAttn(nn.Module):
    def __init__(self, num_heads, dropout=0., embed_dim=None) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.attention = fmha.memory_efficient_attention
        # self.use_torch2 = not version.parse(torch.__version__) < version.parse('2.0.0')
        self.use_torch2 = True

    def attend(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            mask=None,
            key_padding_mask=None,
            use_cache=False,
            layer_past=None
    ) -> Tuple[Tensor, Optional[Tuple[Tensor]]]:
        # q-> seq_length, bsz, (num_heads * head_dim)
        tgt_len, bsz, embed_dim = q.size()

        if layer_past is not None:
            # k,v: seqlen batch, numheads, headdim
            past_key, past_value = layer_past[0], layer_past[1]
            k = torch.cat((past_key, k), dim=0)
            v = torch.cat((past_value, v), dim=0)

        if use_cache:
            present = (k, v)
        else:
            present = None
        if key_padding_mask is not None and mask is not None:
            key_padding_mask = (
                key_padding_mask.view(bsz, 1, 1, tgt_len)
                .expand(-1, self.num_heads, -1, -1)
                .reshape(bsz * self.num_heads, 1, tgt_len)
                .expand(-1, tgt_len, -1)
            )
            mask = mask.unsqueeze(0).expand(bsz * self.num_heads, -1, -1).clone()
            mask.masked_fill_(key_padding_mask, float('-inf'))
        if self.use_torch2:
            def massage(x):
                return (
                    x.contiguous()
                    .view(-1, bsz, self.num_heads, self.head_dim)
                    .transpose(0, 1).transpose(1, 2)
                )
            q = massage(q)
            if k is not None:
                k = massage(k)
            if v is not None:
                v = massage(v)
            with torch.backends.cuda.sdp_kernel(**{'enable_flash': True, 'enable_math': False, 'enable_mem_efficient': False}):
                y = F.scaled_dot_product_attention(
                    q, k, v,
                    dropout_p = self.dropout if self.training else 0.,
                    is_causal = True if key_padding_mask is None else False,
                    attn_mask=mask if key_padding_mask is not None else None
                )
            y = (
                y.transpose(1, 2)
                .flatten(start_dim=2, end_dim=3)
                .transpose(0, 1)
            )
        else:
            def massage(x):
                return (
                    x.contiguous()
                    .view(-1, bsz, self.num_heads, self.head_dim)
                    .transpose(0, 1)
                )

            q = massage(q)
            if k is not None:
                k = massage(k)
            if v is not None:
                v = massage(v)
            
            y = self.attention(q, k, v, p=self.dropout, attn_bias=mask)
            y = (
                y.flatten(start_dim=2, end_dim=3)
                .transpose(0, 1)
            )
        return y, present


class Linear(Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = True,
            initialize_params_on_gpu: bool = False,
            dtype: torch.dtype = None,
    ) -> None:
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        device = torch.cuda.current_device() if initialize_params_on_gpu else None
        if dtype is None:
            dtype = torch.float
        self.weight = Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        if bias:
            self.bias = Parameter(torch.empty(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter("bias", None)
        #self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight, self.bias)


class SwiGLUInf(nn.Module):
    """
    x1 = F.linear(x, w1, b1)
    x2 = F.linear(x, w2, b2)
    hidden = F.silu(x1) * x2
    return F.linear(hidden, w3, b3)
    """

    def __init__(
            self,
            in_features: int,
            hidden_features: int,
            out_features: Optional[int] = None,
            bias: bool = True,
            *,
            _pack_weights: bool = True,
            model_parallel=False,
            initialize_params_on_gpu=False
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.w12: Optional[nn.Linear]
        _pack_weights = True
        if _pack_weights:
            self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        else:
            self.w12 = None
            self.w1 = nn.Linear(in_features, hidden_features, bias=bias)
            self.w2 = nn.Linear(in_features, hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

        self.hidden_features = hidden_features
        self.out_features = out_features
        self.in_features = in_features

    def forward(self, x):
        if self.w12 is None:
            return self.w3(F.silu(self.w1(x)) * self.w2(x))
        xx = self.w12(x)
        xx, xy = xx.split(split_size=self.hidden_features, dim=-1)
        return self.w3(F.silu(xx) * xy)


class FeedForward(nn.Module):
    def __init__(self,
                 embed_dim,
                 intermediate_size=None,
                 initialize_params_on_gpu=False,
                 model_parallel=False,
                 layer_index=None) -> None:
        super().__init__()
        self.swiglu = SwiGLUInf(in_features=embed_dim,
                                hidden_features=intermediate_size,
                                out_features=embed_dim,
                                bias=False,
                                model_parallel=model_parallel,
                                initialize_params_on_gpu=initialize_params_on_gpu)
        #self._mlp_init(layer_index=layer_index)

    def forward(self, x):
        x = self.swiglu(x)
        return x

    def _mlp_init(self, layer_index=None):
        # Empirically observed the convergence to be much better with
        # the scaled initialization
        init_method_normal()(self.swiglu.w12.weight)
        if layer_index is not None:
            scaled_init_method_normal(num_layers=layer_index)(self.swiglu.w3.weight)
        else:
            init_method_normal()(self.swiglu.w3.weight)


def get_attention(num_heads=32, 
                  dropout=0., 
                  block_size=None, 
                  seq_length=2048, 
                  embed_dim = 2048,
                  attention_type = "flash"):
    """
    attention_type : blocksparse | flash | scaled | self_attn | scaledot
    """
    # logger.info(f"--------------  self attention: {attention_type}  ---------------")
    # get_rank_and_world_size()
    if attention_type == == 'flash':
        attention = FlashAttn(num_heads=num_heads, dropout=dropout, embed_dim=embed_dim)
    elif attention_type == 'vanilla':
        attention = VanillaAttn(num_heads=num_heads, dropout=dropout, embed_dim=embed_dim)
    else:
        attention = None
    return attention


class ApplyRotaryEmbQKV_(torch.autograd.Function):
    """
    ApplyRotaryEmbQKV_
    """

    @staticmethod
    def forward(ctx, qkv, cos, sin, cos_k=None, sin_k=None):
        """
            qkv: (total, 3, nheads, headdim)
            cos, sin: (seqlen, rotary_dim / 2)
            cos_k, sin_k: (seqlen, rotary_dim / 2), optional
        rotary_dim must be <= headdim
        Apply rotary embedding *inplace* to the first rotary_dim of q and k.
        """
        _, three, _, headdim = qkv.shape
        assert three == 3
        rotary_seqlen, rotary_dim = cos.shape
        rotary_dim *= 2
        assert rotary_dim <= headdim
        cos_k = cos if cos_k is None else cos_k
        sin_k = sin if sin_k is None else sin_k
        assert sin.shape == cos_k.shape == sin_k.shape == (rotary_seqlen, rotary_dim // 2)
        q1, q2 = qkv[:, 0, :, :rotary_dim].chunk(2, dim=-1)
        rotary_emb.apply_rotary(q1, q2, rearrange(cos, "s d -> s 1 d"), rearrange(sin, "s d -> s 1 d"), q1, q2, False)
        k1, k2 = qkv[:, 1, :, :rotary_dim].chunk(2, dim=-1)
        rotary_emb.apply_rotary(
            k1, k2, rearrange(cos_k, "s d -> s 1 d"), rearrange(sin_k, "s d -> s 1 d"), k1, k2, False
        )
        ctx.save_for_backward(cos, sin, cos_k, sin_k)
        return qkv

    @staticmethod
    def backward(ctx, dqkv):
        cos, sin, cos_k, sin_k = ctx.saved_tensors
        rotary_dim = cos.shape[-1]
        rotary_dim *= 2
        dq1, dq2 = dqkv[:, 0, :, :rotary_dim].chunk(2, dim=-1)
        rotary_emb.apply_rotary(
            dq1, dq2, rearrange(cos, "s d -> s 1 d"), rearrange(sin, "s d -> s 1 d"), dq1, dq2, True
        )
        dk1, dk2 = dqkv[:, 1, :, :rotary_dim].chunk(2, dim=-1)
        rotary_emb.apply_rotary(
            dk1, dk2, rearrange(cos_k, "s d -> s 1 d"), rearrange(sin_k, "s d -> s 1 d"), dk1, dk2, True
        )
        return dqkv, None, None, None, None


def apply_rotary_pos_emb(x, cos, sin, cos_cached_expanded, sin_cached_expanded, start=0, seq_len=None, input_len=int, pos_index=None):
    if seq_len==1:
        input_len=[x - 1 for x in input_len]
        cos=cos[:,:,input_len,:].transpose(0, 2).contiguous()
        sin=sin[:,:,input_len,:].transpose(0, 2).contiguous()
        return (x * cos) + (rotate_half(x) * sin)
    elif pos_index is None:
        cos = cos[:, :, start:input_len + start, :]
        sin = sin[:, :, start:input_len + start, :]
        return (x * cos) + (rotate_half(x) * sin)
    else:
        cos = torch.gather(cos_cached_expanded, 2, pos_index.unsqueeze(1).unsqueeze(-1).expand(-1, -1, -1, cos_cached_expanded.shape[-1]))
        sin = torch.gather(sin_cached_expanded, 2, pos_index.unsqueeze(1).unsqueeze(-1).expand(-1, -1, -1, sin_cached_expanded.shape[-1]))
        return (x * cos) + (rotate_half(x) * sin)


class RotaryEmbedding(torch.nn.Module):
    """
    The rotary position embeddings from RoFormer_ (Su et. al).
    A crucial insight from the method is that the query and keys are
    transformed by rotation matrices which depend on the relative positions.

    Other implementations are available in the Rotary Transformer repo_ and in
    GPT-NeoX_, GPT-NeoX was an inspiration

    .. _RoFormer: https://arxiv.org/abs/2104.09864
    .. _repo: https://github.com/ZhuiyiTechnology/roformer
    .. _GPT-NeoX: https://github.com/EleutherAI/gpt-neox


    .. warning: Please note that this embedding is not registered on purpose, as it is transformative
        (it does not create the embedding dimension) and will likely be picked up (imported) on a ad-hoc basis
    """

    def __init__(self, dim_model: int, tokens_per_sample: int, *_, **__):
        super().__init__()
        # Generate and save the inverse frequency buffer (non trainable)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim_model, 2).float() / dim_model))
        self.register_buffer("inv_freq", inv_freq)

        self.tokens_per_sample = max(tokens_per_sample, 8192) # TODO
        
        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None

        self._cos_cached_expanded = None 
        self._sin_cached_expanded = None 
        
        
    def _update_cos_sin_tables(self, bsz, seq_len, device, dtype, pos_index=None):
        # Reset the tables if the sequence length has changed,
        # or if we're on a new device (possibly due to tracing for instance)
        if (
            seq_len != self._seq_len_cached
        ):
            self._seq_len_cached = seq_len
            t = torch.arange(
                seq_len, device=device, dtype=torch.float32
            )
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(dtype))
            emb = torch.cat((freqs, freqs), dim=-1).to(device) # 

            self._cos_cached = emb.cos()[None, None, :, :].to(dtype) # local_to_dt(, x, dim=1) [1, 1, 2048, 64]
            self._sin_cached = emb.sin()[None, None, :, :].to(dtype)
            self._cos_cached_expanded = None 
            self._sin_cached_expanded = None 
            if pos_index is not None:
                self._cos_cached_expanded = self._cos_cached.expand(bsz, -1, -1, -1)
                self._sin_cached_expanded = self._sin_cached.expand(bsz, -1, -1, -1)
          
        return self._cos_cached, self._sin_cached

    def forward(
            self, 
            q: torch.Tensor, 
            k: torch.Tensor, 
            start_pos: Optional[int],
            seq_len: Optional[int],
            input_len: [],
            pos_index = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        self._cos_cached, self._sin_cached = self._update_cos_sin_tables(bsz=q.shape[0], seq_len=self.tokens_per_sample, device=q.device, dtype=q.dtype, pos_index=pos_index)
       
        return (
            apply_rotary_pos_emb(q, self._cos_cached, self._sin_cached, self._cos_cached_expanded, self._sin_cached_expanded, start=start_pos, seq_len=seq_len, input_len=input_len, pos_index=pos_index),
            apply_rotary_pos_emb(k, self._cos_cached, self._sin_cached, self._cos_cached_expanded, self._sin_cached_expanded, start=start_pos, seq_len=seq_len, input_len=input_len, pos_index=pos_index),
        )

def get_position(dim_model, tokens_per_sample = 2048):
    return RotaryEmbedding(dim_model=dim_model, tokens_per_sample=tokens_per_sample)


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        qk_bias = False,
        inference = False,
        block_size = 64,
        tokens_per_sample = 2048,
        layer_index = 0,
        initialize_params_on_gpu=False,
        attn_type = 'flash',
        dtype = None,   
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.layer_index = layer_index
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.model_parallel_size = 1
        self.num_heads_partition = num_heads // self.model_parallel_size
        assert (
            self.num_heads_partition * self.model_parallel_size == num_heads
        ), "Number of heads must be divisible by model parallel size"
        self.embed_dim_partition = embed_dim // self.model_parallel_size
        
        self.pack_qkv = False
        self.q_proj = Linear(embed_dim, embed_dim, bias=bias or qk_bias, initialize_params_on_gpu=initialize_params_on_gpu, dtype=dtype)
        self.k_proj = Linear(embed_dim, embed_dim, bias=bias or qk_bias, initialize_params_on_gpu=initialize_params_on_gpu, dtype=dtype)
        self.v_proj = Linear(embed_dim, embed_dim, bias=bias, initialize_params_on_gpu=initialize_params_on_gpu, dtype=dtype)
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias, initialize_params_on_gpu=initialize_params_on_gpu, dtype=dtype)
       
        self.reset_parameters(layer_index=layer_index)
        self.onnx_trace = False
        self.inference = inference
        self.attention = get_attention(num_heads=self.num_heads_partition, 
                                        dropout=dropout, 
                                        block_size=block_size, 
                                        seq_length=tokens_per_sample, 
                                        embed_dim = self.embed_dim_partition,
                                        attention_type=attn_type)
        self.no_mask = False 
        self.position = get_position(dim_model=self.head_dim, tokens_per_sample=tokens_per_sample)
 
    def forward(
        self,
        query,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask = None,
        start_pos: Optional[int] = None,
        use_cache: bool = False, 
        layer_past: Optional[Tensor] = None,
        input_len: Optional[list] = [],
        pos_index = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        
        seq_len, bsz, embed_dim = query.size()
        # seq_length , bsz ,embed_dim
        q = self.q_proj(query)
        k = self.k_proj(query)
        v = self.v_proj(query)

        def split_heads(x):
            return (
                x.contiguous()
                .view(-1, bsz, self.num_heads, self.head_dim)
                .transpose(0, 1)
                .transpose(1, 2)
            ) # to bsz, num_heads, seq_len, head_dim
        
        q, k = split_heads(q), split_heads(k)
        # q - > bsz, num_heads, seq_length, head_dim
        
        if layer_past is None:
            q_batch = []
            k_batch = []
            for batch_index in range(bsz):
                if input_len[batch_index] <= seq_len:
                    assert pos_index is None 
                    q_index, k_index = self.position(q[batch_index,:,seq_len-input_len[batch_index]:,:], k[batch_index,:,seq_len-input_len[batch_index]:,:], start_pos=start_pos, seq_len=seq_len , input_len=input_len[batch_index], pos_index=pos_index)

                    q_index = torch.cat((q[batch_index:batch_index+1,:,0:seq_len-input_len[batch_index],:], q_index), 2)
                    k_index = torch.cat((k[batch_index:batch_index+1,:,0:seq_len-input_len[batch_index],:], k_index), 2)
                else:
                    q_index, k_index = self.position(q[batch_index,:,:,:], k[batch_index,:,:,:], start_pos=start_pos, seq_len=seq_len , input_len=input_len[batch_index], pos_index=pos_index[batch_index].unsqueeze(0))


                q_batch.append(q_index)
                k_batch.append(k_index)

            q = torch.cat(q_batch, dim=0)
            k = torch.cat(k_batch, dim=0)
        else:
            q, k = self.position(q, k, start_pos=start_pos, seq_len=seq_len , input_len=input_len, pos_index=pos_index)

        
        def back_dim(x):
            return (
                x.transpose(1, 2)
                .transpose(0, 1)
                .contiguous()
                .view(-1, bsz, self.num_heads*self.head_dim)
            )
        q, k = back_dim(q), back_dim(k)
        
        # q->  seq_length, bsz, num_heads * head_dim
        attn, present = self.attention.attend(
            q, 
            k, 
            v, 
            attn_mask, 
            key_padding_mask, 
            use_cache=use_cache, 
            layer_past=layer_past
        )
        
        attn = self.out_proj(attn)
        return attn, present
    
    def reset_parameters(self, layer_index = None):
        if layer_index is not None:
            self._init_attn(li=layer_index)
        else:
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)
        
        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0.0)
        if self.k_proj.bias is not None:
            nn.init.constant_(self.k_proj.bias, 0.0)
        if self.v_proj.bias is not None:
            nn.init.constant_(self.v_proj.bias, 0.0)
    
    def _init_attn(self, li = None):
        logger.info(f"init attn: {li}")
        
        init_method_normal()(self.k_proj.weight)
        init_method_normal()(self.v_proj.weight)
        init_method_normal()(self.q_proj.weight)
        if li is not None:
            scaled_init_method_normal(num_layers=li)(self.out_proj.weight)
        else:
            init_method_normal()(self.out_proj.weight)


class TransformerDecoderBlock(nn.Module):
    def __init__(self, 
                 embed_dim=2048, 
                 intermediate_size = 2048*4,
                 attention_heads = 32,
                 dropout = 0.1,
                 attention_dropout = 0.0,
                 activation_fn = 'gelu',
                 normalize_before=True,
                 block_size = 64,
                 tokens_per_sample = 2048,
                 bias = True,
                 qk_bias = False,
                 layer_index = 0,
                 attn_type = "flash",
                 mlp_name = "swiglu",
                 norm_name = "rmsnorm",
                 dtype = None,
                 initialize_params_on_gpu=False,
                 model_parallel_size = 1,
                 export=False, **kwargs) -> None:
        super().__init__()
        device = torch.cuda.current_device() if initialize_params_on_gpu else None
        self.self_attn_layer_norm = LayerNorm(embed_dim, export=export, norm_name=norm_name)
        self.final_layer_norm = LayerNorm(embed_dim, export=export, norm_name=norm_name)
        if initialize_params_on_gpu:
            # logger.info("-------------- in gpu ----------------")
            self.self_attn_layer_norm.to(device).to(dtype)
            self.final_layer_norm.to(device).to(dtype)

        self.self_attn = MultiheadAttention(
            embed_dim,
            attention_heads,
            dropout=attention_dropout,
            qk_bias = qk_bias,
            block_size=block_size,
            tokens_per_sample=tokens_per_sample,
            bias=bias,
            layer_index=layer_index,
            dtype=dtype,
            initialize_params_on_gpu = initialize_params_on_gpu,
            attn_type = attn_type,
        )
        self.no_mask = self.self_attn.no_mask

        self.dropout_module = Dropout(p=dropout, module_name=self.__class__.__name__)
        self.mlp_name = mlp_name
        self.feedforward = FeedForward(
            embed_dim=embed_dim,
            intermediate_size=intermediate_size,
            layer_index=layer_index,
            initialize_params_on_gpu=initialize_params_on_gpu,
            model_parallel=True if model_parallel_size > 1 else False,
        )
        self.onnx_trace = False


    def forward(
            self, 
            x,
            self_attn_padding_mask = None,
            self_attn_mask = None,
            start_pos = None,
            use_cache: Optional[bool] = False,
            layer_past: Optional[Tuple[torch.Tensor]] = None, 
            input_len: Optional[list] = [],
            pos_index = None,
    ):
        residual = x
        # x -> seq_len x bsz x hidden_dim
        x = self.self_attn_layer_norm(x)
     
        x, present = self.self_attn(
            query=x,
            attn_mask=self_attn_mask,
            key_padding_mask=self_attn_padding_mask,
            start_pos=start_pos,
            use_cache=use_cache,
            layer_past=layer_past,
            input_len=input_len,
            pos_index=pos_index,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        residual = x
        x = self.final_layer_norm(x)
        # ffn
        x = self.feedforward(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        return x, present

    def residual_connection(self, x, residual):
        return residual + x



def Embedding(
    num_embeddings,
    embedding_dim,
    padding_idx,
    initialize_params_on_gpu=False,
    dtype: Optional[torch.dtype] = None,
    model_parallel = False
):
    """
    Returns an embedding initialized to normal(0, 1/sqrt(embedding_dim))
    with the padding token embedding initialized to 0.
    """
    # Passing weights initialized on GPU.
    device = torch.cuda.current_device() if initialize_params_on_gpu else None
    if dtype is None:
        dtype = torch.float
    weight = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
    nn.init.normal_(weight, mean=0, std=embedding_dim**-0.5)
    nn.init.constant_(weight[padding_idx], 0)
    m = nn.Embedding(
        num_embeddings, embedding_dim, padding_idx=padding_idx, _weight=weight
    )

    return m


def _no_op(x: Tensor) -> Tensor:
    if (torch.jit.is_scripting()):
        return x
    else:
        # a no-op function that will have a node in the autograd graph,
        # to avoid certain bugs relating to backward hooks
        return x.chunk(1, dim=-1)[0]


def make_pad_mask(lengths: torch.Tensor) -> torch.Tensor:
    """
    Args:
      lengths:
        A 1-D tensor containing sentence lengths.
    Returns:
      Return a 2-D bool tensor, where masked positions
      are filled with `True` and non-masked positions are
      filled with `False`.

    >>> lengths = torch.tensor([1, 3, 2, 5])
    >>> make_pad_mask(lengths)
    tensor([[False,  True,  True,  True,  True],
            [False, False, False,  True,  True],
            [False, False,  True,  True,  True],
            [False, False, False, False, False]])
    """
    assert lengths.ndim == 1, lengths.ndim

    max_len = lengths.max()
    n = lengths.size(0)

    expaned_lengths = torch.arange(max_len).expand(n, max_len).to(lengths)

    return expaned_lengths >= lengths.unsqueeze(1)


class LimitParamValue(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, min: float, max: float):
        ctx.save_for_backward(x)
        assert max >= min
        ctx.min = min
        ctx.max = max
        return x
    @staticmethod
    def backward(ctx, x_grad: Tensor):
        x, = ctx.saved_tensors
        # where x < ctx.min, ensure all grads are negative (this will tend to make
        # x more positive).
        x_grad = x_grad * torch.where(torch.logical_and(x_grad > 0, x < ctx.min), -1.0, 1.0)
        # where x > ctx.max, ensure all grads are positive (this will tend to make
        # x more negative).
        x_grad *= torch.where(torch.logical_and(x_grad < 0, x > ctx.max), -1.0, 1.0)
        return x_grad, None, None


def limit_param_value(x: Tensor,
                      min: float, max: float,
                      prob: float = 0.6,
                      training: bool = True):
    # You apply this to (typically) an nn.Parameter during training to ensure that its
    # (elements mostly) stays within a supplied range.  This is done by modifying the
    # gradients in backprop.
    # It's not necessary to do this on every batch: do it only some of the time,
    # to save a little time.
    if training and random.random() < prob:
        return LimitParamValue.apply(x, min, max)
    else:
        return x

class ClipGradFunction(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx,
            x: Tensor,
            limit: float):
        ctx.limit = limit
        return x

    @staticmethod
    def backward(ctx, x_grad, *args):
        return x_grad.clamp(-ctx.limit, ctx.limit), None

def clip_grad(x: Tensor, limit: float):
    return ClipGradFunction.apply(x, limit)


class Dropout(nn.Module):
    def __init__(self, p, module_name=None):
        super().__init__()
        self.p = p
        self.module_name = module_name
        self.apply_during_inference = False

    def forward(self, x, inplace: bool = False):
        if self.p > 0 and (self.training or self.apply_during_inference):
            return F.dropout(x, p=self.p, training=True, inplace=inplace)
        else:
            return x
        


class PiecewiseLinear(object):
    """
    Piecewise linear function, from float to float, specified as nonempty list of (x,y) pairs with
    the x values in order.  x values <[initial x] or >[final x] are map to [initial y], [final y]
    respectively.
    """
    def __init__(self, *args):
        assert len(args) >= 1
        if len(args) == 1 and isinstance(args[0], PiecewiseLinear):
            self.pairs = list(args[0].pairs)
        else:
            self.pairs = [ (float(x), float(y)) for x,y in args ]
        for (x,y) in self.pairs:
            assert isinstance(x, float) or isinstance(x, int)
            assert isinstance(y, float) or isinstance(y, int)

        for i in range(len(self.pairs) - 1):
            assert self.pairs[i + 1][0] > self.pairs[i][0], self.pairs


    def __str__(self):
        # e.g. 'PiecewiseLinear((0., 10.), (100., 0.))'
        return f'PiecewiseLinear({str(self.pairs)[1:-1]})'

    def __call__(self, x):
        if x <= self.pairs[0][0]:
            return self.pairs[0][1]
        elif x >= self.pairs[-1][0]:
            return self.pairs[-1][1]
        else:
            cur_x, cur_y = self.pairs[0]
            for i in range(1, len(self.pairs)):
                next_x, next_y = self.pairs[i]
                if x >= cur_x and x <= next_x:
                    return cur_y + (next_y - cur_y) * (x - cur_x) / (next_x - cur_x)
                cur_x, cur_y = next_x, next_y
            assert False

    def __mul__(self, alpha):
        return PiecewiseLinear(
            * [(x, y * alpha) for x, y in self.pairs])

    def __add__(self, x):
        if isinstance(x, float) or isinstance(x, int):
            return PiecewiseLinear(
                * [(p[0], p[1] + x) for p in self.pairs])
        s, x = self.get_common_basis(x)
        return PiecewiseLinear(
            * [(sp[0], sp[1] + xp[1]) for sp, xp in zip(s.pairs, x.pairs)])

    def max(self, x):
        if isinstance(x, float) or isinstance(x, int):
            x = PiecewiseLinear( (0, x) )
        s, x = self.get_common_basis(x, include_crossings=True)
        return PiecewiseLinear(
            * [(sp[0], max(sp[1], xp[1])) for sp, xp in zip(s.pairs, x.pairs)])

    def min(self, x):
        if isinstance(x, float) or isinstance(x, int):
            x = PiecewiseLinear( (0, x) )
        s, x = self.get_common_basis(x, include_crossings=True)
        return PiecewiseLinear(
            * [ (sp[0], min(sp[1], xp[1])) for sp, xp in zip(s.pairs, x.pairs)])

    def __eq__(self, other):
        return self.pairs == other.pairs

    def get_common_basis(self,
                         p: 'PiecewiseLinear',
                         include_crossings: bool = False):
        """
        Returns (self_mod, p_mod) which are equivalent piecewise lienar
        functions to self and p, but with the same x values.

          p: the other piecewise linear function
          include_crossings: if true, include in the x values positions
              where the functions indicate by this and p crosss.
        """
        assert isinstance(p, PiecewiseLinear)

        # get sorted x-values without repetition.
        x_vals = sorted(set([ x for x, y in self.pairs ] + [ x for x, y in p.pairs ]))
        y_vals1 = [ self(x) for x in x_vals ]
        y_vals2 = [ p(x) for x in x_vals ]

        if include_crossings:
            extra_x_vals = []
            for i in range(len(x_vals) - 1):
                if (y_vals1[i] > y_vals2[i]) != (y_vals1[i+1] > y_vals2[i+1]):
                    # if the two lines in this subsegment potentially cross each other..
                    diff_cur = abs(y_vals1[i] - y_vals2[i])
                    diff_next = abs(y_vals1[i+1] - y_vals2[i+1])
                    # `pos`, between 0 and 1, gives the relative x position,
                    # with 0 being x_vals[i] and 1 being x_vals[i+1].
                    pos = diff_cur / (diff_cur + diff_next)
                    extra_x_val = x_vals[i] + pos * (x_vals[i+1] - x_vals[i])
                    extra_x_vals.append(extra_x_val)
            if len(extra_x_vals) > 0:
                x_vals = sorted(set(x_vals + extra_x_vals))
        y_vals1 = [ self(x) for x in x_vals ]
        y_vals2 = [ p(x) for x in x_vals ]
        return ( PiecewiseLinear(* zip(x_vals, y_vals1)),
                 PiecewiseLinear(* zip(x_vals, y_vals2)) )


class ScheduledFloat(torch.nn.Module):
    """
    This object is a torch.nn.Module only because we want it to show up in [top_level module].modules();
    it does not have a working forward() function.  You are supposed to cast it to float, as
    in, float(parent_module.whatever), and use it as something like a dropout prob.

    It is a floating point value whose value changes depending on the batch count of the
    training loop.  It is a piecewise linear function where you specifiy the (x,y) pairs
    in sorted order on x; x corresponds to the batch index.  For batch-index values before the
    first x or after the last x, we just use the first or last y value.

    Example:
       self.dropout = ScheduledFloat((0.0, 0.2), (4000.0, 0.0), default=0.0)

    `default` is used when self.batch_count is not set or in training or mode or in
     torch.jit scripting mode.
    """
    def __init__(self,
                 *args,
                 default: float = 0.0):
        super().__init__()
        # self.batch_count and self.name will be written to in the training loop.
        self.batch_count = None
        self.name = None
        self.default = default
        self.schedule = PiecewiseLinear(*args)

    def extra_repr(self) -> str:
        return f'batch_count={self.batch_count}, schedule={str(self.schedule.pairs[1:-1])}'

    def __float__(self):
        batch_count = self.batch_count
        if batch_count is None or not self.training or torch.jit.is_scripting():
            return float(self.default)
        else:
            ans = self.schedule(self.batch_count)
            if random.random() < 0.0002:
                logging.info(f"ScheduledFloat: name={self.name}, batch_count={self.batch_count}, ans={ans}")
            return ans

    def __add__(self, x):
        if isinstance(x, float) or isinstance(x, int):
            return ScheduledFloat(self.schedule + x,
                                  default=self.default)
        else:
            return ScheduledFloat(self.schedule + x.schedule,
                                  default=self.default+x.default)

    def max(self, x):
        if isinstance(x, float) or isinstance(x, int):
            return ScheduledFloat(self.schedule.max(x),
                                  default=self.default)
        else:
            return ScheduledFloat(self.schedule.max(x.schedule),
                                  default=max(self.default, x.default))

FloatLike = Union[float, ScheduledFloat]
    

class CutoffEstimator:
    """
    Estimates cutoffs of an arbitrary numerical quantity such that a specified
    proportion of items will be above the cutoff on average.

      p is the proportion of items that should be above the cutoff.
    """
    def __init__(self, p: float):
        self.p = p
        # total count of items
        self.count = 0
        # total count of items that were above the cutoff
        self.count_above = 0
        # initial cutoff value
        self.cutoff = 0


    def __call__(self, x: float) -> bool:
        """
        Returns true if x is above the cutoff.
        """
        ans = (x > self.cutoff)
        self.count += 1
        if ans:
            self.count_above += 1
        cur_p = self.count_above / self.count
        delta_p = cur_p - self.p
        if (delta_p > 0) == ans:
            q = abs(delta_p)
            self.cutoff = x * q + self.cutoff * (1-q)
        return ans



class BalancerFunction(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx,
            x: Tensor,
            min_mean: float,
            max_mean: float,
            min_rms: float,
            max_rms: float,
            grad_scale: float,
            channel_dim: int,
    ) -> Tensor:
        if channel_dim < 0:
            channel_dim += x.ndim
        ctx.channel_dim = channel_dim
        ctx.save_for_backward(x)
        ctx.config = (min_mean, max_mean, min_rms, max_rms, grad_scale, channel_dim)
        return x


    @staticmethod
    def backward(
        ctx, x_grad: Tensor
    ) -> Tuple[Tensor, None, None, None, None, None]:
        x, = ctx.saved_tensors
        (min_mean, max_mean, min_rms, max_rms, grad_scale, channel_dim) = ctx.config


        try:
            with torch.enable_grad():
                with torch.cuda.amp.autocast(enabled=False):
                    x = x.to(torch.float32)
                    x = x.detach()
                    x.requires_grad = True
                    mean_dims = [ i for i in range(x.ndim) if i != channel_dim ]
                    uncentered_var = (x ** 2).mean(dim=mean_dims, keepdim=True)
                    mean = x.mean(dim=mean_dims, keepdim=True)
                    stddev = (uncentered_var - (mean * mean)).clamp(min=1.0e-20).sqrt()
                    rms = uncentered_var.clamp(min=1.0e-20).sqrt()

                    m = mean / stddev
                    # part of loss that relates to mean / stddev
                    m_loss = (m - m.clamp(min=min_mean, max=max_mean)).abs()

                    # put a much larger scale on the RMS-max-limit loss, so that if both it and the
                    # m_loss are violated we fix the RMS loss first.
                    rms_clamped = rms.clamp(min=min_rms, max=max_rms)
                    r_loss = (rms_clamped / rms).log().abs()

                    loss = (m_loss + r_loss)

                    if random.random() < 0.1:
                        logging.info(f"balancer loss={loss}")

                    loss.backward(gradient=torch.ones_like(loss))
                    loss_grad = x.grad
                    loss_grad_rms = (loss_grad ** 2).mean(dim=mean_dims, keepdim=True).sqrt().clamp(min=1.0e-20)

                    loss_grad = loss_grad * (grad_scale / loss_grad_rms)

                    x_grad_float = x_grad.to(torch.float32)
                    # scale each element of loss_grad by the absolute value of the corresponding
                    # element of x_grad, which we view as a noisy estimate of its magnitude for that
                    # (frame and dimension).  later we can consider factored versions.
                    x_grad_mod = x_grad_float + (x_grad_float.abs() * loss_grad)
                    x_grad = x_grad_mod.to(x_grad.dtype)
        except Exception as e:
            logging.info(f"Caught exception in Balancer backward: {e}, size={list(x_grad.shape)}, will continue.")

        return x_grad, None, None, None, None, None, None


class Balancer(torch.nn.Module):
    """
    Modifies the backpropped derivatives of a function to try to encourage, for
    each channel, that it is positive at least a proportion `threshold` of the
    time.  It does this by multiplying negative derivative values by up to
    (1+max_factor), and positive derivative values by up to (1-max_factor),
    interpolated from 1 at the threshold to those extremal values when none
    of the inputs are positive.

    Args:
           num_channels: the number of channels
           channel_dim: the dimension/axis corresponding to the channel, e.g.
               -1, 0, 1, 2; will be interpreted as an offset from x.ndim if negative.
           min_positive: the minimum, per channel, of the proportion of the time
               that (x > 0), below which we start to modify the derivatives.
           max_positive: the maximum, per channel, of the proportion of the time
               that (x > 0), above which we start to modify the derivatives.
           scale_gain_factor: determines the 'gain' with which we increase the
              change in gradient once the constraints on min_abs and max_abs
              are violated.
           min_abs:  the minimum average-absolute-value difference from the mean
               value per channel, which we allow, before we start to modify
               the derivatives to prevent this.
           max_abs:  the maximum average-absolute-value difference from the mean
               value per channel, which we allow, before we start to modify
               the derivatives to prevent this.
         prob: determines the minimum probability with which we modify the
             gradients for the {min,max}_positive and {min,max}_abs constraints,
             on each forward().  This is done randomly to prevent all layers
             from doing it at the same time.
    """
    def __init__(
            self,
            num_channels: int,
            channel_dim: int,
            min_positive: FloatLike = 0.05,
            max_positive: FloatLike = 0.95,
            min_abs: FloatLike = 0.2,
            max_abs: FloatLike = 100.0,
            grad_scale: FloatLike = 0.04,
            prob: Optional[FloatLike] = None,
    ):
        super().__init__()

        if prob is None:
            prob = ScheduledFloat((0.0, 0.5), (8000.0, 0.125), default=0.4)
        self.prob = prob
        # 5% of the time we will return and do nothing because memory usage is
        # too high.
        self.mem_cutoff = CutoffEstimator(0.05)

        # actually self.num_channels is no longer needed except for an assertion.
        self.num_channels = num_channels
        self.channel_dim = channel_dim
        self.min_positive = min_positive
        self.max_positive = max_positive
        self.min_abs = min_abs
        self.max_abs = max_abs
        self.grad_scale = grad_scale

    def forward(self, x: Tensor) -> Tensor:
        if (torch.jit.is_scripting() or not x.requires_grad or
            (x.is_cuda and self.mem_cutoff(torch.cuda.memory_allocated()))):
            return _no_op(x)

        prob = float(self.prob)
        if random.random() < prob:
            # The following inner-functions convert from the way we historically specified
            # these limitations, as limits on the absolute value and the proportion of positive
            # values, to limits on the RMS value and the (mean / stddev).
            def _abs_to_rms(x):
                # for normally distributed data, if the expected absolute value is x, the
                # expected rms value will be sqrt(pi/2) * x.
                return 1.25331413732 * x
            def _proportion_positive_to_mean(x):
                def _atanh(x):
                    eps = 1.0e-10
                    # eps is to prevent crashes if x is exactly 0 or 1.
                    # we'll just end up returning a fairly large value.
                    return (math.log (1+x+eps) - math.log (1-x+eps)) / 2.
                def _approx_inverse_erf(x):
                    # 1 / (sqrt(pi) * ln(2)),
                    # see https://math.stackexchange.com/questions/321569/approximating-the-error-function-erf-by-analytical-functions
                    # this approximation is extremely crude and gets progressively worse for
                    # x very close to -1 or +1, but we mostly care about the "middle" region
                    # e.g. _approx_inverse_erf(0.05) = 0.0407316414078772,
                    # and math.erf(0.0407316414078772) = 0.045935330944660666,
                    # which is pretty close to 0.05.
                    return 0.8139535143 * _atanh(x)
                # first convert x from the range 0..1 to the range -1..1 which the error
                # function returns
                x = -1 + (2 * x)
                return _approx_inverse_erf(x)

            min_mean = _proportion_positive_to_mean(float(self.min_positive))
            max_mean = _proportion_positive_to_mean(float(self.max_positive))
            min_rms = _abs_to_rms(float(self.min_abs))
            max_rms = _abs_to_rms(float(self.max_abs))
            grad_scale = float(self.grad_scale)
            # logging.info(f"min_mean={min_mean}, max_mean={max_mean}, min_rms={min_rms}, max_rms={max_rms}")

            assert x.shape[self.channel_dim] == self.num_channels

            return BalancerFunction.apply(
                x, min_mean, max_mean, min_rms, max_rms, grad_scale, self.channel_dim
            )
        else:
            return _no_op(x)


class BypassModule(nn.Module):
    """
    An nn.Module that implements a learnable bypass scale, and also randomized per-sequence
    layer-skipping.  The bypass is limited during early stages of training to be close to
    "straight-through", i.e. to not do the bypass operation much initially, in order to
    force all the modules to learn something.
    """
    def __init__(
            self,
            embed_dim: int,
            skip_rate: FloatLike = 0.0,
            straight_through_rate: FloatLike = 0.0,
            scale_min: FloatLike = ScheduledFloat((0.0, 0.9), (20000.0, 0.2), default=0),
            scale_max: FloatLike = 1.0,
            ):
        super().__init__()
        self.bypass_scale = nn.Parameter(torch.full((embed_dim,), 0.5))
        self.skip_rate = copy.deepcopy(skip_rate)
        self.straight_through_rate = copy.deepcopy(straight_through_rate)
        self.scale_min = copy.deepcopy(scale_min)
        self.scale_max = copy.deepcopy(scale_max)

    def embed_dim(self):
        return self.bypass_scale.numel()

    def _get_bypass_scale(self, batch_size: int):
        # returns bypass-scale of shape (num_channels,),
        # or (batch_size, num_channels,).  This is actually the
        # scale on the non-residual term, so 0 correponds to bypassing
        # this module.
        if torch.jit.is_scripting() or not self.training:
            return self.bypass_scale
        else:
            ans = limit_param_value(self.bypass_scale,
                                    min=float(self.scale_min),
                                    max=float(self.scale_max))
            skip_rate = float(self.skip_rate)
            if skip_rate != 0.0:
                mask = torch.rand((batch_size, 1), device=ans.device) > skip_rate
                ans = ans * mask
                # now ans is of shape (batch_size, num_channels), and is zero for sequences
                # on which we have randomly chosen to do layer-skipping.
            straight_through_rate = float(self.straight_through_rate)
            if straight_through_rate != 0.0:
                mask = torch.rand((batch_size, 1), device=ans.device) < straight_through_rate
                ans = torch.maximum(ans, mask.to(ans.dtype))

            return ans

    def forward(self,
                src_orig: Tensor,
                src: Tensor):
        """
        Args: src_orig and src are both of shape (seq_len, batch_size, num_channels)
        Returns: something with the same shape as src and src_orig
        """
        bypass_scale = self._get_bypass_scale(src.shape[1])
        return src_orig + (src - src_orig)  * bypass_scale


class LearnedDownsamplingModule(nn.Module):
    """
    Module that allows you to choose which frames to keep for transformer-type
    modules.  Effectively downsampling, but not necessarily "evenly"- you just
    keep some proportion of frames determined by the embedding.

    Args:
      embed_dim: embedding dimension
      downsampling_factor:  factor to downsample by, e.g. 2 or 4.  There is no
         fundamental reason why this has to be an integer, but we make it so
         anyway.
    """
    def __init__(self,
                 embed_dim: int,
                 downsampling_factor: int,
                 grad_scale: float):
        assert downsampling_factor > 1

        super().__init__()

        self.to_scores = nn.Linear(embed_dim, 1, bias=False)
        self.to_scores.lr_scale = 0.5
        # score_balancer is just to keep the magnitudes of the scores in
        # a fixed range and keep them balanced around zero, to stop
        # these drifting around.
        # largish range used to keep grads relatively small and avoid overflow in grads.
        min_positive = 1 / downsampling_factor - 0.05
        # min_positive = 1 /(2*downsampling_factor)
        max_positive = 1 / downsampling_factor + 0.05
        logging.info('downsampling_factor {}'.format(downsampling_factor))
        logging.info('grad_scale {}'.format(grad_scale))
        self.score_balancer = Balancer(1, channel_dim=-1,
                                       min_positive=min_positive,
                                       max_positive=max_positive,
                                       min_abs=1.0,
                                       max_abs=4.0,
                                       grad_scale=grad_scale,
                                       prob=ScheduledFloat((0.0, 1.0), (8000.0, 0.25), default=0.0))


        # below are for diagnostics.
        self.copy_weights1 = nn.Identity()
        self.copy_weights2 = nn.Identity()

        self.downsampling_factor = downsampling_factor


    def forward(self,
                x: Tensor,
                padding_mask: Tensor=None) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
          x: a Tensor of shape (seq_len, batch_size, embed_dim)
          paddint_mask: a Tensor of shape (batch_size, seq_len)

        Returns: (frame_indexes, weights, kept)

         frame_indexes: a Tensor of integer type, of shape (batch_size, reduced_seq_len)
              where reduced_seq_len = (seq_len + d - 1) // d.  It contains elements
              0 <= frame_indees < seq_len, in sorted (increasing) order

            weights: a Tensor of shape (batch_size, reduced_seq_len),
                 corresponding to the kept frames; these will be between 0 and 1, but
                 mostly exactly 1.
        """
        (seq_len, batch_size, _) = x.shape
        
        scores = self.to_scores(x)  # (seq_len, batch_size, 1)
        scores = self.score_balancer(scores)

        scores = scores.squeeze(-1).t()  # (batch_size, seq_len)

        # test mode.  because the sequence might be short, we keep all nonzero scores;
        # and there is no need for any penalty.
        weights = scores.clamp(min=0.0, max=1.0)
        mask = weights > 0.0
        if padding_mask is not None:
            mask = mask & (~padding_mask)
        # The per-sample lengths we will keep
        count = mask.to(torch.int32).sum(dim=-1)
        # The columns we will keep
        indexes = mask.nonzero(as_tuple=True)[1]
        indexes = indexes.split(count.tolist())
        # Padding with the last elements, e.g., with index=seq_len-1
        indexes = torch.nn.utils.rnn.pad_sequence(
            indexes, batch_first=True, padding_value=seq_len - 1
        )  # (batch_size, seq_len_reduced)

        weights = torch.gather(weights, dim=-1, index=indexes)

        padding_mask = make_pad_mask(count)
        weights = weights.masked_fill(padding_mask, 0.0)

        # if random.random() < 0.9:
        seq_len_reduced = indexes.shape[1]
        logging.info(f"seq_len={seq_len}, seq_len_reduced={seq_len_reduced}")

        x_downsampled = self.downsample(x, indexes)
        return indexes, weights, x_downsampled


    def downsample(self, x: Tensor, indexes: Tensor) -> Tensor:
        """
        Downsamples x via indexing with the indexes obtained from the
        forward() function.

        Args:
           x: tensor of shape (seq_len, batch_size, num_channels)
         indexes: integer indexes of shape (batch_size, seq_len_reduced), with elements
                 0 <= indexes < seq_len.
        Returns:
           x_downsampled, of shape (seq_len_reduced, batch_size, num_channels)
        """
        indexes_expanded = indexes.t().unsqueeze(-1).expand(-1, -1, x.shape[-1])
        # indexe_expanded: (seq_len_reduced, batch_size, num_channels)
        ans = torch.gather(x, dim=0, index=indexes_expanded)

        if __name__ == '__main__':
            # temp, for testing
            x_reconstructed = self.upsample(x, ans, indexes)
            assert torch.allclose(x, x_reconstructed)

        return ans

    @staticmethod
    def upsample(x_orig: Tensor, x: Tensor, indexes: Tensor,
                 weights: Optional[Tensor] = None) -> Tensor:
        """
        Upsamples, reversing the downsample() operation and filling in
        any not-chosen frames with their original value before downsampling
        (or with whatever x_orig contains).

        Args:
            x_orig: (seq_len, batch_size, num_channels)
            x: (seq_len_reduced, batch_size, num_channels)
          indexes: (batch_size, seq_len_reduced), contains original frame indexes
          weights: optional tensor of shape (batch_size, seq_len_reduced)

        Downsamples x via indexing with the indexes obtained from the
        forward() function.

        Args:
            x: tensor of shape (seq_len, batch_size, indexes)
         weights: a tensor of shape (batch_size, seq_len_reduced) containing weights between
             0 and 1, where 1 means fully use this x value and 0 means use x_orig
         indexes: integer indexes of shape (batch_size, seq_len_reduced), with elements
                 0 <= indexes < seq_len.
        """
        (seq_len, batch_size, num_channels) = x_orig.shape

        x_weight = 1.0 if weights is None else weights.t().unsqueeze(-1)
        # x_weight: (seq_len_reduced, batch_size, 1) if a tensor

        orig_x_weight = torch.ones(batch_size, seq_len,
                                   device=x.device, dtype=x.dtype)
        if weights is None:
            orig_x_weight.scatter_(dim=1, index=indexes, value=0.)
        else:
            orig_x_weight.scatter_(dim=1, index=indexes,
                                   src=(1. - weights).to(x.dtype))

        indexes = indexes.t().unsqueeze(-1).expand(-1, batch_size, num_channels)
        # indexes now: (seq_len_reduced, batch_size, num_channels)

        ans = torch.zeros_like(x_orig)

        ans.scatter_(dim=0, index=indexes, src=(x * x_weight))

        # add in x_orig in the frames that were not originally kept.
        return ans + x_orig * orig_x_weight.t().unsqueeze(-1)

def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float("-inf")).type_as(t)


class DownsampledDecoder(nn.Module):
    def __init__(self, 
                 num_embeddings,
                 padding_idx,
                 decoder_embed_dim = 2048,
                 decoder_ffn_embed_dim = 2048*4,
                 decoder_attention_heads = 32,
                 no_scale_embedding = True,
                 dropout = 0.1,
                 activation_fn = 'gelu',
                 decoder_layers = 24,
                 inference = False,
                 checkpoint_activations = False, 
                 offload_activations = False, 
                 min_params_to_wrap = 0,
                 bias = False,
                 qk_bias = False,
                 attn_type = "flash",
                 norm_name = "rmsnorm",
                 mlp_name = "swiglu",
                 tokens_per_sample = 2048,
                 model_parallel_size = 1,
                 args = None,
                 initialize_params_on_gpu = False,
                 tie_weights = True,
                 dtype = torch.float16,
                 device: Optional[torch.device] =None,
                 structure: str = None,
                 downsampling_factor: Tuple[int, ...] = (2,),
                 grad_scale: float = 0.04,
                 **kwargs) -> None:
        super().__init__()
        self.args = args
        self.inference = inference
        self.padding_idx = padding_idx
        embedding_dim = decoder_embed_dim
        heads = decoder_attention_heads
        intermediate_size = decoder_ffn_embed_dim
        initialize_params_on_gpu = True
        if initialize_params_on_gpu:
            device = torch.cuda.current_device() if initialize_params_on_gpu else device
        self._future_mask = torch.empty(0)
        self.output_embed_dim = embedding_dim
        self.embed_scale = 1.0 if no_scale_embedding else math.sqrt(embedding_dim)
        logger.info(f"embed scale: {self.embed_scale}")
        self.embed_tokens = Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
            dtype=dtype,
            initialize_params_on_gpu=initialize_params_on_gpu,
            model_parallel=True if model_parallel_size>1 else False
        )
        self.model_parallel = True if model_parallel_size>1 else False
        self.dropout_module = Dropout(
            dropout, module_name=self.__class__.__name__
        )
        
        self.layers = nn.ModuleList([])
        downsamplers = nn.ModuleList([])
        bypasses = nn.ModuleList([])
        
        layer_indexes = []
        downsampling_factors_list = []
        
        self.no_mask = False
        self.structure = structure
        num_downsamplers = len([s for s in structure if s == '('])
        downsampling_factor = list(downsampling_factor)
        logger.info(f"downsampling_factor: {downsampling_factor}")
        if len(downsampling_factor) == 1:
            downsampling_factor = downsampling_factor * num_downsamplers
        assert len(downsampling_factor) == num_downsamplers
        
        self.tokens_per_sample = tokens_per_sample

        for i in range(len(self.structure)):
            cur_structure = self.structure[i]
            if cur_structure == "T":
                layer = TransformerDecoderBlock(
                        embed_dim = embedding_dim,
                        intermediate_size=intermediate_size,
                        attention_heads=heads,
                        activation_fn=activation_fn,
                        layer_index=len(self.layers),
                        qk_bias = qk_bias,
                        tokens_per_sample = tokens_per_sample,
                        bias=bias,
                        attn_type=attn_type,
                        mlp_name=mlp_name,
                        norm_name=norm_name,
                        dtype=dtype,
                        initialize_params_on_gpu=initialize_params_on_gpu,
                        model_parallel_size=model_parallel_size
                    )
                if checkpoint_activations:
                    offload_to_cpu = offload_activations
                    assert not offload_to_cpu
                    layer = checkpoint_wrapper(
                        layer,
                        offload_to_cpu=offload_to_cpu
                    )
                if layer.no_mask and not self.no_mask:
                    self.no_mask = True
                layer_indexes.append(len(self.layers))
                self.layers.append(layer)
            elif cur_structure == "(":
                index = len(downsamplers)
                downsampler = LearnedDownsamplingModule(decoder_embed_dim, downsampling_factor[index], grad_scale=grad_scale)
                downsampling_factors_list.append(downsampling_factor[index])
                layer_indexes.append(len(downsamplers))
                downsamplers.append(downsampler)
            else:
                assert cur_structure == ")"
                bypass = BypassModule(decoder_embed_dim, straight_through_rate=0.0)
                layer_indexes.append(len(bypasses))
                bypasses.append(bypass)
                downsampling_factors_list.pop()


        logging.info('no_mask {}'.format(self.no_mask))
        assert len(self.layers) == decoder_layers

        self.downsamplers = downsamplers
        self.bypasses = bypasses
        self.layer_indexes = layer_indexes

        self.layer_norm = LayerNorm(embedding_dim, norm_name = norm_name)
        self.output_projection = nn.Linear(
                self.output_embed_dim, num_embeddings, bias=False, device=device, dtype=dtype
            )
        if tie_weights:
            self.output_projection.weight = self.embed_tokens.weight
        if initialize_params_on_gpu:
            device = torch.cuda.current_device() if initialize_params_on_gpu else None
            self.layer_norm.to(device).to(dtype)
        self.onnx_trace = False
        
    def get_embed_before_attn(self, prev_output_tokens):
        # 1. embedding
        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)
        x = self.dropout_module(x)
        x = x.transpose(0, 1).contiguous() #  T x B x H
        return x
    
    def _get_attn_offset(self, x: Tensor, src_key_padding_mask: Optional[Tensor]) -> Optional[Tensor]:
        """
        Return attention offset of shape (1 or batch_size, seq_len, seq_len), interpreted as (1 or batch_size, tgt_seq_len,
            src_seq_len); this reflects masking, if causal == True, otherwise will be all zeros.

        Args:
           x: embeddings after self.encoder_embed(), of shape (seq_len, batch_size, embed_dim).
         src_key_padding_mask: optional key-padding mask of shape (batch_size, seq_len) with True in masked positions.
        """
        batch_size, seq_len, _num_channels = x.shape

        ans = torch.zeros(1, seq_len, seq_len, device=x.device)

        
        # t is frame index, shape (seq_len,)
        t = torch.arange(seq_len, dtype=torch.int32, device=x.device)
        src_t = t
        tgt_t = t.unsqueeze(-1)
        attn_mask = (src_t > tgt_t)
        ans.masked_fill_(attn_mask, -1000)

        if src_key_padding_mask is not None:
            ans = ans.masked_fill(src_key_padding_mask.unsqueeze(1), -1000)
            # now ans: (batch_size, seq_len, seq_len).

        return ans

    def forward(
            self, 
            input_ids,
            start_pos=None,
            need_inner=None,
            use_cache=False,
            past_key_values=None,
            input_len = [],
            **kwargs,
            )->CausalLMOutputs:
        # x -> seq_len x bsz x hidden_dim
        x = self.get_embed_before_attn(input_ids)
     
        if past_key_values is not None:
            start_pos = past_key_values[0][0].shape[0]
        else:
            start_pos = 0

        seq_length_with_past = x.shape[0] + start_pos 
        query_length = x.shape[0]

        self_attn_padding_mask: Optional[Tensor] = None
        if input_ids.eq(self.padding_idx).any():
            self_attn_padding_mask = input_ids.eq(self.padding_idx)

        attn: Optional[Tensor] = None
        inner_states: Optional[List[Optional[Tensor]]] = None 
        if need_inner is not None:
            inner_states = [x]
        
        presents = None 
        if use_cache:
            presents = ()
            if past_key_values is None:
                past_key_values = tuple([None] * len(self.layer_indexes))

            self_attn_mask = self.buffered_future_mask(
                seq_length_with_past,  
                x.device,
                )[-query_length:, :]
        else:
            # The code `assert` in Python is used to check a condition. If the condition is true, the
            # program continues to execute normally. If the condition is false, an AssertionError is
            # raised, which will stop the program and provide information about the failure.
            assert False
            self_attn_mask = self.buffered_future_mask_for_tensor(x)
            
        downsample_info = [None,]
        curr_layer_idx = 0
        downsample_layer_num = 0
        for s, i in zip(self.structure, self.layer_indexes):
            layer_past = past_key_values[curr_layer_idx] if use_cache else None 
            if s == "T":
                layer = self.layers[i] # one transformer layer
                if downsample_info[-1] is None:
                    x, ret_present = layer(
                        x,
                        self_attn_mask=self_attn_mask,
                        self_attn_padding_mask=self_attn_padding_mask,
                        start_pos=start_pos,
                        use_cache=use_cache,
                        layer_past=layer_past,
                        input_len=input_len,
                        pos_index=None,
                    ) # (N,T,C)
                else:
                    if x.shape[0] >= 1:
                        indexes, _, _, _, _ = downsample_info[-1]
                        x, ret_present = layer(
                            x,
                            self_attn_mask=self_attn_mask,
                            self_attn_padding_mask=self_attn_padding_mask,
                            start_pos=start_pos,
                            use_cache=use_cache,
                            layer_past=layer_past,
                            input_len=input_len,
                            pos_index=indexes,
                        )
                    else:
                        ret_present = layer_past 

            elif s == "(":
                downsampler = self.downsamplers[i]
                # x -> seq_len, bsz, dim -> bsz, seq_len, dim
                
                x = x.transpose(0, 1)
                indexes, weights, x_new = downsampler(x.permute(1,0,2), self_attn_padding_mask,)
                downsample_info.append((indexes, weights, x, self_attn_mask, self_attn_padding_mask))
                # x = x_new.permute(1,0,2) # (N,T,C)
                x = x_new # reduced_seq_len x bsz x dim
                seq_len_reduced, bsz, _ = x.size()
                ret_present = None 
                if x.shape[0] >= 1:
                    # recompute self_attn_mask 
                    self_attn_mask = self.buffered_future_mask( 
                        seq_len_reduced,
                        x.device
                        )[-seq_len_reduced:, :]
                
                    self_attn_padding_mask = None 
                
            else:
                assert s == ")"
                indexes, weights, x_orig, self_attn_mask, self_attn_padding_mask = downsample_info.pop()
                # _attn_offset = attn_masks.pop()
                
                x = LearnedDownsamplingModule.upsample(x_orig.permute(1,0,2), x, indexes, weights)
                seq_len_upsampled, bsz, _ = x.size()
                ret_present = None 
                
                x = x.permute(1,0,2) # (bsz,seq_len,dim)
                bypass = self.bypasses[i]
                x = bypass(x_orig, x)
                x = x.transpose(0, 1) # seq_len x bsz x dim

            curr_layer_idx += 1

            if use_cache:
                presents = presents + (ret_present,)

            if need_inner is not None:
                inner_states.append(x)  

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        x_t = x.transpose(0, 1)
        x_t = self.output_layer(x_t).contiguous()

        # return x
        return CausalLMOutputs(
            logits=x_t,
            hidden_states=x,
            past_key_values=presents,
            inner_states=inner_states,
            attn=[attn],
        )

    def output_layer(self, features, **kwargs):
        x = self.output_projection(features)
        return x
    
    def buffered_future_mask(self, length, device):
        # self._future_mask.device != tensor.device is not working in TorchScript. This is a workaround.
        if (
                self._future_mask.size(0) == 0
                or (not self._future_mask.device == device)
                or self._future_mask.size(0) < length
        ):
            buffer_length = max(length + (8 - length % 8), 2048)
            self._future_mask = torch.triu(
                utils_fill_with_neg_inf(torch.zeros([buffer_length, buffer_length])), 1
            ).to(device)
        return self._future_mask[:length, :length]
    
    


def scaled_init_method_normal(num_layers, sigma=0.006, truncate_init=True):
    """Init method based on N(0, sigma/sqrt(2*num_layers)."""
    std = sigma / math.sqrt(2.0 * (num_layers + 1))

    def init_(tensor):
        if sigma <= 1e-8:  # effectively 0
            return torch.nn.init.zeros_(tensor)
        if truncate_init:
            return torch.nn.init.trunc_normal_(
                tensor, mean=0.0, std=std, a=-3 * std, b=3 * std
            )
        else:
            return torch.nn.init.normal_(tensor, mean=0.0, std=std)

    return init_


def init_method_normal(sigma=0.006, truncate_init=True):
    """Init method based on N(0, sigma)."""

    def init_(tensor):
        if sigma <= 1e-8:  # effectively 0
            return torch.nn.init.zeros_(tensor)
        if truncate_init:
            return torch.nn.init.trunc_normal_(
                tensor, mean=0.0, std=sigma, a=-3 * sigma, b=3 * sigma
            )
        else:
            return torch.nn.init.normal_(tensor, mean=0.0, std=sigma)

    return init_


class SubllmPretrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and
    a simple interface for downloading and loading pretrained models.
    """
    is_parallelizable = False
    supports_gradient_checkpointing = True
    config_class = SubllmConfig
    base_model_prefix = "transformer"
    _no_split_modules = ["TransformerDecoderBlock"]

    # _keys_to_ignore_on_load_missing = []
    # _keys_to_ignore_on_load_unexpected = []
    # _keys_to_ignore_on_save = []

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module: nn.Module):
        """Initialize the weights."""
        return

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, DownsampledDecoder):
            module.gradient_checkpointing = value


cross_entropy = CrossEntropyLoss(ignore_index=-100)


class SubllmModel(SubllmPretrainedModel):
    def __init__(self, config: SubllmConfig) -> None:
        super().__init__(config)
        self.config = config
        self.decoder = DownsampledDecoder(**config.to_dict())

    def get_input_embeddings(self):
        return self.decoder.embed_tokens

    def set_input_embeddings(self, new_embeddings: torch.Tensor):
        self.decoder.embed_tokens = new_embeddings

    def forward(self, input_ids, labels=None, **kwargs):
        input_len = (kwargs['attention_mask'] == 1).sum(dim=1)
        # input_len = input_len.tolist()
        outputs: CausalLMOutputs = self.decoder(input_ids, input_len=input_len, **kwargs)
        lm_logits = outputs.logits
        loss = None
        if labels is not None:
            backup_dtype = lm_logits.dtype
            lm_logits = lm_logits.to(torch.float32)
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss = cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            lm_logits = lm_logits.to(backup_dtype)
            loss = loss.to(backup_dtype)

        return CausalLMOutputsWithLoss(
            loss=loss,
            logits=lm_logits,
            hidden_states=outputs.hidden_states,
            past_key_values=outputs.past_key_values,
            attn=outputs.attn,
            inner_states=outputs.inner_states
        )

    def prepare_inputs_for_generation(
            self,
            input_ids: torch.LongTensor,
            past_key_values: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            inputs_embeds=None,
            use_cache=False,
            **kwargs
    ):
        if past_key_values is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update({
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "attention_mask": attention_mask,
        })
        return model_inputs

    def quantize(self, bits: int):
        try:
            from .quantizer import QLinear
        except ImportError:
            raise ImportError(
                f"Needs QLinear to run quantize."
            )

        for layer in self.decoder.layers:
            layer.self_attn.q_proj = QLinear(
                bits=bits,
                weight=layer.self_attn.q_proj.weight,
                bias = None,
            )
            layer.self_attn.k_proj = QLinear(
                bits=bits,
                weight=layer.self_attn.k_proj.weight,
                bias = None,
            )
            layer.self_attn.v_proj = QLinear(
                bits=bits,
                weight=layer.self_attn.v_proj.weight,
                bias = None,
            )
            layer.self_attn.out_proj = QLinear(
                bits=bits,
                weight=layer.self_attn.out_proj.weight,
                bias = None,
            )
            layer.feedforward.swiglu.w12 = QLinear(
                bits=bits,
                weight=layer.feedforward.swiglu.w12.weight,
                bias=None)
            layer.feedforward.swiglu.w3 = QLinear(
                bits=bits,
                weight=layer.feedforward.swiglu.w3.weight,
                bias=None)

        return self


    def _reorder_cache(self, past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past
