# @Author : guopeiming
# @Contact : guopeiming.gpm@{qq, gmail}.com
import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from typing import Tuple


ACT2FN = {"gelu": F.gelu, "relu": F.relu}


class SinusoidalPositionalEmbedding(nn.Module):
    """
    This module produces sinusoidal positional embeddings of any length.
    """
    def __init__(self, embedding_dim: int):
        super(SinusoidalPositionalEmbedding, self).__init__()
        assert embedding_dim % 2 == 0
        self.embedding_dim = embedding_dim

        self.emb = self._get_emb()  # [1, embedding_dim//2]

        self.reset_parameters()

    def forward(self, batch_size: int, seq_len: int, offset=0) -> torch.Tensor:
        """
        :param batch_size:
        :param seq_len:
        :param offset:
        """
        # [seq_len, embedding_dim//2]
        emb = torch.arange(offset, seq_len+offset, step=1, device=self.emb.device).unsqueeze(1) * self.emb
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], 1)  # [seq_len, embedding_dim]

        # zero pad
        if self.embedding_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(seq_len, 1)], 1)
        return emb.unsqueeze(0)  # [1, seq_len, embedding_dim]

    def _get_emb(self):
        half_dim = self.embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb).unsqueeze(0)
        # emb belongs to [1.0, 1e-4] because 10000
        return emb

    def reset_parameters(self):
        pass


class LearnedPositionalEmbedding(nn.Module):
    """
    This module produces LearnedPositionalEmbedding.
    """
    def __init__(self, embedding_dim: int, max_len: int = 512):
        super(LearnedPositionalEmbedding, self).__init__()
        self.embs = nn.Embedding(max_len, embedding_dim)

        self.reset_parameters()

    def forward(self, batch_size: int, seq_len: int, offset=0) -> torch.Tensor:
        """
        :param batch_size:
        :param seq_len:
        :param offset:
        """
        # [seq_len]
        positions = torch.add(torch.arange(seq_len, dtype=torch.long, device=self.embs.weight.device), offset)
        emb = self.embs(positions)
        return emb.unsqueeze(0)  # [1, seq_len, embedding_dim]

    def reset_parameters(self):
        init.xavier_normal_(self.embs.weight)


class MultiheadAttention(nn.Module):

    def __init__(self, hidden_size: int, nhead: int, attention_dropout: float, kqv_dim: int = 0):
        super(MultiheadAttention, self).__init__()
        assert hidden_size % nhead != 0 or kqv_dim != 0
        self.hidden_size = hidden_size
        self.nhead = nhead
        self.head_size = hidden_size // nhead if kqv_dim == 0 else kqv_dim

        self.query = nn.Linear(hidden_size, self.nhead*self.head_size)
        self.key = nn.Linear(hidden_size, self.nhead*self.head_size)
        self.value = nn.Linear(hidden_size, self.nhead*self.head_size)

        self.dropout = nn.Dropout(attention_dropout)

        self.multihead_proj = nn.Linear(self.nhead*self.head_size, self.hidden_size)

    def transpose_for_scores(self, x: torch.Tensor):
        new_x_shape = x.size()[:-1] + (self.nhead, self.head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None, head_mask: torch.Tensor = None,
                encoder_hidden_states: torch.Tensor = None, encoder_attention_mask: torch.Tensor = None,
                output_attentions: bool = False) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states)  # [batch_size, seq_len, hidden_size]

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)  # [batch_size, nhead, seq_len, head_size]
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.head_size)
        # [batch_size, nhead, seq_len, seq_len]
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in Transformer forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)  # [batch_size, nhead, seq_len, seq_len]

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.nhead*self.head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)  # [batch_size, seq_len, nhead*head_size]

        context_layer = self.multihead_proj(context_layer)  # [batch_size, seq_len, hidden_size]

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


class TransformerAttentionLayer(nn.Module):
    def __init__(self, hidden_size: int, nhead: int, hidden_dropout: float, attention_dropout: float, activation: str,
                 layer_norm_eps: float, kqv_dim: int = 0):
        super(TransformerAttentionLayer, self).__init__()
        self.attn = MultiheadAttention(hidden_size, nhead, attention_dropout, kqv_dim)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None, head_mask: torch.Tensor = None,
                encoder_hidden_states: torch.Tensor = None, encoder_attention_mask: torch.Tensor = None,
                output_attentions: bool = False) -> Tuple[torch.Tensor]:
        input_tensor = hidden_states
        attn_outputs = self.attn(
            hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask, output_attentions,
        )

        hidden_states = attn_outputs[0]
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        outputs = (hidden_states,) + attn_outputs[1:]  # add attentions if we output them
        return outputs


class TransformerFFNLayer(nn.Module):
    def __init__(self, hidden_size: int, dim_feedforward: int, hidden_dropout: float, activation: str,
                 layer_norm_eps: float):
        super(TransformerFFNLayer, self).__init__()
        self.dense1 = nn.Linear(hidden_size, dim_feedforward)
        assert activation in ACT2FN
        self.act_fn = ACT2FN[activation]

        self.dense2 = nn.Linear(dim_feedforward, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_tensor = hidden_states
        hidden_states = self.dense1(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.dense2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class TransformerLayer(nn.Module):
    def __init__(self, hidden_size: int, nhead: int, dim_feedforward: int, hidden_dropout: float,
                 attention_dropout: float, activation: str, layer_norm_eps: float, is_decoder: bool, kqv_dim: int = 0):
        super(TransformerLayer, self).__init__()
        self.selfattention = TransformerAttentionLayer(hidden_size, nhead, hidden_dropout, attention_dropout,
                                                       activation, layer_norm_eps, kqv_dim)
        self.is_decoder = is_decoder
        if self.is_decoder:
            self.crossattention = TransformerAttentionLayer(hidden_size, nhead, hidden_dropout, attention_dropout,
                                                            activation, layer_norm_eps, kqv_dim)
        self.ffnlayer = TransformerFFNLayer(hidden_size, dim_feedforward, hidden_dropout, activation, layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None, head_mask: torch.Tensor = None,
                encoder_hidden_states: torch.Tensor = None, encoder_attention_mask: torch.Tensor = None,
                output_attentions: bool = False) -> Tuple[torch.Tensor]:
        self_attention_outputs = self.selfattention(
            hidden_states, attention_mask, head_mask, output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        if self.is_decoder:
            assert encoder_hidden_states is not None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights

        layer_output = self.ffnlayer(attention_output)
        outputs = (layer_output,) + outputs
        return outputs


class Transformer(nn.Module):
    """
    Args:
        hidden_size: the number of expected features in the encoder/decoder inputs (default=512).
        nhead: the number of heads in the multiheadattention models (default=8).
        num_layers: the number of sub-layers (default=6).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        hidden_dropout: the dropout value (default=0.1).
        attention_dropout: the dropout value (default=0.1).
        activation: the activation function of encoder/decoder intermediate layer, relu or gelu (default=gelu).
        is_decoder: wheter to config transformer as decoders.
        kqv_dim: customed dimention of k, q, v.

    """
    def __init__(self, hidden_size: int = 512, num_layers: int = 3, nhead: int = 8, dim_feedforward: int = 1024,
                 hidden_dropout: float = 0.1, attention_dropout: float = 0.1, activation: str = "gelu",
                 layer_norm_eps: float = 1e-12, is_decoder: bool = False, kqv_dim: int = 0):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([TransformerLayer(hidden_size, nhead, dim_feedforward, hidden_dropout,
                                     attention_dropout, activation, layer_norm_eps, is_decoder, kqv_dim)
                                     for _ in range(num_layers)])

        self.num_layers = num_layers
        self.is_decoder = is_decoder

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None, head_mask: torch.Tensor = None,
                encoder_hidden_states: torch.Tensor = None, encoder_attention_mask: torch.Tensor = None,
                output_attentions: bool = False, output_hidden_states: bool = False) -> Tuple[torch.Tensor]:
        batch_size, seq_len, dim = hidden_states.shape
        all_hidden_states = ()
        all_attentions = ()

        # processing mask matrices before hidden states computing.
        # if mask dim is 2, it means whether there is a pad in input hidden states. [batch_size, seq_len]
        # if mask dim is 3, it means whether to attend the customized position.
        # [batch_size, (to_)seq_len, (from_)seq_len]
        # dim meaning is the same as all kinds of masks.

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask is not None:
            attention_mask = self.get_extended_attention_mask(attention_mask, (batch_size, seq_len))

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if encoder_attention_mask is not None:
            encoder_attention_mask = self.get_encoder_attention_mask(encoder_attention_mask)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask)

        # computation
        for i, layer_module in enumerate(self.layers):
            layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i], encoder_hidden_states,
                                         encoder_attention_mask, output_attentions)
            hidden_states = layer_outputs[0]

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            if output_attentions:
                all_attentions = all_attentions + layer_outputs[1:]

        outputs = (hidden_states,)
        if output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)

    def get_encoder_attention_mask(self, encoder_attention_mask: torch.Tensor) -> torch.Tensor:
        """type: torch.Tensor -> torch.Tensor"""
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        if encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
        encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=torch.float)

        encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -1e9

        return encoder_extended_attention_mask

    def get_extended_attention_mask(self, attention_mask: torch.Tensor, input_shape: Tuple) -> torch.Tensor:
        """Makes broadcastable attention mask and causal mask so that future and maked tokens are ignored.

        Arguments:
            attention_mask: torch.Tensor with 1 indicating tokens to ATTEND to
            input_shape: tuple, shape of input_ids

        Returns:
            torch.Tensor with dtype of attention_mask.dtype
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_len]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_len, seq_len]
            if self.is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=attention_mask.device)
                causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
                # causal and attention masks must have same type with pytorch version < 1.3
                causal_mask = causal_mask.to(attention_mask.dtype)
                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(torch.float)
        extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
        return extended_attention_mask

    def get_head_mask(self, head_mask: torch.Tensor) -> torch.Tensor:
        """
        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        attention_probs has shape bsz x n_heads x N x N
        Arguments:
            head_mask: torch.Tensor or None: has shape [num_heads] or [num_hidden_layers x num_heads]
        Returns:
             Tensor of shape shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
             or list with [None] for each layer
        """
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask, self.num_layers)
        else:
            head_mask = [None] * self.num_layers

        return head_mask

    def _convert_head_mask_to_5d(self, head_mask: torch.Tensor) -> torch.Tensor:
        """-> [num_hidden_layers x batch x num_heads x seq_length x seq_length]"""
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(self.num_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
        assert head_mask.dim() == 5, f"head_mask.dim != 5, instead {head_mask.dim()}"
        head_mask = head_mask.to(torch.float)  # switch to fload if need + fp16 compatibility
        return head_mask


class TransformerPooler(nn.Module):
    def __init__(self, hidden_size: int):
        super(TransformerPooler, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self.reset_parameters()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

    def reset_parameters(self):
        init.xavier_normal_(self.dense.weight, init.calculate_gain('Tanh'))
        init.xavier_uniform_(self.dense.bias, init.calculate_gain('Tanh'))


class SimpleTransformerLayer(nn.Module):
    """
    SimpleTransformerLayer is different from TransformerLayer.
    latter has one more multihead-attention layer than former.
    when k length is 1, it is important and necessary.
    """
    def __init__(self, hidden_size: int, nhead: int, dim_feedforward: int, hidden_dropout: float,
                 attention_dropout: float, activation: str, layer_norm_eps: float):
        super(SimpleTransformerLayer, self).__init__()
        self.crossattention = TransformerAttentionLayer(hidden_size, nhead, dim_feedforward, hidden_dropout,
                                                        attention_dropout, activation, layer_norm_eps)
        self.ffnlayer = TransformerFFNLayer(hidden_size, dim_feedforward, hidden_dropout, activation, layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None, head_mask: torch.Tensor = None,
                encoder_hidden_states: torch.Tensor = None, encoder_attention_mask: torch.Tensor = None,
                output_attentions: bool = False) -> Tuple[torch.Tensor]:
        encoder_attention_mask = encoder_attention_mask[:, None, None, :]
        cross_attention_outputs = self.crossattention(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            output_attentions,
        )
        attention_output = cross_attention_outputs[0]
        outputs = cross_attention_outputs[1:]  # add cross attentions if we output attention weights

        layer_output = self.ffnlayer(attention_output)
        outputs = (layer_output,) + outputs
        return outputs
