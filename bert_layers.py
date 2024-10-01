from transformers.models.distilbert.modeling_distilbert import MultiHeadSelfAttention, TransformerBlock, Transformer
from transformers.models.bert.modeling_bert import BertSelfAttention, BertAttention, BertLayer, BertEncoder
import math
import copy
import torch
import torch.nn as nn


class BertSelfAttentionPast(BertSelfAttention):
    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            layer_past=None,
            cache_query=False,
    ):
        mixed_query_layer = self.query(hidden_states)

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

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        if layer_past is not None:
            if cache_query:
                past_q = layer_past[2]
                query_layer = torch.cat((past_q, query_layer), dim=-2)

            past_k, past_v = layer_past[0], layer_past[1]
            key_layer = torch.cat((past_k, key_layer), dim=-2)
            value_layer = torch.cat((past_v, value_layer), dim=-2)

        if cache_query:
            present = torch.stack([key_layer, value_layer, query_layer])
        else:
            present = torch.stack([key_layer, value_layer])

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if layer_past is None and attention_mask is not None:
            attention_scores += attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs, present) if self.output_attentions else (context_layer, present)
        return outputs


class BertAttentionPast(BertAttention):
    def __init__(self, config):
        super().__init__(config)
        self.self = BertSelfAttentionPast(config)
        self.self.output_attentions = True

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            layer_past=None,
            cache_query=False,
    ):
        self_outputs = self.self(
            hidden_states, attention_mask, head_mask, encoder_hidden_states,
            encoder_attention_mask, layer_past, cache_query
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class BertLayerPast(BertLayer):
    def __init__(self, config):
        super().__init__(config)
        self.attention = BertAttentionPast(config)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            layer_past=None,
            cache_query=False
    ):
        self_attention_outputs = self.attention(hidden_states, attention_mask,
                                                head_mask, layer_past=layer_past,
                                                cache_query=cache_query)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]

        if self.is_decoder and encoder_hidden_states is not None:
            cross_attention_outputs = self.crossattention(
                attention_output, attention_mask, head_mask,
                encoder_hidden_states, encoder_attention_mask
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:]

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs
        return outputs


class BertEncoderPast(BertEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.output_past = getattr(config, 'output_past', True)
        self.output_hidden_states = True
        self.output_attentions = True
        self.layer = nn.ModuleList(
            [BertLayerPast(config) for _ in range(config.num_hidden_layers)])

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past=None,
            cache_query=False
    ):
        if past is None:
            past = [None] * len(self.layer)

        all_hidden_states = ()
        all_attentions = ()
        presents = ()

        for i, (layer_module, layer_past) in enumerate(zip(self.layer, past)):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states, attention_mask, head_mask[i], encoder_hidden_states,
                encoder_attention_mask, layer_past, cache_query
            )
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

            present = layer_outputs[-1]
            if self.output_past:
                presents = presents + (present,)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_past:
            outputs = outputs + (presents,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs


class MaskedMultiHeadSelfAttention(MultiHeadSelfAttention):
    def forward(self, query, key, value, layer_past=None,
                mask=None, head_mask=None):
        bs, q_length, dim = query.size()
        dim_per_head = self.dim // self.n_heads

        def shape(x):
            """ separate heads """
            return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)

        def unshape(x):
            """ group heads """
            return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)

        q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
        k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
        v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)
        if layer_past is not None:
            past_k, past_v = layer_past[0], layer_past[1]
            k = torch.cat((past_k, k), dim=-2)
            v = torch.cat((past_v, v), dim=-2)

        present = torch.stack([k, v])

        q = q / math.sqrt(dim_per_head)  # (bs, n_heads, q_length, dim_per_head)
        # (bs, n_heads, q_length, k_length)
        scores = torch.matmul(q, k.transpose(2, 3))
        if layer_past is None and mask is not None:
            scores += mask

        weights = nn.Softmax(dim=-1)(scores)  # (bs, n_heads, q_length, k_length)
        weights = self.dropout(weights)  # (bs, n_heads, q_length, k_length)

        # Mask heads if we want to
        if head_mask is not None:
            weights = weights * head_mask

        context = torch.matmul(weights, v)  # (bs, n_heads, q_length, dim_per_head)
        context = unshape(context)  # (bs, q_length, dim)
        context = self.out_lin(context)  # (bs, q_length, dim)

        if self.output_attentions:
            return context, present, weights
        else:
            return context, present


class MaskedTransformerBlock(TransformerBlock):
    def __init__(self, config):
        super().__init__(config)
        self.attention = MaskedMultiHeadSelfAttention(config)

    def forward(self, x, layer_past=None, attn_mask=None, head_mask=None):
        sa_output = self.attention(query=x, key=x, value=x, layer_past=layer_past,
                                   mask=attn_mask, head_mask=head_mask)
        if self.output_attentions:
            # (bs, seq_length, dim), (bs, n_heads, seq_length, seq_length)
            sa_output, sa_present, sa_weights = sa_output
        else:
            assert type(sa_output) == tuple
            sa_output, sa_present = sa_output
        sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)

        # Feed Forward Network
        ffn_output = self.ffn(sa_output)  # (bs, seq_length, dim)
        ffn_output = self.output_layer_norm(
            ffn_output + sa_output)  # (bs, seq_length, dim)

        output = (ffn_output, sa_present)
        if self.output_attentions:
            output = (sa_weights,) + output
        return output


class MaskedTransformer(Transformer):
    def __init__(self, config):
        super().__init__(config)
        self.output_past = getattr(config, 'output_past', True)
        layer = MaskedTransformerBlock(config)
        self.layer = nn.ModuleList(
            [copy.deepcopy(layer) for _ in range(config.n_layers)])

    def forward(self, x, past=None, attn_mask=None, head_mask=None):
        if past is None:
            past = [None] * len(self.layer)

        all_hidden_states = ()
        all_attentions = ()
        presents = ()

        hidden_state = x
        for i, (layer_module, layer_past) in enumerate(zip(self.layer, past)):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_state,)

            layer_outputs = layer_module(x=hidden_state, layer_past=layer_past,
                                         attn_mask=attn_mask, head_mask=head_mask[i])
            hidden_state = layer_outputs[-2]
            present = layer_outputs[-1]
            if self.output_past:
                presents = presents + (present,)

            if self.output_attentions:
                assert len(layer_outputs) == 3
                attentions = layer_outputs[0]
                all_attentions = all_attentions + (attentions,)
            else:
                assert len(layer_outputs) == 2

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_state,)

        outputs = (hidden_state,)
        if self.output_past:
            outputs = outputs + (presents,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs
