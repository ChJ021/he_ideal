from __future__ import annotations

import math


def build_attention_wrapper(original_module, softmax_fn):
    class_name = original_module.__class__.__name__
    if class_name in {"MultiHeadSelfAttention", "DistilBertSdpaAttention"}:
        return DistilBertAttentionSoftmaxWrapper(original_module, softmax_fn)
    if class_name in {"BertSelfAttention", "BertSdpaSelfAttention"}:
        return BertSelfAttentionSoftmaxWrapper(original_module, softmax_fn)
    raise TypeError(f"Unsupported attention module for softmax replacement: {class_name}")


class DistilBertAttentionSoftmaxWrapper:
    def __new__(cls, original_module, softmax_fn):
        import torch

        class _Wrapper(torch.nn.Module):
            def __init__(self, original, replacement_softmax):
                super().__init__()
                self.original = original
                self.replacement_softmax = replacement_softmax

            def forward(
                self,
                query,
                key,
                value,
                mask,
                head_mask=None,
                output_attentions=False,
            ):
                import torch

                bs, q_length, dim = query.size()
                k_length = key.size(1)
                dim_per_head = self.original.dim // self.original.n_heads
                mask_reshp = (bs, 1, 1, k_length)

                def shape(x):
                    return x.view(bs, -1, self.original.n_heads, dim_per_head).transpose(1, 2)

                def unshape(x):
                    return (
                        x.transpose(1, 2)
                        .contiguous()
                        .view(bs, -1, self.original.n_heads * dim_per_head)
                    )

                q = shape(self.original.q_lin(query))
                k = shape(self.original.k_lin(key))
                v = shape(self.original.v_lin(value))
                q = q / math.sqrt(dim_per_head)
                scores = torch.matmul(q, k.transpose(2, 3))
                if mask.dim() == 2:
                    expanded_mask = (mask == 0).view(mask_reshp).expand_as(scores)
                    scores = scores.masked_fill(
                        expanded_mask,
                        torch.tensor(torch.finfo(scores.dtype).min, device=scores.device),
                    )
                elif mask.dim() == 4:
                    if mask.dtype == torch.bool:
                        expanded_mask = ~mask.expand_as(scores)
                        scores = scores.masked_fill(
                            expanded_mask,
                            torch.tensor(torch.finfo(scores.dtype).min, device=scores.device),
                        )
                    else:
                        scores = scores + mask.to(dtype=scores.dtype, device=scores.device)
                else:
                    raise ValueError(f"Unsupported DistilBERT attention mask shape: {tuple(mask.shape)}")
                weights = self.replacement_softmax(scores)
                weights = self.original.dropout(weights)
                if head_mask is not None:
                    weights = weights * head_mask
                context = torch.matmul(weights, v)
                context = unshape(context)
                context = self.original.out_lin(context)
                if output_attentions:
                    return (context, weights)
                return (context,)

        return _Wrapper(original_module, softmax_fn)


class BertSelfAttentionSoftmaxWrapper:
    def __new__(cls, original_module, softmax_fn):
        import torch

        class _Wrapper(torch.nn.Module):
            def __init__(self, original, replacement_softmax):
                super().__init__()
                self.original = original
                self.replacement_softmax = replacement_softmax

            def forward(
                self,
                hidden_states,
                attention_mask=None,
                head_mask=None,
                encoder_hidden_states=None,
                past_key_values=None,
                output_attentions=False,
                cache_position=None,
            ):
                import torch

                if encoder_hidden_states is not None or past_key_values is not None:
                    return self.original(
                        hidden_states,
                        attention_mask=attention_mask,
                        head_mask=head_mask,
                        encoder_hidden_states=encoder_hidden_states,
                        past_key_values=past_key_values,
                        output_attentions=output_attentions,
                        cache_position=cache_position,
                    )

                batch_size, _, _ = hidden_states.shape
                query_layer = self.original.query(hidden_states)
                query_layer = query_layer.view(
                    batch_size,
                    -1,
                    self.original.num_attention_heads,
                    self.original.attention_head_size,
                ).transpose(1, 2)

                key_layer = self.original.key(hidden_states)
                key_layer = key_layer.view(
                    batch_size,
                    -1,
                    self.original.num_attention_heads,
                    self.original.attention_head_size,
                ).transpose(1, 2)
                value_layer = self.original.value(hidden_states)
                value_layer = value_layer.view(
                    batch_size,
                    -1,
                    self.original.num_attention_heads,
                    self.original.attention_head_size,
                ).transpose(1, 2)

                attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
                attention_scores = attention_scores / math.sqrt(self.original.attention_head_size)
                if attention_mask is not None:
                    attention_scores = attention_scores + attention_mask
                attention_probs = self.replacement_softmax(attention_scores)
                attention_probs = self.original.dropout(attention_probs)
                if head_mask is not None:
                    attention_probs = attention_probs * head_mask
                context_layer = torch.matmul(attention_probs, value_layer)
                context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
                new_context_layer_shape = context_layer.size()[:-2] + (
                    self.original.all_head_size,
                )
                context_layer = context_layer.view(new_context_layer_shape)
                return context_layer, attention_probs

        return _Wrapper(original_module, softmax_fn)
