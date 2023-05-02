import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import timm
import tome
from transformers.models.clip.modeling_clip import CLIPEncoderLayer, CLIPAttention
from torch import nn
from transformers import AutoModel, CLIPProcessor
from PIL import Image
from tome.merge import bipartite_soft_matching, merge_source, merge_wavg
import torch
import time
import numpy as np


class ToMeCLIPAttention(CLIPAttention):
    def forward(
            self,
            hidden_states,
            attention_mask=None,
            causal_attention_mask=None,
            output_attentions=False):
        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        k_mean = key_states.mean(1)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        # apply the causal_attention_mask first
        if causal_attention_mask is not None:
            if causal_attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is"
                    f" {causal_attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + causal_attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if output_attentions:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, k_mean


class ToMeCLIPEncoderLayer(CLIPEncoderLayer):
    def _drop_path1(self, x):
        return self.drop_path1(x) if hasattr(self, "drop_path1") else self.drop_path(x)

    def _drop_path2(self, x):
        return self.drop_path2(x) if hasattr(self, "drop_path2") else self.drop_path(x)

    def forward(
            self,
            hidden_states,
            attention_mask,
            causal_attention_mask,
            output_attentions=False):
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        # `metric` should be (1, 50, 64) 
        hidden_states, attn_weights, metric  = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states

        r = self._tome_info['r']
        if r > 0:
            # Apply ToMe here
            merge, _ = bipartite_soft_matching(
                metric,
                r,
                self._tome_info["class_token"],
                self._tome_info["distill_token"],
            )
            hidden_states, _ = merge_wavg(merge, hidden_states, self._tome_info["size"])
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


if __name__ == '__main__':
    # # Load a pretrained model, can be any vit / deit model.
    # model = timm.create_model("vit_base_patch16_224", pretrained=True)
    # # Patch the model with ToMe.
    # tome.patch.timm(model)
    # # Set the number of tokens reduced per layer. See paper for details.
    # model.r = 16
    model = AutoModel.from_pretrained('openai/clip-vit-base-patch32')
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    image = Image.open('/home/palm/PycharmProjects/clipme/imgs/Turkish_Angora_in_Ankara_Zoo_(AOÇ).JPG')
    inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)

    ori = []
    for i in range(100):
        t = time.time()
        outputs = model(**inputs)
        if i > 10:
           ori.append(time.time() - t) 
    print('cpu:', np.mean(ori))

    model.cuda()
    ori = []
    for i in range(100):
        t = time.time()
        outputs = model(**inputs)
        if i > 10:
           ori.append(time.time() - t) 
    print('cuda:', np.mean(ori))

    model = AutoModel.from_pretrained('openai/clip-vit-base-patch32')
    model.r = 4
    model._tome_info = {
        "r": model.r,
        "size": None,
        "source": None,
        "trace_source": False,
        "prop_attn": True,
        "class_token": False,
        "distill_token": False,
    }

    for module in model.vision_model.modules():
        if isinstance(module, CLIPEncoderLayer):
            module.__class__ = ToMeCLIPEncoderLayer
            module._tome_info = model._tome_info
        elif isinstance(module, CLIPAttention):
            module.__class__ = ToMeCLIPAttention
            module._tome_info = model._tome_info

    merged = []
    for i in range(100):
        t = time.time()
        outputs = model(**inputs)
        if i > 10:
           merged.append(time.time() - t) 
        # print('merged:', time.time() - t)
    print('merged:', np.mean(merged))

    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    print(probs)
