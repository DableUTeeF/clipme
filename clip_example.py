import timm
import tome
from transformers.models.clip.modeling_clip import CLIPEncoderLayer, CLIPAttention
from torch import nn
from transformers import AutoModel, CLIPProcessor
from PIL import Image
from tome.merge import bipartite_soft_matching, merge_source, merge_wavg


class ToMeCLIPAttention(CLIPAttention):
    def forward(
            self,
            hidden_states,
            attention_mask=None,
            causal_attention_mask=None,
            output_attentions=False):
        bsz, tgt_len, embed_dim = hidden_states.size()


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
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )
        r = self._tome_info["r"].pop(0)
        if r > 0:
            # Apply ToMe here
            merge, _ = bipartite_soft_matching(
                metric,
                r,
                self._tome_info["class_token"],
                self._tome_info["distill_token"],
            )


if __name__ == '__main__':
    # # Load a pretrained model, can be any vit / deit model.
    # model = timm.create_model("vit_base_patch16_224", pretrained=True)
    # # Patch the model with ToMe.
    # tome.patch.timm(model)
    # # Set the number of tokens reduced per layer. See paper for details.
    # model.r = 16

    model = AutoModel.from_pretrained('openai/clip-vit-base-patch32')
    model.r = 0
    model._tome_info = {
        "r": model.r,
        "size": None,
        "source": None,
        "trace_source": False,
        "prop_attn": True,
        "class_token": False,
        "distill_token": False,
    }

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    image = Image.open('/home/palm/Pictures/Turkish-Angora-Cat-compressed-768x384.jpg')

    inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)

    outputs = model(**inputs)
