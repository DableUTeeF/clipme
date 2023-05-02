from transformers import AutoModel, CLIPProcessor
import torch
from PIL import Image
import time
import numpy as np

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
    model.eval()
    torch.onnx.export(model,
                  (inputs['input_ids'], inputs['pixel_values'], inputs['attention_mask']),
                  "clip.onnx",
                  export_params=True,
                  opset_version=14,
                  do_constant_folding=True,
                  input_names = ['input_ids', 'pixel_values', 'attention_mask'],
                  output_names = ['output'],
                  dynamic_axes={'input' : {0 : 'batch_size'},
                                'output' : {0 : 'batch_size'},
                                'pixel_values' : {0 : 'batch_size'},
                                'attention_mask' : {0 : 'batch_size'},
                                })

