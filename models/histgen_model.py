import numpy as np
import torch
import torch.nn as nn
from modules.visual_extractor import VisualExtractor
from modules.histgen_module import BaseHistGen

class HistGenModel(nn.Module):
    def __init__(self, args, tokenizer):
        super(HistGenModel, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.encoder_decoder = BaseHistGen(args, tokenizer)
        self.wsi_mapping = torch.nn.Linear(768, self.args.d_vf) if "ctranspath" in args.image_dir else torch.nn.Linear(1024, self.args.d_vf)
        self.forward = self.forward_pathology
        self.visual_extractor = VisualExtractor(args)

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    # def forward_pathology(self, images, targets=None, mode='train', update_opts={}):
        
    #     att_feats = self.wsi_mapping(images)
    #     fc_feats = torch.mean(att_feats, dim=1)
        
    #     if mode == 'train':
    #         output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
    #         return output
    #     elif mode == 'sample':
    #         output, output_probs = self.encoder_decoder(fc_feats, att_feats, mode='sample')
    #         return output
    #     else:
    #         raise ValueError
    def forward_pathology(self, images, targets=None, mode='train', update_opts={}):
    # images shape: (batch_size, sequence_length, hidden_size)
    
    num_regions = self.args.num_regions  # Make sure you have this in your args
    region_size = self.args.region_size  # Make sure you have this in your args
    
    # Transform into regions (reshape)
    hidden_states_reshape = transform_tokens2regions(images, num_regions, region_size)
    # hidden_states_reshape shape: (batch_size * num_regions, region_size, hidden_size)

    # Flatten or pool region tokens to get 2D tensor (N, in_features)
    # Option A: Flatten region tokens (recommended if linear layer expects 1024 or 768 inputs)
    input_linear = hidden_states_reshape.view(hidden_states_reshape.size(0), -1)  # shape (N, region_size * hidden_size)
    
    # If the input feature size doesn't match wsi_mapping's expected input (768 or 1024),
    # adjust your region_size or hidden_size accordingly
    
    # Pass to linear layer
    att_feats = self.wsi_mapping(input_linear)  # shape (batch_size * num_regions, d_vf)

    # Reshape att_feats back to (batch_size, num_regions, d_vf)
    batch_size = images.size(0)
    att_feats = att_feats.view(batch_size, num_regions, -1)
    
    # Compute mean pooled features (fc_feats)
    fc_feats = torch.mean(att_feats, dim=1)  # shape (batch_size, d_vf)
    
    if mode == 'train':
        output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        return output
    elif mode == 'sample':
        output, output_probs = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        return output
    else:
        raise ValueError

