import clip
import torch
import torch.nn as nn

class CLIP_Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.model, transform = clip.load(self.config.clip.model)

        if self.config.training.freeze_clip:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

        #this is a hacky way to get the embedding size - but it works!
        x = transform(torch.zeros(1, 3, 1, 1))
        with torch.no_grad():
            embedding = self.model.encode_image(x)
        self.embedding_size = embedding.shape[1:]

    def forward(self, x):
        return self.model.encode_image(x)
