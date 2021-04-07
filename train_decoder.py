import pytorch_lightning as pl
from pytorch_lightning import Trainer, LightningModule, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import clip
from torch.utils.data.dataset import Subset
import numpy as np
import torchvision
#import lpips
from lpips_pytorch import LPIPS
import argparse
import yaml
from utils import ImageNet_Train, ImageNet_Val, img_grid, yaml_config_hook, ImageNet_Train_Downsample
from models import Generator, Generator2, ResnetGenerator, ResnetGenerator_small
import os

class Produce_Embeddings:
    def __init__(self, args):
        super(Produce_Embeddings, self).__init__()
        self.clip = clip.load(args.clip_model, device=args.device, jit=False)[0].eval()

    def get_embeddings(self, x):
        with torch.no_grad():
            img_embeddings = self.clip.encode_image(x).float()
            #img_embeddings /= img_embeddings.norm(dim=-1, keepdim=True) #UNCOMMENT FOR L2 = 1
        
        return img_embeddings

class Clip_decoder(LightningModule):
    def __init__(self, args):
        super().__init__()
        
        self.hparams = args
        
        #The decoder which does img_embeddings --> images_hat
        #This is the one we shall train
        if args.decoder == "resnet":
            self.decoder = ResnetGenerator(nz=args.nz, ngf=args.ngf)
        elif args.decoder == "dcgan":
            self.decoder = Generator2(ngf = args.ngf, nz = args.nz)
        elif args.decoder == "resnet-small":
            self.decoder = ResnetGenerator_small(nz=args.nz, ngf=args.ngf)
        
        #The CLIP model which does images --> img_embeddings
        self.embedding_layer = Produce_Embeddings(args)
        
        #The mean and std of the CLIP model
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(args.device)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(args.device)

        if args.dataset == "Imagenet":
            self.classes = [281, 282, 283, 284, 285] #Cats
            #self.classes = np.arange(151, 269).tolist() #Dogs
            #self.classes = np.arange(200, 220) #Dogs small
        
        if args.loss == "lpips":
            #self.criterion = lpips.LPIPS(net='vgg')
            self.criterion = LPIPS(
                net_type='vgg',
                version='0.1'
            )
        else:
            self.criterion = nn.MSELoss(reduction='mean')
        
    def forward(self, x):
        """
        Given an image(s) x, get the clip image embeddings and then pass them through the decoder.
        """
        img_embeddings = self.embedding_layer.get_embeddings(x)
        
        return self.decoder(img_embeddings)
    
    def configure_optimizers(self):
        """
        Returns an optimizer for the decoder.
        """
        opt = torch.optim.Adam(self.decoder.parameters(), lr = self.hparams.lr)
        
        return opt
    
    def training_step(self, batch, batch_idx):
        """
        Given a bacth of images: (1) get the clip visual embeddings (2) pass these through the decoder (3) calculate loss
        """
        #here, x is a tuple of normalized and unnormalized images
        x, _ = batch
        
        x_hat = self.forward(x[0])
        
        if self.hparams.loss == "lpips":
            loss = self.criterion(x[1], x_hat).mean()
        else:
            loss = self.criterion(x[1], x_hat)

        if batch_idx == 0 and self.current_epoch%10 == 0:
            self.logger.experiment.add_image('Train_Sample', img_grid(x[1], x_hat), self.current_epoch)
        
        loss_dict = {"Train_Loss": loss}

        output = {
            'loss': loss,
            'progress_bar': loss_dict,
            'log': loss_dict
        }

        return output
    
    def validation_step(self, batch, batch_idx):
        x, _ = batch

        x_hat = self.forward(x[0])

        if self.hparams.loss == "lpips":
            loss = self.criterion(x[1], x_hat).mean()
        else:
            loss = self.criterion(x[1], x_hat)
        
        if batch_idx == 0:
            self.logger.experiment.add_image('Val_Sample', img_grid(x[1], x_hat), self.current_epoch)

        loss_dict = {
            "val_loss": loss
        }

        return loss_dict
    
    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()

        loss_dict = {
            'val_loss': val_loss_mean
        }

        output = {
            'val_loss_meaningless': val_loss_mean,
            'log': loss_dict,
            'progress_bar': loss_dict
        }

        return output

    def train_dataloader(self):
        """
        Returns a training dataloader - only ImageNet supported for now
        """
        if "small" in self.hparams.decoder:
            transform = ImageNet_Train_Downsample()
        else:
            transform = ImageNet_Train()

        train_dataset = torchvision.datasets.ImageNet(
            root = self.hparams.data_path,
            split = "train",
            transform = transform
        )

        inds_train = [i for i, label in enumerate(train_dataset.targets) if label in self.classes]
        
        train_dataloader = DataLoader(Subset(train_dataset, inds_train), batch_size=self.hparams.batch_size, num_workers=self.hparams.workers,\
                                        pin_memory=True, shuffle=True)
        
        return train_dataloader
    
    def val_dataloader(self):
        """
        Returns a validation dataloader
        """
        if "small" in self.hparams.decoder:
            transform = ImageNet_Train_Downsample()
        else:
            transform = ImageNet_Val()

        val_dataset = torchvision.datasets.ImageNet(
            root = self.hparams.data_path,
            split = "val",
            transform = transform
        )

        inds_val = [i for i, label in enumerate(val_dataset.targets) if label in self.classes]
        
        val_dataloader = DataLoader(Subset(val_dataset, inds_val), batch_size=self.hparams.batch_size, num_workers=self.hparams.workers,\
                                        pin_memory=True, shuffle=False)
        
        return val_dataloader
    
def train_decoder():
    parser = argparse.ArgumentParser(description="SimCLR")

    config = yaml_config_hook("./config/decode_config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()

    seed_everything(args.seed)

    model = Clip_decoder(args)

    trainer = Trainer.from_argparse_args(args)

    logger = TensorBoardLogger(
        save_dir= os.getcwd(),
        version=args.experiment_name,
        name='../Logs'
    )
    trainer.logger = logger

    trainer.fit(model)

if __name__ == "__main__":
    train_decoder()