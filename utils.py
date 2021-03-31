from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomResizedCrop, RandomHorizontalFlip
import torchvision
import torch
import yaml

#Downsamples the training images to 64x64 before continuing!
class ImageNet_Train_Downsample:
    def __init__(self):

        self.training = Compose([
            Resize(256),
            CenterCrop(224),
            Resize(64),
            ToTensor(),
            Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        self.embedding = Compose([
            Resize(256),
            CenterCrop(224),
            ToTensor(),
            Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
        ])
    
    def __call__(self, x):
        return self.embedding(x), self.training(x)

#Return [transformed and normalised image, transformed image]
class ImageNet_Train:
    def __init__(self):
        self.normalize = Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])

        self.rescale = Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

        self.transform = Compose([
            RandomResizedCrop(224), #Uncomment this and comment the next two lines for some real ish 
            #Resize(256),
            #CenterCrop(224),
            RandomHorizontalFlip(),
            ToTensor()
        ])
    
    def __call__(self, x):
        return self.normalize(self.transform(x)), self.rescale(self.transform(x)) 

class ImageNet_Val:
    def __init__(self):
        self.normalize = Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])

        self.rescale = Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

        self.transform = Compose([
            Resize(256),
            CenterCrop(224),
            ToTensor()
        ])
    
    def __call__(self, x):
        return self.normalize(self.transform(x)), self.rescale(self.transform(x)) 

def img_grid(data, data_hat):
    data = data.cpu()[0:8]
    data_hat = data_hat.cpu()[0:8]

    display = torch.zeros((16, data.size(1), data.size(2), data.size(3)))
    display[0:8] = data
    display[8:16] = data_hat

    grid = torchvision.utils.make_grid(display, nrow=4, normalize=True)

    return grid

def yaml_config_hook(config_file):
    """
    Custom YAML config loader, which can include other yaml files (I like using config files
    insteaad of using argparser)
    """

    # load yaml files in the nested 'defaults' section, which include defaults for experiments
    with open(config_file) as f:
        cfg = yaml.safe_load(f)
        for d in cfg.get("defaults", []):
            config_dir, cf = d.popitem()
            cf = os.path.join(os.path.dirname(config_file), config_dir, cf + ".yaml")
            with open(cf) as f:
                l = yaml.safe_load(f)
                cfg.update(l)

    if "defaults" in cfg.keys():
        del cfg["defaults"]

    return cfg