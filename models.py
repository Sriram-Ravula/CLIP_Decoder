import torch.nn as nn
from non_local import NLBlockND

class Generator(nn.Module):
    def __init__(self, ngf, nz, nc = 3):
        super(Generator, self).__init__()
        
        self.ngf = ngf
        self.nz = nz
        self.nc = nc
        
        first_chans = nz // 49 + 1 #The number of channels to reshape the input code to 
        
        #First project the [N, nz] tensor to [N, 7*7*first_chans]
        self.project = nn.Linear(in_features=nz, out_features=7*7*first_chans, bias=False)
        
        #Here we must reshape the tensor to [N, first_chans, 7, 7]
        self.deepen = nn.Conv2d(first_chans, ngf * 16, kernel_size=1, stride=1, padding=0, bias=False)
        
        self.main = nn.Sequential(
            #Input: [N, ngf * 16, 7, 7]
            nn.ConvTranspose2d(ngf * 16, ngf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(),
            
            #Input: [N, ngf * 8, 14, 14]
            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(),
            
            #Input: [N, ngf * 4, 28, 28]
            nn.ConvTranspose2d(ngf * 4, ngf* 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(),
            
            #Input: [N, ngf * 2, 56, 56]
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(),
            
            #Input: [N, ngf, 112, 112]
            nn.ConvTranspose2d(ngf, nc, kernel_size=4, stride=2, padding=1, bias=False)
            
            #Output: [N, ngf, 224, 224]
        )

    def forward(self, x):
        N = x.size(0)
        
        x = self.project(x)
        x = self.deepen(x.view(N, -1, 7, 7))
        x = self.main(x)
        
        return x

class Generator2(nn.Module):
    def __init__(self, ngf, nz, nc = 3):
        super(Generator2, self).__init__()
        
        self.ngf = ngf
        self.nz = nz
        self.nc = nc

        #input: [N, nz] --> reshape [N, nz, 1, 1]
        
        #input: [N, nz, 1, 1] --> [N, 7*7* ngf*32, 1, 1]
        #self.input = nn.Conv2d(nz, 7 * 7 * ngf*32, kernel_size=1, stride=1, padding=0, bias=False)

        self.project = nn.Linear(in_features=nz, out_features=7*7*ngf*32, bias=False)

        #input: [N, 7*7*ngf*32, 1, 1] --> reshape [N, ngf*32, 7, 7]
        
        self.main = nn.Sequential(
            #Input: [N, ngf * 32, 7, 7]
            nn.BatchNorm2d(ngf * 32),
            nn.ReLU(),

            #Input: [N, ngf * 32, 7, 7]
            nn.Conv2d(ngf * 32, ngf * 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            
            #Input: [N, ngf * 16, 14, 14]
            nn.Conv2d(ngf * 16, ngf * 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            
            #Input: [N, ngf * 8, 28, 28]
            nn.Conv2d(ngf * 8, ngf* 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            
            #Input: [N, ngf * 4, 56, 56]
            nn.Conv2d(ngf * 4, ngf * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            
            #Input: [N, ngf * 2, 112, 112]
            nn.Conv2d(ngf * 2, ngf, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            
            #Input: [N, ngf, 224, 224]
            nn.Conv2d(ngf, nc, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh()

            #Output: [N, nc, 224, 224]
        )

    def forward(self, x):
        N = x.size(0)
        
        #x = self.input(x.view(N, -1, 1, 1))
        x = self.project(x)
        x = self.main(x.view(N, -1, 7, 7))
        
        return x


#Does Conv(Relu(BN(Conv(UP(Relu(BN(x))))))) + UP(x)
class ResBlockGenerator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlockGenerator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.xavier_uniform(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform(self.conv2.weight.data, 1.)

        self.model = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            self.conv1,
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            self.conv2
            )

        residual = []

        if stride != 1:
            residual += [nn.Upsample(scale_factor=2)]
        if in_channels != out_channels:
            residual += [nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)]

        self.bypass = nn.Sequential(*residual)

    def forward(self, x):
        return self.model(x) + self.bypass(x)

class ResnetGenerator(nn.Module):
    def __init__(self, nz, ngf, nc=3):
        super(ResnetGenerator, self).__init__()
        self.nz = nz
        self.ngf = ngf

        #Expect [N, nz] --> [N, 7 * 7 * ngf*32]
        self.dense = nn.Linear(self.nz, 7 * 7 * 32*ngf)

        #Expect [N, ngf, 224, 224] --> [N, 3, 224, 224]
        self.final = nn.Conv2d(ngf, nc, 3, stride=1, padding=1)

        nn.init.xavier_uniform(self.dense.weight.data, 1.)
        nn.init.xavier_uniform(self.final.weight.data, 1.)

        self.model = nn.Sequential(
            #Input: [32*ngf, 7, 7]
            ResBlockGenerator(32*ngf, 16*ngf, stride=2),
            #Input: [16*ngf, 14, 14]
            ResBlockGenerator(16*ngf, 8*ngf, stride=2),
            #Input: [8*ngf, 28, 28]
            ResBlockGenerator(8*ngf, 4*ngf, stride=2),
            #Input: [4*ngf, 56, 56]
            ResBlockGenerator(4*ngf, 2*ngf, stride=2),
            #Input: [2*ngf, 112, 112]
            NLBlockND(in_channels=2*ngf, dimension=2),
            ResBlockGenerator(2*ngf, ngf, stride=2),
            #Input: [ngf, 224, 224]
            nn.BatchNorm2d(ngf),
            nn.ReLU(),
            self.final,
            #Input: [3, 224, 224]
            nn.Tanh())

    def forward(self, z):
        N = z.size(0)

        x = self.dense(z)

        return self.model(x.view(N, -1, 7, 7))

class ResnetGenerator_small(nn.Module):
    def __init__(self, nz, ngf, nc=3):
        super(ResnetGenerator_small, self).__init__()
        self.nz = nz
        self.ngf = ngf

        #Expect [N, nz] --> [N, 4 * 4 * ngf*16]
        self.dense = nn.Linear(self.nz, 4 * 4 * 16*ngf)

        #Expect [N, ngf, 224, 224] --> [N, 3, 224, 224]
        self.final = nn.Conv2d(ngf, nc, 3, stride=1, padding=1)

        nn.init.xavier_uniform(self.dense.weight.data, 1.)
        nn.init.xavier_uniform(self.final.weight.data, 1.)

        self.model = nn.Sequential(
            #Input: [16*ngf, 4, 4]
            ResBlockGenerator(16*ngf, 8*ngf, stride=2),

            #Input: [8*ngf, 8, 8]
            ResBlockGenerator(8*ngf, 4*ngf, stride=2),

            #Input: [4*ngf, 16, 16]
            ResBlockGenerator(4*ngf, 2*ngf, stride=2),

            #Input: [2*ngf, 32, 32]
            NLBlockND(in_channels=2*ngf, dimension=2),
            ResBlockGenerator(2*ngf, ngf, stride=2),

            #Input: [ngf, 64, 64]
            nn.BatchNorm2d(ngf),
            nn.ReLU(),
            self.final,

            #Input: [3, 64, 64]
            nn.Tanh())

    def forward(self, z):
        N = z.size(0)

        x = self.dense(z)

        return self.model(x.view(N, -1, 4, 4))
