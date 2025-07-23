from typing import List
import torch
import torch.nn as nn
from utils import list_check

class CNNclassifier(nn.Module):
    def __init__(self, blocks: int):
        super().__init__()
        encoder_features = 32*(128**2)//(2**(blocks+1)) #simplified version of (img_size/(2**i))**2 x filter_no(2**(i-1))
        self._encoder = EncoderCNN(blocks)
        self._regression = mlpRegression(encoder_features,128,1)

    def forward(self,x):
        hidden = self._encoder(x)
        hidden = hidden.view(x.shape[0],-1)
        return self._regression(hidden)
    
class CNNclassifierWithProcess(nn.Module):
    def __init__(self, blocks: int):
        super().__init__()
        encoder_features = 32*(128**2)//(2**(blocks+1)) #simplified version of (img_size/(2**i))**2 x filter_no(2**(i-1))
        self._conv_encoder = EncoderCNN(blocks)
        self._mlp_encoder = mlpRegression(4,[16,64],128)
        self._regression = mlpRegression(encoder_features+128,128,1)

    def forward(self,proc,img):
        hidden_conv = self._conv_encoder(img)
        hidden_conv = hidden_conv.view(img.shape[0],-1)
        hidden_mlp = self._mlp_encoder(proc)
        hidden = torch.cat([hidden_mlp,hidden_conv],dim=-1)
        return self._regression(hidden)

    
class EncoderCNN(nn.Module):
    def __init__(self, blocks: int):
        super().__init__()
        filters = 32
        kernel_size = 3
        stride = 1
        padding = kernel_size // 2
        bias = False
        filter_list = [1] + [filters * (2**i) for i in range(blocks)]
        encoder_layers = []
        for in_c, out_c in zip(filter_list,filter_list[1:]):
            encoder_layers.append(ConvBlock(in_c,out_c,kernel_size,stride,padding,bias,BatchNorm=True))

        self._encoder_layers = nn.Sequential(*encoder_layers)

    def forward(self,x):
        return self._encoder_layers(x)

class ConvBlock(nn.Module):
    """Standard 2D convolution block"""
    def __init__(self,in_channels:int,out_channels:int,kernel:int,stride:int,pad:int,bias:bool = False,BatchNorm:bool = False,dropout:float=0):
        super().__init__()
        block_list:List = [
            nn.Conv2d(in_channels,out_channels,kernel,stride,pad,bias=bias),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),]

        if BatchNorm:
            block_list.append(nn.BatchNorm2d(out_channels))
        if dropout > 0:
            block_list.append(nn.Dropout(dropout))        
        self.block = nn.Sequential(*block_list)

    def forward(self,x):
        return self.block(x)

class mlpRegression(nn.Module):
    def __init__(self,
                 in_neurons: int,
                 hidden_neurons: int|List[int],
                 out_neurons: int,
                 dropout: float = 0.,
                )->None:
        
        super().__init__()
        hidden_neurons = list_check(hidden_neurons)
        neurons = [in_neurons] + hidden_neurons
        layers = []

        for in_n,out_n in zip(neurons,neurons[1:]):
            layers.extend([
                nn.Linear(in_n,out_n),
                nn.ReLU(),
                nn.Dropout(dropout),])
        layers.append(nn.Linear(neurons[-1],out_neurons))
        self.model = nn.Sequential(*layers)
        
    def forward(self,x) -> torch.Tensor:
        x = self.model(x)
        return x
    
class UspampleConvBlock(nn.Module):
    """Custom "Deconv" block """
    def __init__(self,in_channels:int,out_channels:int,kernel:int,stride:int,pad:int,bias:bool = False,BatchNorm:bool = False,dropout:float=0):
        super().__init__()
        block_list:List = [
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels,out_channels,kernel,stride,pad,bias=bias),
            nn.ReLU(),]

        if BatchNorm:
            block_list.append(nn.BatchNorm2d(out_channels))
        if dropout > 0:
            block_list.append(nn.Dropout(dropout))        
        self.block = nn.Sequential(*block_list)        
    def forward(self,x):
        return self.block(x)
    
class DecoderCNN(nn.Module):
    def __init__(self, blocks: int):
        super().__init__()
        filters = 32
        kernel_size = 3
        stride = 1
        padding = kernel_size // 2
        bias = False
        filter_list = [filters * (2**i) for i in range(blocks)][::-1] #note removed [1]
        decoder_layers = []
        for in_c, out_c in zip(filter_list,filter_list[1:]):
            decoder_layers.append(UspampleConvBlock(in_c,out_c,kernel_size,stride,padding,bias,BatchNorm=True))
        decoder_layers.extend(
            [
            nn.Upsample(scale_factor=2),
            nn.Conv2d(filter_list[-1],1,kernel_size,stride,padding,bias=bias)
            ])
        self._decoder_layers = nn.Sequential(*decoder_layers)

    def forward(self,x):
        return self._decoder_layers(x)

class CNNAutoencoder(nn.Module):
    def __init__(self, blocks:int):
        super().__init__()
        self._encoder = EncoderCNN(blocks)
        self._decoder = DecoderCNN(blocks)
    
    def forward(self,x):
        x = self._encoder(x)
        return self._decoder(x)