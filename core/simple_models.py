import torch
import torch.nn as nn
import torch.nn.functional as F

from core.base.modules import Activation



class SpectralEncoderDecoder(nn.Module):
    def __init__(self, in_ch, out_ch, num_filters=64, activation_name = 'softmax'):
        super(SpectralEncoderDecoder, self).__init__()

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.num_filters = num_filters
        self.activation_name = activation_name
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, num_filters, kernel_size=1),
            nn.ReLU(inplace=True),
            # nn.MaxPool3d(kernel_size=(2, 1, 1)),
            nn.Conv2d(num_filters, num_filters, kernel_size=1),
            # nn.MaxPool3d(kernel_size=(2, 1, 1)),
            nn.ReLU(inplace=True)            
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_filters, num_filters, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(num_filters, out_ch, kernel_size=1)
        )

        self.activation = Activation(activation_name)
        self.model_config = self.get_model_card()
    
    def get_model_card(self):
        card_dict = {
            'model_name': 'EncoderDecoder',
            'image_channels': self.in_ch,
            'global_pool': '',
            'model_mean': [0.0]*self.in_ch,
            'model_std': [1.0]*self.in_ch,

        }

        return card_dict
    def forward(self, x, apply_activation = True):
        # Encoder
        x1 = self.encoder(x)
        
        # Decoder
        x2 = self.decoder(x1)
        
        if apply_activation:
            x2 = self.activation(x2)

        return x2

class EncoderDecoder(nn.Module):
    def __init__(self, in_ch, out_ch, num_filters=64, activation_name = 'softmax'):
        super(EncoderDecoder, self).__init__()

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.num_filters = num_filters
        self.activation_name = activation_name
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, num_filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(num_filters, num_filters, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(num_filters, out_ch, kernel_size=2, stride=2)
        )



        self.activation = Activation(activation_name)
        self.model_config = self.get_model_card()
    
    def get_model_card(self):
        card_dict = {
            'model_name': 'EncoderDecoder',
            'image_channels': self.in_ch,
            'global_pool': '',
            'model_mean': [0.0]*self.in_ch,
            'model_std': [1.0]*self.in_ch,

        }

        return card_dict
    def forward(self, x, apply_activation = True):
        # Encoder
        x1 = self.encoder(x)
        
        # Decoder
        x2 = self.decoder(x1)
        
        if apply_activation:
            x2 = self.activation(x2)

        # correct size on even images (specially on deploy)
        diffY = x.size()[2] - x2.size()[2]
        diffX = x.size()[3] - x2.size()[3]

        x3 = F.pad(x2, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        return x3
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x
class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Up, self).__init__()
        self.up_scale = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)

    def forward(self, x1, x2):
        x2 = self.up_scale(x2)

        diffY = x1.size()[2] - x2.size()[2]
        diffX = x1.size()[3] - x2.size()[3]

        x2 = F.pad(x2, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return x
class DownLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DownLayer, self).__init__()
        self.pool = nn.MaxPool2d(2, stride=2, padding=0)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(self.pool(x))
        return x
class UpLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpLayer, self).__init__()
        self.up = Up(in_ch, out_ch)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        a = self.up(x1, x2)
        x = self.conv(a)
        return x


class UNet(nn.Module):
    def __init__(self, in_ch, out_ch, activation_name = 'softmax'):
        super(UNet, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.activation_name = activation_name

        self.conv1 = DoubleConv(self.in_ch, 64)
        self.down1 = DownLayer(64, 128)
        self.down2 = DownLayer(128, 256)
        # self.down3 = DownLayer(256, 512)
        # self.down4 = DownLayer(512, 1024)
        # self.up1 = UpLayer(1024, 512)
        # self.up2 = UpLayer(512, 256)
        self.up3 = UpLayer(256, 128)
        self.up4 = UpLayer(128, 64)
        self.last_conv = nn.Conv2d(64, self.out_ch, 1)

        self.activation = Activation(activation_name)
        self.model_config = self.get_model_card()

    def forward(self, x, apply_activation = True):
        x1 = self.conv1(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        # x4 = self.down3(x3)
        # x5 = self.down4(x4)
        # x1_up = self.up1(x4, x5)
        # x2_up = self.up2(x3, x1_up)
        x3_up = self.up3(x2, x3)
        x4_up = self.up4(x1, x3_up)
        output = self.last_conv(x4_up)

        if apply_activation:
            output = self.activation(output)
        return output
    
    def get_model_card(self):
        card_dict = {
            'model_name': 'UNet',
            'image_channels': self.in_ch,
            'global_pool': '',
            'model_mean': [0.0]*self.in_ch,
            'model_std': [1.0]*self.in_ch,

        }
        return card_dict
