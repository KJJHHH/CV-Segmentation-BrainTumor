from torchaudio.models import Conformer
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Conformer_Unet(nn.Module):
    def __init__(self, num_class, conformer = False, res = True):
        super(Conformer_Unet, self).__init__()
        
        """self.conv_init1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv_init2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv_init3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv_init4 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv_init5 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv_init6 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.maxpool_u = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)"""
        
        
        # =======
        # Unet
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(256)
        self.transconv5 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.transconv4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.transconv3 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.transconv2 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.transconv1 = nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.transbn5 = nn.BatchNorm2d(128)
        self.transbn4 = nn.BatchNorm2d(64)
        self.transbn3 = nn.BatchNorm2d(32)
        self.transbn2 = nn.BatchNorm2d(16)
        self.transbn1 = nn.BatchNorm2d(1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.unmaxpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(256 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_class)  
        self.softmax = nn.Softmax(dim=1)
        self.ln1 = nn.LayerNorm((1, 224, 224))
        
        # =======
        # Conformer
        self.con = conformer
        if conformer:
            self.conformer = Conformer(
                input_dim=224,
                num_heads=8,
                ffn_dim=128,
                num_layers=16,
                depthwise_conv_kernel_size=31)
        
        
        # =======
        # res
        self.res = res
    
    def MinMax(self, x):        
        max_ = x.max().item()
        min_ = x.min().item()
        x = (x - min_) / (max_ - min_)
        return x
    
    def forward(self, x):
        """
        x: batch, 1, 224, 224
        """
        # =======
        # CNN
        """
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        """
        
        # =======
        # Conformer
        """
        Input scale: (0, 255)
        Output scale: (0, 255)
        """
        if self.con:
            # x = self.ln1(x)
            x_i = x.clone()
            x_s = x.size()    
            x = x.view(x_s[0], x_s[1] * x_s[3], x_s[2])
            lengths = torch.tensor([x.shape[1] for i in range(len(x))]).to(device)
            x, len_ = self.conformer(x, lengths)
            x = x.view(x_s)
            x = x + x_i
            
        x = self.ln1(x)
        
        # U-net
        x0 = x.clone()
        h_0 = x.size()
        x, indices_1 = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x1 = x.clone()
        h_1 = x.size()        
        x, indices_2 = self.maxpool(self.relu(self.bn2(self.conv2(x))))
        x2 = x.clone()
        h_2 = x.size()
        x, indices_3 = self.maxpool(self.relu(self.bn3(self.conv3(x))))
        x3 = x.clone()
        h_3 = x.size()
        x, indices_4 = self.maxpool(self.relu(self.bn4(self.conv4(x))))
        x4 = x.clone()
        h_4 = x.size()
        x, indices_5 = self.maxpool(self.relu(self.bn5(self.conv5(x))))
        
        l = x.view(x.size(0), -1)  # Flatten the tensor
        l = self.fc1(l)
        l = self.relu(l)
        l = self.fc2(l)
        l = self.softmax(l)                
        
        x = self.unmaxpool(x, indices_5)
        x = self.transbn5(self.transconv5(x, output_size=h_4))
        x = x + x4
        x = self.relu(x)
        
        x = self.unmaxpool(x, indices_4)
        x = self.transbn4(self.transconv4(x, output_size=h_3))
        x = x + x3
        x = self.relu(x)
        
        x = self.unmaxpool(x, indices_3)
        x = self.transbn3(self.transconv3(x, output_size=h_2))
        x = x + x2
        x = self.relu(x)
        
        x = self.unmaxpool(x, indices_2)
        x = self.transbn2(self.transconv2(x, output_size=h_1))
        x = x + x1
        x = self.relu(x)
        
        x = self.unmaxpool(x, indices_1)
        x = self.transbn1(self.transconv1(x, output_size=h_0))
        x = x + x0
        x = self.relu(x)
        if self.con:
            x = x + x_i
        x = self.MinMax(x)
        return x, l

class Pretrain_Unet_Medical(nn.Module):
    def __init__(self, num_class):
        super(Pretrain_Unet_Medical, self).__init__()
        
        self.model_type = 'Pretrain_Unet_Medical'
        
        """self.conv_init1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv_init2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv_init3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv_init4 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv_init5 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv_init6 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.maxpool_u = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)"""
        
        
        # =======
        'self.conv_channel1to3 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)'
        self.conv_channel1to3 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1, stride=1, padding=0)
        self.ln_channel1to3 = nn.LayerNorm((3, 224, 224))        
        
        
        # =======
        self.pretrain = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=3, out_channels=1, init_features=32, pretrained=True)
        
        # =======
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(256)
        
        # =======
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.unmaxpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(256 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_class) 
        self.softmax = nn.Softmax(dim=1)
        
    def MinMax(self, x):        
        max_ = x.max().item()
        min_ = x.min().item()
        x = (x - min_) / (max_ - min_)
        return x
    
    def forward(self, x):
        """
        x: batch, 1, 224, 224
        """
        x = self.conv_channel1to3(x)
        x = self.ln_channel1to3(x)
        
        # =====
        # Pretrains
        x = self.pretrain(x)
        
        # =====
        # To Masks
        mask = x
        mask = self.MinMax(mask)
        # print(torch.max(mask), torch.min(mask))
        
        # =====
        # To Labels
        x = self.relu(self.maxpool(self.conv1(x)))
        x = self.relu(self.maxpool(self.conv2(x)))
        x = self.relu(self.maxpool(self.conv3(x)))
        x = self.relu(self.maxpool(self.conv4(x)))
        x = self.relu(self.maxpool(self.conv5(x)))
        x = x.view(x.size(0), -1)
        l = self.fc1(x)
        l = self.fc2(l)
        l = self.softmax(l)
        
        return mask, l