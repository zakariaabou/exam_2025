#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Architectures:
- UNet_causal_5mn_atrous (no terminal scaling step)
- UNet_causal_5mn_atrous_rescale (terminal scaling step with a unique parameter)
- UNet_causal_5mn_atrous_complex_rescale (terminal scaling step with 2-layers perceptron -additive form)
- UNet_causal_5mn_atrous_multiplicative_rescale (terminal scaling step with 2-layers perceptron -multiplicative form)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

###############################################################################################
################################  building blocks  ############################################


class double_conv_causal(nn.Module):
    '''(conv => BN => ReLU) * 2, with causal convolutions that preserve input size'''
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1):
        super(double_conv_causal, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=0, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=kernel_size, padding=0, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        # Apply causal padding manually for the first convolution
        x = F.pad(x, ((self.kernel_size - 1) * self.dilation, 0))  # Pad 2 on the left (time dimension), 0 on the right
        x = self.conv1(x)
        # testx
        x = self.bn1(x)
        x = self.relu1(x)

        # Apply causal padding manually for the second convolution
        x = F.pad(x, ((self.kernel_size - 1) * self.dilation, 0))   # Further padding to maintain causality
        x = self.conv2(x)
        # testx
        x = self.bn2(x)
        x = self.relu2(x)
        return x



class Down_causal(nn.Module):
    ''' Downscaling with maxpool then double conv '''
    def __init__(self, in_ch, out_ch, pooling=True, pooling_kernel_size=2, pooling_stride=2, dilation=1):
        super(Down_causal, self).__init__()
        if pooling:
            self.mpconv = nn.Sequential(
                nn.MaxPool1d(kernel_size=pooling_kernel_size, stride=pooling_stride),
                double_conv_causal(in_ch, out_ch)
            )
        else:
            self.mpconv = nn.Sequential(
                double_conv_causal(in_ch, out_ch,dilation=dilation)
            )           

    def forward(self, x):
        x = self.mpconv(x)
        return x



class Up_causal(nn.Module):
    ''' Upscaling then double conv, modified to maintain causality '''
    def __init__(self, in_ch, out_ch, kernel_size=2, stride=2):
        super(Up_causal, self).__init__()

        self.up = nn.ConvTranspose1d(in_ch, in_ch, kernel_size=kernel_size, stride=stride, padding=0, output_padding=0)

        self.conv = double_conv_causal(2 * in_ch, out_ch)  # Using causal convolution here

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Calculate necessary padding for concatenation to match the earlier layer's feature dimension
        diffX = x2.size()[2] - x1.size()[2]
        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2))  # Symmetric padding for alignment, maintain causality

        x = torch.cat([x1, x2], dim=1)  # Concatenate features from down-path
        x = self.conv(x)
        return x

class Up_causal2(nn.Module):
    ''' Upscaling then double conv, modified to maintain causality '''
    def __init__(self, in_ch, out_ch, kernel_size=2, stride=2):
        super(Up_causal2, self).__init__()

        self.up = nn.ConvTranspose1d(in_ch, in_ch, kernel_size=kernel_size, stride=stride, padding=0, output_padding=0)

        self.conv = double_conv_causal(3 * in_ch // 2, out_ch)  # Using causal convolution here

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Calculate necessary padding for concatenation to match the earlier layer's feature dimension
        diffX = x2.size()[2] - x1.size()[2]
        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2))  # Symmetric padding for alignment, maintain causality

        x = torch.cat([x1, x2], dim=1)  # Concatenate features from down-path
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, 1)
        

    def forward(self, x):
        x = self.conv(x)
        return x


# For ASPP

class _ConvBnReLU(nn.Module):
    """ Helper class that groups convolution, BN, and ReLU layers, modified to be causal. """
    def __init__(self, in_ch, out_ch, kernel_size, stride, dilation):
        super(_ConvBnReLU, self).__init__()
        # Initialize convolution without padding
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, stride, padding=0, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.kernel_size = kernel_size
        self.dilation = dilation

    def forward(self, x):
        # Calculate the required padding
        required_padding = (self.kernel_size - 1) * self.dilation

        # Apply padding manually to the left side only
        if required_padding != 0:
            x = F.pad(x, (required_padding, 0))  # (padding_left, padding_right)

        x = self.conv(x)
        # testx
        x = self.bn(x)
        x = self.relu(x)
        return x



class ASPP(nn.Module):
    """
    Atrous spatial pyramid pooling adapted for causal convolution, without image-level feature pooling.
    same number of channels at the end (needs inc_ch // (len(atrous_rates) + 1 )
    """
    def __init__(self, in_ch, rates):
        super(ASPP, self).__init__()
        self.stages = nn.ModuleList()
        self.out_ch = in_ch // (len(rates) + 1)
        self.stages.append(_ConvBnReLU(in_ch, self.out_ch, 1, 1, 1))  # 1x1 Convolution
        for i, rate in enumerate(rates):
            # Causal padding: padding only on the 'left' side
            causal_padding = (3 - 1) * rate  # (kernel_size - 1) * dilation
            self.stages.append(
                _ConvBnReLU(in_ch, self.out_ch, 3, 1, rate)
            )

    def forward(self, x):
        outputs = [stage(x) for stage in self.stages]
        return torch.cat(outputs, dim=1)  # Concatenate along channel dimension



###############################################################################################
################################     1 D UNets     ############################################


class UNet_causal_5mn_atrous(nn.Module):
    def __init__(self, n_channels, n_classes, size=64, dilation=1, atrous_rates=[6, 12, 18], fixed_cumul=False, additional_parameters=2):
        super(UNet_causal_5mn_atrous, self).__init__()
        self.inc = double_conv_causal(n_channels, size)  # Using double_conv_causal directly for simplicity
        self.down1 = Down_causal(size, 2*size)
        self.down2 = Down_causal(2*size, 4*size)
        self.down3 = Down_causal(4*size, 8*size, pooling_kernel_size=5, pooling_stride=5)
        self.down4 = Down_causal(8*size, 4*size, pooling=False, dilation=dilation)
        self.atrous = ASPP(4*size, rates=atrous_rates)       
        self.up2 = Up_causal(4*size, 2*size, kernel_size=5, stride=5)
        self.up3 = Up_causal(2*size, size)
        self.up4 = Up_causal(size, size)
        self.outc = outconv(size, n_classes)
        self.n_classes = n_classes
        self.p = nn.Parameter(torch.ones(additional_parameters))
        self.fixed_cumul = fixed_cumul
        self.pad_size = 20 - 1
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.atrous(x5)
        x = self.up2(x6, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x) 
        if not self.fixed_cumul:
            return x
        else:
            z = x[:,[2]]
            out_ch = 3 #x.shape[1]
            z = F.pad(z, (self.pad_size, 0), "constant", 0)
            conv_filter = torch.ones((out_ch - 2, 1, 20), dtype=torch.float32, device = x.device)
            z = F.conv1d(z, conv_filter, groups=out_ch-2)
            z = self.relu(z)
            
            x[:,[2]] = z
            return x


class UNet_causal_5mn_atrous_rescale(nn.Module):
    def __init__(self, n_channels, n_classes, size=64, dilation=1, atrous_rates=[6, 12, 18], fixed_cumul=False, additional_parameters=2):
        super(UNet_causal_5mn_atrous_rescale, self).__init__()
        self.inc = double_conv_causal(n_channels, size)  # Using double_conv_causal directly for simplicity
        self.down1 = Down_causal(size, 2*size)
        self.down2 = Down_causal(2*size, 4*size)
        self.down3 = Down_causal(4*size, 8*size, pooling_kernel_size=5, pooling_stride=5)
        self.down4 = Down_causal(8*size, 4*size, pooling=False, dilation=dilation)
        self.atrous = ASPP(4*size, rates=atrous_rates)       
        self.up2 = Up_causal(4*size, 2*size, kernel_size=5, stride=5)
        self.up3 = Up_causal(2*size, size)
        self.up4 = Up_causal(size, size)
        self.outc = outconv(size, n_classes)
        self.n_classes = n_classes
        self.p = nn.Parameter(torch.ones(additional_parameters))
        self.fixed_cumul = fixed_cumul
        self.pad_size = 20 - 1
        self.relu = nn.ReLU()
        
    def forward(self, x, indices=torch.tensor([0])):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.atrous(x5)
        x = self.up2(x6, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        indices = torch.cat([torch.arange(5).to(indices.device), 5 + indices])
        if not self.fixed_cumul:
            return x, self.p[indices]
        else:
            z = x[:,[2]]
            out_ch = 3 #x.shape[1]
            z = self.relu(z)
            z = F.pad(z, (self.pad_size, 0), "constant", 0)
            conv_filter = torch.ones((out_ch - 2, 1, 20), dtype=torch.float32, device = x.device)
            z = F.conv1d(z, conv_filter, groups=out_ch-2)
            # z = self.relu(z)
            x[:,[2]] = z
            return x, self.p[indices]


class UNet_causal_5mn_atrous_complex_rescale(nn.Module):
    def __init__(self, n_channels, n_classes, size=64, dilation=1, atrous_rates=[6, 12, 18], fixed_cumul=False, additional_parameters=2, num_cmls=1000, input_size_fc_layer=5, hidden_size_fc_layer=5):
        super(UNet_causal_5mn_atrous_complex_rescale, self).__init__()
        self.inc = double_conv_causal(n_channels, size)  # Using double_conv_causal directly for simplicity
        self.down1 = Down_causal(size, 2*size)
        self.down2 = Down_causal(2*size, 4*size)
        self.down3 = Down_causal(4*size, 8*size, pooling_kernel_size=5, pooling_stride=5)
        self.down4 = Down_causal(8*size, 4*size, pooling=False, dilation=dilation)
        self.atrous = ASPP(4*size, rates=atrous_rates)       
        self.up2 = Up_causal(4*size, 2*size, kernel_size=5, stride=5)
        self.up3 = Up_causal(2*size, size)
        self.up4 = Up_causal(size, size)
        self.outc = outconv(size, n_classes)
        self.n_classes = n_classes
        self.input_size_fc_layer = input_size_fc_layer
        self.hidden_size_fc_layer = hidden_size_fc_layer
        self.p = nn.Parameter(torch.ones(16))
        self.num_cmls = num_cmls
        self.linears1 = nn.ModuleList([nn.Linear(self.input_size_fc_layer, self.hidden_size_fc_layer) for i in range(self.num_cmls + 1)])
        for linear in self.linears1:
            linear.weight.data.fill_(0.)
            linear.bias.data.fill_(0.)
        self.linears2 = nn.ModuleList([nn.Linear(self.hidden_size_fc_layer, 1) for i in range(self.num_cmls + 1)])
        for linear in self.linears2:
            linear.weight.data.fill_(0.)
            linear.bias.data.fill_(0.)
        self.fixed_cumul = fixed_cumul
        self.pad_size = 20 - 1
        self.relu = nn.ReLU()

    def freeze_generic_parts(self):
        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False
        
        # Unfreeze parameters in linears1 and linears2
        for linear in self.linears1:
            for param in linear.parameters():
                param.requires_grad = True
        
        for linear in self.linears2:
            for param in linear.parameters():
                param.requires_grad = True

    def freeze_specific_parts(self):
        # Unfreeze parameters in linears1 and linears2
        for linear in self.linears1:
            for param in linear.parameters():
                param.requires_grad = False
        
        for linear in self.linears2:
            for param in linear.parameters():
                param.requires_grad = False

    def unfreeze_generic_parts(self):
        # Unfreeze all parameters
        for param in self.parameters():
            param.requires_grad = True


    def freeze_specific_parts(self):
        
        # Unfreeze parameters in linears1 and linears2
        for linear in self.linears1:
            for param in linear.parameters():
                param.requires_grad = False
        
        for linear in self.linears2:
            for param in linear.parameters():
                param.requires_grad = False

    def unfreeze_specific_parts(self):
        # Unfreeze parameters in linears1 and linears2
        for linear in self.linears1:
            for param in linear.parameters():
                param.requires_grad = True
        
        for linear in self.linears2:
            for param in linear.parameters():
                param.requires_grad = True


    def rescale(self, inputs, batch_ids):
        for i in range(batch_ids.shape[0]):
            batch_id = batch_ids[i]
            x = self.linears1[1 + batch_id](inputs[i].transpose(0,1).contiguous())
            x = self.relu(x)
            x = self.linears2[1 + batch_id](x)
            inputs[i, [0]] += x.transpose(0,1).contiguous() # residual

        return inputs[:,[0]]


    def forward(self, x, batch_ids):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.atrous(x5)
        x = self.up2(x6, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        if not self.fixed_cumul:
            x[:,2:] = self.rescale(x[:,2:], batch_ids)
            return x
        else:
            z = x[:,2:]
            z = self.rescale(z, batch_ids)
            z = self.relu(z)
            out_ch = x.shape[1]
            z = F.pad(z, (self.pad_size, 0), "constant", 0)
            conv_filter = torch.ones((1, 1, 20), dtype=torch.float32, device = x.device)
            z = F.conv1d(z, conv_filter, groups=1)
            x[:,2:] = z
            return x





class UNet_causal_5mn_atrous_multiplicative_rescale(nn.Module):
    def __init__(self, n_channels, n_classes, size=64, dilation=1, atrous_rates=[6, 12, 18],
                 fixed_cumul=False, additional_parameters=2, num_cmls=1000,
                 input_size_fc_layer=5, hidden_size_fc_layer=5):
        super(UNet_causal_5mn_atrous_multiplicative_rescale, self).__init__()
        self.inc = double_conv_causal(n_channels, size)  # Using double_conv_causal directly for simplicity
        self.down1 = Down_causal(size, 2*size)
        self.down2 = Down_causal(2*size, 4*size)
        self.down3 = Down_causal(4*size, 8*size, pooling_kernel_size=5, pooling_stride=5)
        self.down4 = Down_causal(8*size, 4*size, pooling=False, dilation=dilation)
        self.atrous = ASPP(4*size, rates=atrous_rates)       
        self.up2 = Up_causal(4*size, 2*size, kernel_size=5, stride=5)
        self.up3 = Up_causal(2*size, size)
        self.up4 = Up_causal(size, size)
        self.input_size_fc_layer = input_size_fc_layer
        self.outc = outconv(size, self.input_size_fc_layer)
        self.n_classes = n_classes
        self.hidden_size_fc_layer = hidden_size_fc_layer
        self.p = nn.Parameter(torch.ones(16))
        self.num_cmls = num_cmls
        self.linears1 = nn.ModuleList([nn.Linear(self.input_size_fc_layer, self.hidden_size_fc_layer) for i in range(self.num_cmls + 1)])
        for linear in self.linears1[1:]:
            linear.load_state_dict(self.linears1[0].state_dict())
        self.linears2 = nn.ModuleList([nn.Linear(self.hidden_size_fc_layer, self.n_classes) for i in range(self.num_cmls + 1)])
        for linear in self.linears2[1:]:
            linear.load_state_dict(self.linears2[0].state_dict())
        self.fixed_cumul = fixed_cumul
        self.pad_size = 20 - 1
        self.relu = nn.ReLU()

    def freeze_generic_parts(self):
        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False
        
        # Unfreeze parameters in linears1 and linears2
        for linear in self.linears1:
            for param in linear.parameters():
                param.requires_grad = True
        
        for linear in self.linears2:
            for param in linear.parameters():
                param.requires_grad = True

    def unfreeze_generic_parts(self):
        # Unfreeze all parameters
        for param in self.parameters():
            param.requires_grad = True

    def freeze_specific_parts(self):
        
        # Unfreeze parameters in linears1 and linears2
        for linear in self.linears1:
            for param in linear.parameters():
                param.requires_grad = False
        
        for linear in self.linears2:
            for param in linear.parameters():
                param.requires_grad = False

    def unfreeze_specific_parts(self):
        
        # Unfreeze parameters in linears1 and linears2
        for linear in self.linears1:
            for param in linear.parameters():
                param.requires_grad = True
        
        for linear in self.linears2:
            for param in linear.parameters():
                param.requires_grad = True

    def rescale(self, inputs, batch_ids):
        for i in range(batch_ids.shape[0]):
            batch_id = batch_ids[i]
            x = self.linears1[1 + batch_id](inputs[i].transpose(0,1).contiguous())
            x = self.relu(x)
            x = self.linears2[1 + batch_id](x)
            inputs[i, 0:self.n_classes] *= 1 + x.transpose(0,1).contiguous() # residual

        return inputs[:, 0:self.n_classes]


    def forward(self, x, batch_ids):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.atrous(x5)
        x = self.up2(x6, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        if not self.fixed_cumul:
            x = self.rescale(x, batch_ids)
            return x
        else:
            z = x
            z = self.rescale(z, batch_ids)
            z = self.relu(z)
            out_ch = x.shape[1]
            z = F.pad(z, (self.pad_size, 0), "constant", 0)
            conv_filter = torch.ones((1, 1, 20), dtype=torch.float32, device = x.device)
            z = F.conv1d(z, conv_filter, groups=1)
            x = z
            return x






##############################################################################################
################################     UNet 1D building blokcs    ##############################
import torch
import torch.nn as nn
import torch.nn.functional as F

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x




class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool1d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x

#Given transposed=1, weight[1024, 256, 2, 2], so expected input[64, 512, 4, 4] to have 1024 channels, but got 512 channels instead
        
    
    
class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear')#nn.UpsamplingBilinear2d(scale_factor=2)
        else:
            self.up = nn.ConvTranspose1d(in_ch, in_ch, kernel_size=2, stride=2)

        self.conv = double_conv(2*in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        # diffY = x1.size()[3] - x2.size()[3]
        # x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
        #                 diffY // 2, int(diffY / 2)))
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2)))

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, 1)
        

    def forward(self, x):
        x = self.conv(x)
        return x




####################################################################################################################################
######################################## class UNet ################################################################################

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, size=64):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, size)
        self.down1 = Down(size, 2*size)
        self.down2 = Down(2*size, 4*size)
        self.down3 = Down(4*size, 8*size)
        self.down4 = Down(8*size, 8*size)
        self.up1 = Up(8*size, 4*size)
        self.up2 = Up(4*size, 2*size)
        self.up3 = Up(2*size, size)
        self.up4 = Up(size, size)
        self.outc = outconv(size, n_classes)
        self.n_classes=n_classes
        self.p = nn.Parameter(torch.ones(16))
        
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        del x4, x5
        x = self.up2(x, x3)
        del x3
        x = self.up3(x, x2)
        del x2
        x = self.up4(x, x1)
        del x1
        x = self.outc(x) 
        return   x