import torch
import torch.nn as nn
import torch.nn.functional as F


class iCNNnode(torch.nn.Module):
    def __init__(self):

        super(iCNNnode,self).__init__()


        self.in_channels = 3
        self.out_channels = 32
        self.kernel_size = 3
        self.stride = self.kernel_size//2
        self.padding = 1

        self.interpol_size = 3
        self.interpol_mode = 'nearest'

        self.pool_size = 3
        self.pool_stride = self.pool_size//2

        self.conv_node = nn.Conv2d(in_channels=self.in_channels,out_channels=self.out_channels,kernel_size=self.kernel_size,stride=self.stride,padding=self.padding)

    def set_Conv(self,in_channels,out_channels,kernel_size,stride,padding):

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def set_upsample(self, interpol_size, interpol_mode):
        self.interpol_size = interpol_size
        self.interpol_mode = interpol_mode


    def set_downsample(self, pool_size, pool_stride):

        self.pool_size = pool_size
        self.pool_stride = pool_stride


    def upsample(self, input):
        result = F.interpolate(input, scale_factor=self.interpol_size,mode=self.interpol_mode)
        return result

    def downsample(self, input):
        result = F.max_pool2d(input,kernel_size=self.pool_size,stride=self.pool_stride)
        return result

    def forward(self,x,pre,post):

        y = self.conv_node(x)
        self.pre =  self.upsample(y)
        self.post = self.downsample(y)
        y = pre + self.y + post

        return y,self.pre,self.post

class iCNN_Cell(torch.nn.Module):

    def __init__(self):
        super(iCNN_Cell,self).__init__()
        iCNN_Nodelist =



class iCNNcell(torch.nn.Module):



    def __init__(self,in_channels,out_channels,kernel_size,stride,padding,pool_size,pool_stride,interpol_size,interpol_mode):

        super(iCNNcell,self).__init__()


        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.pool_size = pool_size
        self.pool_stride = pool_stride
        self.interpol_size = interpol_size
        self.interpol_mode = interpol_mode


        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def upSample(self,obj):
        result = F.interpolate(obj, scale_factor=self.interpol_size,mode=self.interpol_mode)
        return result

    def downSample(self,obj):
        result = F.max_pool2d(obj,kernel_size=self.pool_size,stride=self.pool_stride)
        return result


    def forward(self, x0,x1,h_0,h_1):
        # Computes the activation of the first convolution
        self.F0 = F.relu(self.conv1(x0))
        self.F1 = F.relu(self.conv2(x1))

        self.Fp_down = h_0
        self.F2_up = h_1


        self.F0_down = self.downSample(self.F0)
        self.F0_up =   self.upSample(self.F0)
        self.F1_down = self.downSample(self.F1)
        self.F1_up =   self.upSample(self.F1)



        y_0 = self.Fp_down + self.F0 + self.F1_up
        y_1 =  self.F0_down + self.F1 + self.F2_up

        h_pre = self.F0_up
        h_next = self.F1_down

        return y_0,y_1,h_pre,h_next


class face_model(torch.nn.Module):

    def __init__(self):

        input_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        interlink_layer1 = iCNNcell()
        interlink_layer2 = iCNNcell()
        interlink_layer3 = iCNNcell()


    def forward(self):





