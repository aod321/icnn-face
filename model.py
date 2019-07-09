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
        self.pool_stride = 1




        self.conv_node = nn.Conv2d(in_channels=self.in_channels,out_channels=self.out_channels,kernel_size=self.kernel_size,stride=self.stride,padding=self.padding)

    def set_Conv(self,in_channels,out_channels,kernel_size,stride,padding):

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.conv_node = nn.Conv2d(in_channels=self.in_channels,out_channels=self.out_channels,kernel_size=self.kernel_size,stride=self.stride,padding=self.padding)


    def set_upsample(self, interpol_size, interpol_mode='nearest'):
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

    def forward(self,x,feature_from_pre,feature_from_post):

        ##Downsample the features from the pre-node
        feature_from_pre = self.downsample(feature_from_pre)
        ##Upsample the features from the post-node
        feature_from_post = self.upsample(feature_from_post)
        ##Interlink them before convolution
        x = feature_from_pre  + x + feature_from_post

        y = F.relu(self.conv_node(x))
        return y


class iCNN_Cell(torch.nn.Module):

    def __init__(self,recurrent_number,in_channels,out_channels,kernel_size,stride,padding,down_size,down_stride,up_size,up_mode='nearest'):
        super(iCNN_Cell,self).__init__()

        #Init Conv parameters
        self.recurrent_number = recurrent_number
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        #Init down_sample parameters
        self.down_size = down_size
        self.down_stride = down_stride
        #Init up_sample parameters
        self.up_size = up_size
        self.up_mode = up_mode

        self.iCNN_Nodelist = nn.ModuleList([iCNNnode() for i in range(self.recurrent_number)])

        # Init  parameters for each iCNNnode in iCNN_Nodelist
        for i in range(self.recurrent_number):
            self.iCNN_Nodelist[i].set_Conv(self.in_channels[i],self.out_channels[i],self.kernel_size[i],self.stride[i],self.padding[i])
            self.iCNN_Nodelist[i].set_upsample(self.up_size[i],self.up_mode[i])
            self.iCNN_Nodelist[i].set_downsample(self.down_size[i],self.down_stride[i])


    def forward(self,first_features):

        # first_features = []

        # ## Convole and output the first round feature maps
        # ## After this: first_features = feature maps of [node_0,node_1,node_2,node_3,...,node_i]
        #
        # for i in range(self.recurrent_number):
        #     zeros_tensor = torch.zeros(x[i].shape)
        #     temp_feature = self.iCNN_Nodelist[i](x[i],feature_from_pre=zeros_tensor,
        #                                          feature_from_post=zeros_tensor)
        #     first_features.append(temp_feature)


        interlink_features = []

        ## Step forward for each interlinking node
        ##
        ## Compute the head node firstly, since it doesn't has any front nodes.

        zeros_tensor = torch.zeros(first_features[0].shape)
        temp_feature = self.iCNN_Nodelist[0](first_features[0], feature_from_pre=zeros_tensor,
                                                 feature_from_post=first_features[1])
        interlink_features.append(temp_feature)

        ## Then compute all the middle interlinked nodes

        for i in range(1,self.recurrent_number-1):
            temp_feature = self.iCNN_Nodelist[i](first_features[i], feature_from_pre=first_features[i-1],
                                                 feature_from_post=first_features[i+1])
            interlink_features.append(temp_feature)

        ## Finally handle the tail node, as it doesn't has any next nodes.

        j = self.recurrent_number-1
        zeros_tensor = torch.zeros(first_features[j].shape)
        temp_feature = self.iCNN_Nodelist[j](first_features[j], feature_from_pre=first_features[j-1],
                                                 feature_from_post=zeros_tensor)
        interlink_features.append(temp_feature)

        return interlink_features




class face_model(torch.nn.Module):

    def __init__(self):

        super(face_model,self).__init__()
        self.kernel_size = 5
        self.last_kernel_size = 9

        self.recurrent_number = 4

        self.in_channels = 3
        self.first_channels_size = [8*(i+1) for i in range(self.recurrent_number)]                     #[8,16,24,32]

        input_layers = nn.ModuleList([nn.Conv2d(self.in_channels,self.first_channels_size[i], self.kernel_size, stride=1, padding=self.kernel_size//2)
                                      for i in range(self.recurrent_number)])

        interlink_layer1 = iCNN_Cell(recurrent_number=self.recurrent_number,in_channels=self.first_channels_size,
                                     out_channels=self.first_channels_size,kernel_size=self.kernel_size,stride=1,
                                     padding=self.kernel_size//2,down_size=,down_stride=,up_size=):



    def forward(self):





