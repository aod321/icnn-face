import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Interpolate(nn.Module):
    def __init__(self, size, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode)
        return x


class iCNN_Node(torch.nn.Module):
    def __init__(self):

        super(iCNN_Node, self).__init__()

        self.in_channels = 3
        self.out_channels = 32
        self.kernel_size = 3
        self.stride = self.kernel_size//2
        self.padding = 1

        self.interpol_size = 3
        self.interpol_mode = 'nearest'
        self.interp_layer = Interpolate(size=self.interpol_size, mode=self.interpol_mode)

        self.pool_size = 3
        self.pool_stride = 1

        self.conv_node = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=self.
                                   kernel_size, stride=self.stride, padding=self.padding)

    def set_Conv(self, in_channels, out_channels, kernel_size, stride, padding):

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.conv_node = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=self.
                                   kernel_size, stride=self.stride, padding=self.padding)

    def set_upsample(self, interpol_size, interpol_mode='nearest'):
        self.interpol_size = interpol_size
        self.interpol_mode = interpol_mode
        self.interp_layer = Interpolate(size=self.interpol_size, mode=self.interpol_mode)

    def set_downsample(self, pool_size, pool_stride):

        self.pool_size = pool_size
        self.pool_stride = pool_stride

    def upsample(self, x):
        result = self.interp_layer(x)
        return result

    def downsample(self, x):
        result = F.max_pool2d(x,kernel_size=self.pool_size, stride=self.pool_stride,padding=1)
        return result

    def forward(self, x, feature_from_pre, feature_from_post):
        # Downsample the features from the pre-node
        feature_from_pre = self.downsample(feature_from_pre)
        # Upsample the features from the post-node
        feature_from_post = self.upsample(feature_from_post)

        # cuda support
        feature_from_pre = feature_from_pre.to(device)
        feature_from_post = feature_from_post.to(device)
        x = x.to(device)

        # Interlink them before convolution
        # Use Conv2d whose weight is all 1 to compute  x = feature_from_pre + x + feature_from_post
        x_out = torch.cat([feature_from_pre, x, feature_from_post], dim=1)
        filters = torch.ones((x.shape[1], x_out.shape[1], 3, 3)).to(device)
        x_out = F.conv2d(input=x_out,
                         weight=filters,
                         stride= 1,
                         padding= 1
                         )
        y = F.relu(self.conv_node(x_out),inplace=True)
        return y


class iCNN_Cell(torch.nn.Module):

    def __init__(self, recurrent_number, in_channels, out_channels, kernel_size, stride,
                 padding, down_size, down_stride, up_size, up_mode='nearest'):
        super(iCNN_Cell, self).__init__()

        # Init Conv parameters
        self.recurrent_number = recurrent_number
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Init down_sample parameters
        self.down_size = down_size
        self.down_stride = down_stride
        # Init up_sample parameters
        self.up_size = up_size
        self.up_mode = up_mode

        self.iCNN_Nodelist = nn.ModuleList([iCNN_Node() for i in range(self.recurrent_number)])
        self.interlink_features = []

        # Init  parameters for each iCNNnode in iCNN_Nodelist
        for i in range(self.recurrent_number):
            self.iCNN_Nodelist[i].set_Conv(self.in_channels[i], self.out_channels[i],
                                           self.kernel_size[i], self.stride[i], self.padding[i])
            self.iCNN_Nodelist[i].set_upsample(self.up_size[i], self.up_mode[i])
            self.iCNN_Nodelist[i].set_downsample(self.down_size[i], self.down_stride[i])

    def forward(self, first_features):

        self.interlink_features = []

        # Set upsample size
        for i in range(self.recurrent_number):
            self.iCNN_Nodelist[i].set_upsample(interpol_size=(first_features[i].shape[2],
                                                              first_features[i].shape[3])
                                               )

        # Step forward for each interlinking node
        # Compute the head node firstly, since it doesn't has any front nodes.
        n, c, h, w = first_features[0].size()
        zeros_tensor = torch.zeros((n, c, 2 * h, 2 * w))
        now_node = self.iCNN_Nodelist[0]
        temp_feature = now_node(first_features[0],feature_from_pre=zeros_tensor,
                                feature_from_post=first_features[1])
        self.interlink_features.append(temp_feature)

        # Then compute all the middle interlinked nodes
        for i in range(1,self.recurrent_number-1):
            temp_feature = self.iCNN_Nodelist[i](first_features[i],
                                                 feature_from_pre=first_features[i-1],
                                                 feature_from_post=first_features[i+1])
            self.interlink_features.append(temp_feature)

        # Finally handle the tail node, as it doesn't has any next nodes.

        j = self.recurrent_number-1
        zeros_tensor = torch.zeros(first_features[j].shape)
        temp_feature = self.iCNN_Nodelist[j](first_features[j],
                                             feature_from_pre=first_features[j-1],
                                             feature_from_post=zeros_tensor)
        self.interlink_features.append(temp_feature)

        return self.interlink_features

    def get_output_integration(self):
        # copy origin interlink_features list to temp
        temp_feature = self.interlink_features[:]

        # i from N-1 to 1
        for i in range(self.recurrent_number - 1, 0, -1):

            node_pre = self.iCNN_Nodelist[i-1]
            node_now = self.iCNN_Nodelist[i]
            feature_pre = temp_feature[i-1].to(device)
            feature_now = temp_feature[i].to(device)
            feature_to_pre = node_pre.upsample(feature_now)

            # cat feature_pre and feature_to_pre
            cat_feature = torch.cat([feature_pre, feature_to_pre],
                                    dim=1)

            # Use conv2d to compute  feature_pre += feature_to_pre
            filters = torch.ones((feature_pre.shape[1], cat_feature.shape[1], 3, 3)).to(device)
            temp_feature[i - 1] = F.conv2d(input=cat_feature,
                                           weight=filters,
                                           stride=1,
                                           padding=1
                                           )
        return temp_feature[0]


class FaceModel(torch.nn.Module):

    def __init__(self):

        super(FaceModel, self).__init__()

        # Parameters initiation

        # L:the number of label channels
        self.lable_channel_size = 9

        # Conv paramters

        self.in_channels = 3
        self.interlink_layer_number = 3
        self.recurrent_number = 4
        self.kernel_size = 5
        self.last_kernel_size = 9

        self.down_stride = 2
        self.down_size = 3
        self.up_size = 2
        # Parameters for each icnn_node in iCNNnodelist

        self.kernel_sizes_list = [self.kernel_size for _ in range(self.recurrent_number)]
        self.stride_list = [1 for i in range(self.recurrent_number)]
        self.padding_list = [self.kernel_size//2 for _ in range(self.recurrent_number)]

        self.down_size_list = [self.down_size for _ in range(self.recurrent_number)]
        self.down_stride_list = [self.down_stride for _ in range(self.recurrent_number)]
        self.up_size_list = [self.up_size for _ in range(self.recurrent_number)]

        self.first_channels_size = [8*(i+1) for i in range(self.recurrent_number)]                     # [8,16,24,32]
        # Relu layer
        self.relu_layer = nn.ReLU(inplace=True)

        # Input layer
        self.input_conv = nn.ModuleList([nn.Conv2d(in_channels=self.in_channels,
                                                   out_channels=self.first_channels_size[i],
                                                   kernel_size=self.kernel_size,
                                                   stride=1,
                                                   padding=self.kernel_size//2)
                                         for i in range(self.recurrent_number)
                                           ]
                                          )

        # interlink_layer0 interlink_layer1 interlink_layer2

        self.interlink_layers = nn.ModuleList([iCNN_Cell(recurrent_number=self.recurrent_number,
                                                         in_channels=self.first_channels_size,
                                                         out_channels=self.first_channels_size,
                                                         kernel_size=self.kernel_sizes_list,
                                                         stride=self.stride_list,
                                                         padding=self.padding_list,
                                                         down_size=self.down_size_list,
                                                         down_stride=self.down_stride_list,
                                                         up_size=self.up_size_list)
                                              for _ in range(self.interlink_layer_number)
                                              ]
                                             )

        # last conv layer1     input channels:8     output channels:2L+8
        self.last_conv1 = nn.Conv2d(in_channels=self.first_channels_size[0],
                                    out_channels=2 * self.lable_channel_size + 8,
                                    kernel_size=self.kernel_size,stride=1,
                                    padding=self.kernel_size//2)

        # last conv layer2     input channels:2L+8  output channels:L channels
        self.last_conv2 = nn.Conv2d(in_channels=2 * self.lable_channel_size + 8,
                                    out_channels=self.lable_channel_size,
                                    kernel_size=self.kernel_size,stride=1,
                                    padding=self.kernel_size//2)

        # last conv layer3     input channels:L  output channels:L channels
        self.last_conv3 = nn.Conv2d(in_channels=self.lable_channel_size,
                                    out_channels=self.lable_channel_size,
                                    kernel_size=self.last_kernel_size,
                                    stride=1,
                                    padding=self.last_kernel_size//2)

        # SoftMax layer
        self.softmax_layer = nn.Softmax2d()

    def forward(self, x):

        # Scale the input
        # After this scaled_x = Scaleed image of [row1,row2,row3,row4]
        scaled_x = []
        scaled_x.append(x)

        for i in range(1, self.recurrent_number):
            x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
            scaled_x.append(x)

        # batch,c,h,w = x.shape
        # convolve and output feature maps
        # After this inputs = feature maps of [row1,row2,row3,row4]

        inputs = [self.relu_layer(self.input_conv[0](scaled_x[0]))]
        for i in range(1, self.recurrent_number):
            temp_i = self.relu_layer(self.input_conv[i](scaled_x[i]))
            inputs.append(temp_i)

        # Step forward for each interlinking layer
        # After this inputs = feature maps of [row1,row2,row3,row4] after interlinking layer
        for i in range(self.interlink_layer_number):
            inputs = self.interlink_layers[i](inputs)

        # Get output integration
        output = self.interlink_layers[self.interlink_layer_number - 1].get_output_integration()

        # Final Output

        final_output = self.relu_layer(self.last_conv1(output))
        final_output = self.relu_layer(self.last_conv2(final_output))
        final_output = self.relu_layer(self.last_conv3(final_output))
        final_output = self.softmax_layer(final_output)

        return final_output


