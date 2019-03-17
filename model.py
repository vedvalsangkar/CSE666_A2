#!/usr/bin/python3.5

# The following model has been built with the following site as reference:
# https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
#
# The nested nn.Module class structure was picked up from this site.
#


from torch import nn, optim
from torch.nn import functional as F


class Block(nn.Module):

    def __init__(self, layer1ch, layer2ch, bias=False):
        """
        Block class for Residual Neural Network. 2 or 3 layer.
        :param layer1ch: Input channels
        :param layer2ch:
        :param bias:
        """
        super(Block, self).__init__()

        self.layer1 = nn.Sequential(nn.Conv2d(in_channels=layer1ch,
                                              out_channels=layer2ch,
                                              # kernel_size=1,
                                              kernel_size=3,
                                              stride=1,
                                              # padding=0,
                                              padding=1,
                                              bias=bias),
                                    nn.BatchNorm2d(layer2ch),
                                    nn.ReLU(),
                                    )
        self.layer2 = nn.Sequential(nn.Conv2d(in_channels=layer2ch,
                                              out_channels=layer2ch,
                                              kernel_size=3,
                                              stride=1,
                                              padding=1,
                                              bias=bias),
                                    nn.BatchNorm2d(layer2ch)
                                    # nn.ReLU(),
                                    )
        # self.layer3 = nn.Sequential(nn.Conv2d(in_channels=layer2ch,
        #                                       out_channels=layer2ch,
        #                                       kernel_size=1,
        #                                       stride=1,
        #                                       padding=0,
        #                                       bias=bias),
        #                             nn.ReLU(),
        #                             )

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        # out = self.layer3(out)
        out += x
        return self.relu(out)
        # return out


class ResNet(nn.Module):
    """
    Custom dataset class for loading training images.

    resnet34  : 3,4,6,3 with 2 layer block.
    resnet50  : 3,4,6,3 with 3 layer block.
    resnet101 : 3,4,23,3 with 3 layer block.
    """

    def __init__(self, num_classes=500, final_features=1024, bias=False):
        """
        ResNet constructor class.
        :param num_classes : Number of classes (default 500).
        :param bias        : Bias for the layers
        """
        super(ResNet, self).__init__()

        # Dropout rate
        self.do_rate = 0.1

        self.temperature = 0.9

        # 203x202 image with 32 channels in the last superblock and 2x2 MaxPool layer at the end.
        # 203->101->50
        self.L5_out_size = int(50 * 50 * 32)

        # Final features
        self.features = final_features

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=3,
                                             out_channels=16,
                                             kernel_size=3,
                                             stride=1,
                                             padding=1,
                                             bias=bias),
                                   nn.ReLU(),
                                   )

        self.layer1 = self._build_layer(3, Block, 16, 16)

        self.layer2 = self._build_layer(4, Block, 16, 16)

        self.layer2_5 = nn.Sequential(nn.Conv2d(in_channels=16,
                                                out_channels=32,
                                                kernel_size=1,
                                                stride=1,
                                                padding=0,
                                                bias=bias),
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=2,
                                                   stride=2)
                                      )  # Size is halved

        # self.layer3 = self.build_layer(23, Block, 32, 32)
        self.layer3 = self._build_layer(6, Block, 32, 32)

        self.layer4 = self._build_layer(3, Block, 32, 32)

        self.layer5 = nn.Sequential(nn.MaxPool2d(kernel_size=2,
                                                 stride=2),
                                    # nn.Dropout(self.do_rate)  # Size is halved
                                    )

        self.feat_layer = nn.Sequential(nn.Linear(self.L5_out_size, self.features),
                                        nn.ReLU(),
                                        # nn.Tanh(),
                                        # nn.Dropout(self.do_rate)
                                        )
        self.class_layer = nn.Sequential(nn.Linear(self.features, num_classes),
                                         # nn.ReLU(),
                                         # nn.LogSoftmax(dim=1)
                                         # nn.Softmax(dim=1)
                                         )

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer2_5(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.reshape(out.size(0), -1)
        out = self.feat_layer(out)
        if self.training:
            # out = self.class_layer(out)
            # out = F.log_softmax(out / self.temperature, 1)
            out = F.log_softmax(self.class_layer(out) / self.temperature, 1)
        # else:
        #     out = out

        return out

    @staticmethod
    def _build_layer(num_blocks, block_class, in_ch, out_ch):
        """
        Static method defined to create nested nn.Module classes for deep residual network.
        :param num_blocks  : Number of blocks of sub-class to add.
        :param block_class : Class of the block to be added.
        :param in_ch       : Input planes i.e. channels of input image/previous convolution.
        :param out_ch      : Number of planes i.e. channels of output after convolution.
        :return:
        """

        layers = [block_class(layer1ch=in_ch, layer2ch=out_ch)]

        for n in range(num_blocks - 1):
            layers.append(block_class(layer1ch=out_ch, layer2ch=out_ch))
        return nn.Sequential(*layers)

    def set_temperature(self, temp):
        self.temperature = temp


def get_model(device, opt='Adam', num_classes=500, lamb=0.01, learning_rate=0.01, final_features=1024):

    model = ResNet(num_classes=num_classes, final_features=final_features).to(device)
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.NLLLoss()
    if opt == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=lamb, amsgrad=False,)
    elif opt == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=lamb)
    elif opt == 'RMS':
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=lamb, momentum=0.1, centered=True)
    else:
        print("Unknown value. Choosing Adam instead.")
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=lamb)

    return model, criterion, optimizer
