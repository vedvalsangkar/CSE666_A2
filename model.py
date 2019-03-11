import torch.nn as nn
import torch.optim as optim


class Block(nn.Module):

    def __init__(self, layer1ch=3, layer2ch=32, bias=False):
        super(Block, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(in_channels=layer1ch,
                                              out_channels=layer2ch,
                                              kernel_size=3,
                                              stride=1,
                                              padding=1,
                                              bias=bias),
                                    nn.ReLU(),
                                    )
        self.layer2 = nn.Sequential(nn.Conv2d(in_channels=layer2ch,
                                              out_channels=layer2ch,
                                              kernel_size=3,
                                              stride=1,
                                              padding=1,
                                              bias=bias),
                                    nn.ReLU(),
                                    )
        self.layer3 = nn.Sequential(nn.Conv2d(in_channels=layer2ch,
                                              out_channels=layer2ch,
                                              kernel_size=3,
                                              stride=1,
                                              padding=1,
                                              bias=bias),
                                    nn.ReLU(),
                                    )
        # self.layer1 = nn.Conv2d(layer1ch, layer2ch, kernel_size=3, stride=1, padding=1, bias=bias)
        # self.layer2 = nn.Conv2d(layer2ch, layer2ch, kernel_size=3, stride=1, padding=1, bias=bias)
        # self.layer3 = nn.Conv2d(layer2ch, layer2ch, kernel_size=3, stride=1, padding=1, bias=bias)
        self.res_layer = nn.Sequential()

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out += self.res_layer(x)
        out = out.reshape(out.size(0), -1)
        # out = self.fc1(out)
        # out = self.fc(out)
        return out


class ResNet101(nn.Module):

    def __init__(self, num_classes=500):
        super(ResNet101, self).__init__()

        # Dropout rate
        self.do_rate = 0.1

        # 203x202 image with 32 channels in the last superblock and 2x2 MaxPool layer at the end.
        self.l5out = int((203*202*32)/4)

        # Final features
        self.features = 2048

        self.layer1 = self.build_layer(3, Block, 3, 16)
        self.layer2 = self.build_layer(4, Block, 16, 16)
        self.layer3 = self.build_layer(23, Block, 16, 32)
        self.layer4 = self.build_layer(3, Block, 32, 32)
        self.layer5 = nn.Sequential(nn.MaxPool2d(kernel_size=2,
                                                 stride=2),
                                    nn.Dropout(self.do_rate)
                                    )
        self.feat_layer = nn.Sequential(nn.Linear(self.l5out, self.features),
                                        nn.ReLU(),
                                        nn.Dropout(self.do_rate)
                                        )
        self.class_layer = nn.Sequential(nn.Linear(self.features, num_classes),
                                         nn.Softmax()
                                         )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.reshape(out.size(0), -1)
        out = self.feat_layer(out)
        out = self.class_layer(out)
        return out

    def build_layer(self, num_blocks, block_class, in_ch, out_ch):
        layers = []
        layers.append(block_class(layer1ch=in_ch, layer2ch=out_ch))
        for n in range(num_blocks-1):
            layers.append(block_class(layer1ch=out_ch, layer2ch=out_ch))
        return nn.Sequential(*layers)


# class Model666(nn.Module):
#     def __init__(self, num_classes=200, layer1ch=32, layer2ch=64, hidden_size=512, k_size=3, padding=1, do_rate=0.1):
#         super(Model666, self).__init__()
#         self.layer1 = nn.Sequential(
#             nn.Conv2d(in_channels=3,
#                       out_channels=layer1ch,
#                       kernel_size=k_size,
#                       stride=1,
#                       padding=padding),
#             nn.ReLU(),
#         )
#         # output =
#         self.layer2 = nn.Sequential(
#             nn.Conv2d(in_channels=layer1ch,
#                       out_channels=layer1ch,
#                       kernel_size=k_size,
#                       stride=1,
#                       padding=padding),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2,
#                          stride=2),
#             nn.Dropout(do_rate)
#         )
#         self.layer3 = nn.Sequential(
#             nn.Conv2d(in_channels=layer1ch,
#                       out_channels=layer2ch,
#                       kernel_size=k_size+2,
#                       stride=1,
#                       padding=padding+1),
#             nn.ReLU(),
#         )
#         self.layer4 = nn.Sequential(
#             nn.Conv2d(in_channels=layer2ch,
#                       out_channels=layer2ch,
#                       kernel_size=k_size+2,
#                       stride=1,
#                       padding=padding+1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2,
#                          stride=2),
#             nn.Dropout(do_rate)
#         )
#         self.fc1 = nn.Sequential(
#             nn.Linear(16 * 16 * layer2ch, hidden_size),
#             nn.ReLU(),
#             nn.Dropout(do_rate), )
#
#         # self.fc = nn.Linear(hidden_size, num_classes)
#         self.fc = nn.Sequential(
#             nn.Linear(hidden_size, num_classes),
#             nn.Softmax())
#         # nn.ReLU())
#
#     def forward(self, x):
#         out = self.layer1(x)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
#         out = out.reshape(out.size(0), -1)
#         out = self.fc1(out)
#         out = self.fc(out)
#         return out


def get_model(device, opt='Adam', num_classes=500, lamb=0.01, learning_rate=0.01):
    model = ResNet101(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    if opt == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=lamb)
    elif opt == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=lamb)
    else:
        print("Unknown value. Choosing Adam instead.")
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=lamb)

    return model, criterion, optimizer
