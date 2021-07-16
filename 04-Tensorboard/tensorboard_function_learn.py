import torch
import torch.nn as nn
import numpy as np
import tensorboard
import thop
import torchstat
from torchsummary import summary
from graphviz import Digraph
import netron
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

#权重初始化函数
def init_weight(m):
    """
    this function is created for weight initialization.
    call this function in Model's def __init__() part.
    :param m:
    :return:
    """
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
        # nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        # nn.init.normal_(m.weight)
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class ResNet3(nn.Module):
    """
    Neural Networks for classfication.
    """
    def __init__(self,in_channels):
        super(ResNet3, self).__init__()
        self.num = 3
        #part1 conv net
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels,128,(1,1),(1,1)),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.Conv2d(128, 128, (3, 3), (1, 1),padding=(1,1)),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.Conv2d(128, 512, (1, 1), (1, 1), padding=(0, 0)),
            nn.BatchNorm2d(512),
            nn.PReLU(),
        )
        #part2 liner net
        self.liner = nn.Sequential(
            nn.Linear(512 * 32 * 32, 10),
            nn.Softmax(dim=1)
        )
        #part3 call weight initialization function
        self.apply(init_weight)

    def forward(self,x):
        x = self.layer(x)
        x = x.reshape(x.shape[0],-1)
        x = self.liner(x)
        # print(x.shape)
        # if self.num < 5:
        #     x = self.layer(x)
        #     return x
        # else:
        #     x = self.layer(x)
        return x

if __name__ == '__main__':

    #Instantiate network
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # inputs = torch.randn(1,3,32,32)
    net = ResNet3(3).to(device)


    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    #                    参数量和浮点运算量打印                  #
    #                  print the flops and params             #
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

    #----------------------first method------------------------
    # flops, params = thop.profile(net.cpu(), inputs=(torch.randn(1, 3, 64, 64),))
    # print("flops:", flops)  # 查看浮点运算量，和批次有关
    # print("params:", params)  # 查看参数量，和批次无关

    #----------------------second method------------------------
    # torchstat.stat(net,(3,64,64))

    #-----------------------third method------------------------
    # summary(net,(3,64,64),device = "cpu")

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    #                          打印网络结构图                   #
    #                       print net structure               #
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    # #直接打印
    # print(net)
    # #netron输出
    # torch.onnx.export(net,inputs,"net.pth")
    # netron.start("net.pth")

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    #                          tensorboard使用                 #
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    summarywriter = SummaryWriter("../logs")
    model_path = "../net.pth"
    trans = transforms.ToTensor()
    train_dataset = CIFAR10('D:\data\cifar10', train=True, download=False, transform=trans)
    test_dataset = CIFAR10('D:\data\cifar10', train=False, download=False, transform=trans)
    train_dataloader = DataLoader(train_dataset, batch_size=100, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=100, shuffle=True)

    loss_fun = nn.MSELoss()
    optimizer = optim.Adam(net.parameters())

    net.train()
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    #                          打印指定层权重                   #
    #               Print the weight of the specified layer   #
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    # for epoch in range(10000):
    #     print("epoch = ",epoch)
    #     w0 = net.layer[0].weight
    #     b0 = net.layer[0].bias
    #     w1 = net.layer[1].weight
    #     b1 = net.layer[1].bias
    #     w2 = net.layer[2].weight
    #     # b2 = net.layer[2].bias
    #     w3 = net.layer[3].weight
    #     b3 = net.layer[3].bias
    #
    #     summarywriter.add_histogram('w0', w0, epoch)
    #     summarywriter.add_histogram('b0', b0, epoch)
    #     summarywriter.add_histogram('w1', w1, epoch)
    #     summarywriter.add_histogram('b1', b1, epoch)
    #     summarywriter.add_histogram('w2', w2, epoch)
    #     # summarywriter.add_histogram('b2', b2, epoch)
    #     summarywriter.add_histogram('w3', w3, epoch)
    #     summarywriter.add_histogram('b3', b3, epoch)
    #
    #     for idx, (img, label) in enumerate(train_dataloader):
    #         summarywriter.add_images('img', img[:50])
    #         img, label = img.to(device),label.to(device)
    #         label = F.one_hot(label, 10).type(torch.float32)
    #         # print("label_shape>>>:",label.shape)
    #         predict_label = net(img)
    #         # print("predict_shape>>>:",predict_label.shape)
    #         loss = loss_fun(predict_label,label)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         if idx % 10 == 0:
    #             print(f'epoch:{epoch} -> {idx}/{len(train_dataloader)} -> loss: {loss.item()}')
    #             summarywriter.add_scalar('loss', loss.item(), global_step=epoch)
    #
    #         torch.save(net.state_dict(), model_path)

    print("split")
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    #                       打印所有层权重和梯度                 #
    #                Print all layer weights and gradients    #
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

    for epoch in range(10000):
        print("epoch = ", epoch)
        for name, param in net.named_parameters():  # 返回模型的参数
            # print("param.requires_grad>>>:",param.requires_grad)
            #把打印放在前面首先要判断是否有梯度，因为初始是没有办法计算梯度的
            if param.requires_grad:
                if param.grad is not None:
                    summarywriter.add_histogram(name + '_grad', param.grad.mean(), epoch)
                    print("{}, gradient: {}".format(name, param.grad.mean()))
                else:
                    print("{} has not gradient".format(name))
            # summarywriter.add_histogram(name + '_grad', param.grad, epoch)  # 参数的梯度
            #画权重直方图
            summarywriter.add_histogram(name + '_data', param, epoch)  # 参数的权值

        for idx, (img, label) in enumerate(train_dataloader):
            # print(img.shape)
            # exit()
            img, label = img.to(device), label.to(device)
            label = F.one_hot(label, 10).type(torch.float32)
            grid = torchvision.utils.make_grid(img)
            #绘图
            summarywriter.add_image("images",grid,0)
            summarywriter.add_graph(net, img)
            # summarywriter.add_images('img', img[:50])

            # print("label_shape>>>:",label.shape)
            predict_label = net(img)
            # print("predict_shape>>>:",predict_label.shape)
            loss = loss_fun(predict_label, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if idx % 10 == 0:
                print(f'epoch:{epoch} -> {idx}/{len(train_dataloader)} -> loss: {loss.item()}')
                summarywriter.add_scalar('loss', loss.item(), global_step=epoch)

            torch.save(net.state_dict(), model_path)