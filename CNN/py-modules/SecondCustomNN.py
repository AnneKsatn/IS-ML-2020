#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch

class Custom_2(torch.nn.Module):
    def __init__(self):
        super(Custom_2, self).__init__()

        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=5, kernel_size=3, padding=2)
        self.act1  = torch.nn.Tanh()
        self.pool1 = torch.nn.AvgPool2d(kernel_size=2, stride=2)


        self.conv2 = torch.nn.Conv2d(in_channels=5, out_channels=6, kernel_size=4, padding=1)
        self.act2  = torch.nn.Tanh()
        self.pool2 = torch.nn.AvgPool2d(kernel_size=1, stride=1)

        self.conv3 = torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, padding=0)
        self.act3  = torch.nn.Tanh()
        self.pool3 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.fc1   = torch.nn.Linear(5 * 5 * 16, 120)
        self.act4  = torch.nn.Tanh()

        self.fc2   = torch.nn.Linear(120, 84)
        self.act5  = torch.nn.Tanh()

        self.fc3   = torch.nn.Linear(84, 10)
        self.act6  = torch.nn.Softmax(dim = 1)

    
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.act2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.act3(x)
        x = self.pool3(x)

        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))

        x = self.fc1(x)
        x = self.act4(x)
        x = self.fc2(x)
        x = self.act5(x)
        x = self.fc3(x)
        x = self.act6(x)

        return x

