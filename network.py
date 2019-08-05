import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear_QNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(Linear_QNet, self).__init__()
        self.linear1 = nn.Linear(input_size, 120)
        self.linear2 = nn.Linear(120, 120)
        self.linear3 = nn.Linear(120, 120)
        self.linear4 = nn.Linear(120, output_size)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.15)
        self.softmax = nn.Softmax()
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        #x = self.linear2(x)
        #x = self.relu(x)
        #x = self.dropout(x)
        #x = self.linear3(x)
        #x = self.relu(x)
        #x = self.dropout(x)
        #print(x)
        x = self.linear4(x)
        #x = self.relu(x)
        x = self.softmax(x)
        return x

class QNet(nn.Module):

    def __init__(self, h, w, outputs):
        super(QNet, self).__init__()
        self.conv1 = nn.Conv2d(2, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)
        #print('################ linear_input_size', linear_input_size)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        #print('|||| x shape ||||', x.shape)
        return self.head(x)


class Linear_QNet2(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        #x = self.linear1(x)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
