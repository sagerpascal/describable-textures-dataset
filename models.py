import torch
import torch.nn as nn
import torch.nn.functional as F


#################################################################################################
# Simple FCN
#################################################################################################
class SimpleFullyCnn(nn.Module):

    def __init__(self):
        super(SimpleFullyCnn, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1)
        self.conv5 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1)
        self.conv6 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1)
        self.conv7 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1)
        self.conv8 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1)
        self.conv9 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1)
        self.conv10 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1)
        self.conv11 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1)
        self.conv12 = nn.Conv2d(in_channels=8, out_channels=12, kernel_size=3, stride=1)
        self.conv13 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1)
        self.conv14 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1)
        self.conv15 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1)
        self.conv16 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1)
        self.conv17 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1)
        self.conv18 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1)
        self.conv19 = nn.Conv2d(in_channels=12, out_channels=16, kernel_size=3, stride=1)
        self.conv20 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1)
        self.conv21 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1)
        self.conv22 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1)
        self.conv23 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1)
        self.conv24 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1)
        self.conv25 = nn.Conv2d(in_channels=32, out_channels=46, kernel_size=3, stride=1)
        self.out = nn.Conv2d(in_channels=46, out_channels=46, kernel_size=1, stride=1)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = F.relu(self.conv13(x))
        x = F.relu(self.conv14(x))
        x = F.relu(self.conv15(x))
        x = F.relu(self.conv16(x))
        x = F.relu(self.conv17(x))
        x = F.relu(self.conv18(x))
        x = F.relu(self.conv19(x))
        x = F.relu(self.conv20(x))
        x = F.relu(self.conv21(x))
        x = F.relu(self.conv22(x))
        x = F.relu(self.conv23(x))
        x = F.relu(self.conv24(x))
        x = F.relu(self.conv25(x))
        return self.out(x)

    def get_mask_size(self):
        return 78, 78

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


#################################################################################################
# U-Net
#################################################################################################


