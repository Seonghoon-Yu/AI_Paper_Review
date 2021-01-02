import torch
import torch.nn as nn

# Inception_block과 conv_block을 이용해서 GoogLeNet class 정의하기
class GoogLeNet(nn.Module):
    def __init__(self,in_channels=3, num_classes=1000):
        super(GoogLeNet,self).__init__()

        # 기본 CNN 구조 start
        self.conv1 = conv_block(in_channels=in_channels, out_channels=64, kernel_size=(7,7),
                                stride=(2,2), padding=(3,3))
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = conv_block(64, 192, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Inception block start
        # in this order : in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool
        self.inception3a = Inception_block(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception_block(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception4a = Inception_block(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception_block(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception_block(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception_block(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception_block(428, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception5a = Inception_block(832, 256, 16-, 32-, 32, 128, 128)
        self.inception5b = Inception_block(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.dropout = nn.Dropout(p=0.4)
        self.fc1 = nn.Linear(1024, 1000)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.maxpool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)

        x = x.reshape(x.shape[0],-1)
        x = self.dropout(x)
        x = self.fc1(x)
        return x



# inception block 정의
# inception block은 다양한 kernell을 지닌 conv block를 사용
class Inception_block(nn.Module):
    # red_3x3은 3x3 filter 이전에 적용되는 1x1 filter에 의한 차원 축소 출력값
    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool):
        super(Inception_block, self).__init__()

        # 1x1 conv layer
        self.branch1 = conv_block(in_channels, out_1x1, kernel_size=1)

        # 1x1 conv + 3x3 conv
        self.branch2 = nn.Sequential(
            conv_block(in_channels, red_3x3, kernel_size=1),
            conv_block(red_3x3, out_3x3, kernel_size=3, padding=1)
        )

        # 1x1 conv + 5x5 conv
        self.branch3 = nn.Sequential(
            conv_block(in_channels, red_5x5, kernel_size=1),
            conv_block(red_5x5, out_5x5, kernel_size=5, padding=2)
        )

        # after maxPool layer, apply 1x1 conv
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            conv_block(in_channels, out_1x1pool, kernel_size=1)
        )

    def forward(self,x):
        # torch.cat 함수로 branch를 하나로 통합
        # N x filters x 28 x 28
        return torch.cat(self.branch1(x),self.branch2(x),self.branch3(x),self.branch4(x), 1)

# inception block에 사용될 conv_block 클래스를 생성해서 반복 사용
class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(conv_block, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs) # 1x1, 3x3, 5x5 filter
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self,x):
        return self.relu(self.batchnorm(self.conv(x)))

if __name__ == '__main__':
    x = torch.randn(3,3,224,224)
    model = GoogLeNet()
    print(x.shape)
