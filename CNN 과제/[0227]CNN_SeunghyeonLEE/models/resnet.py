import torch 
import torch.nn as nn
import torch.nn.functional as F 

import pandas as pd 
import numpy as np 

""" 
[Batch, Channel, 224, 224]의 input을 받았을 때, 
최종적으로 sigmoid activation function에 들어갈 값을 반환하는 ResBottleNeck, ResNet50 Class를 정의해주세요.
즉, 정의된 ResNet50의 output 값은 확률이 아니라 sigmoid에 들어가는 값이 됩니다.
"""

class ResBottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1) -> None:
        #### TO DO ####
        super(ResBottleNeck, self).__init__()
        self.expansion = 4  

        # 1x1 Conv
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # 3x3 Conv
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 1x1 Conv 
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        # Skip Connection (Residual Connection)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )
        ###############
                
    def forward(self, x):
        #### TO DO ####
        identity = self.shortcut(x)  # skip connection

        x = F.relu(self.bn1(self.conv1(x)))  # 1x1 Conv
        x = F.relu(self.bn2(self.conv2(x)))  # 3x3 Conv
        x = self.bn3(self.conv3(x))  # 1x1 Conv

        x += identity  # Residual Connection 
        x = F.relu(x)  # activation function
        ###############
        return x
    

class ResNet50(nn.Module):
    def __init__(self) -> None:
        #### TO DO ####
        super(ResNet50, self).__init__()
        self.in_channels = 64  # 첫번째 conv 이후 채널 수 설정

        # initial conv layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet layers
        self.layer1 = self._make_layer(64, 3)   # No stride=2 here
        self.layer2 = self._make_layer(128, 4, stride=2)  # conv3_x (stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)  # conv4_x (stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)  # conv5_x (stride=2)

        # 최종 출력층
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 7x7 → 1x1
        self.fc = nn.Linear(512 * 4, 1)  # Fully Connected Layer
        ###############
        pass
    
    # def _make_layer(self, out_channels, num_blocks, stride):
    #     layers = []

    #     # 첫 번째 블록 (stride 적용)
    #     layers.append(ResBottleNeck(self.in_channels, out_channels, stride))
    #     self.in_channels = out_channels * 4  # 채널 확장

    #     # 나머지 블록 (stride=1)
    #     for _ in range(1, num_blocks):
    #         layers.append(ResBottleNeck(self.in_channels, out_channels))

    #     return nn.Sequential(*layers)
    
    # def _make_layer(self, out_channels, num_blocks, stride=1):
    #     strides = [stride] + [1] * (num_blocks - 1)  # 첫 번째 블록만 stride 적용
    #     layers = []

    #     for i, stride in enumerate(strides):
    #         layers.append(ResBottleNeck(self.in_channels, out_channels, stride))
    #         if i == 0:  # 첫 번째 블록에서만 업데이트!
    #             self.in_channels = out_channels * 4  

    #     return nn.Sequential(*layers)
    
    def _make_layer(self, out_channels, num_blocks, stride=1):
        strides = [stride] + [1]*(num_blocks-1)  # Only first block may have stride=2
        layers = []
        for stride in strides:
            layers.append(ResBottleNeck(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * 4  # Update in_channels for next block
        return nn.Sequential(*layers)


    def forward(self, x):
        #### TO DO ####
        #initial conv
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        # ResNet 블록
        x = self.layer1(x)  # conv2_x
        x = self.layer2(x)  # conv3_x
        x = self.layer3(x)  # conv4_x
        x = self.layer4(x)  # conv5_x

        # 최종 출력
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        ###############
        return x
    
if __name__ == "__main__":
    block = ResBottleNeck(in_channels=64, out_channels=64, stride=1)
    sample_input = torch.randn(1, 64, 56, 56)  # conv2_x의 입력 크기
    output = block(sample_input)

    print(f"ResBottleNeck output shape: {output.shape}")  # 예상: torch.Size([1, 256, 56, 56])
 
if __name__ == "__main__":
    model = ResNet50()
    
    for batch_size in [1, 2, 4, 8]:  # 배치 크기 테스트
        sample_input = torch.randn(batch_size, 3, 224, 224)
        output = model(sample_input)
        print(f"Batch Size: {batch_size}, Output shape: {output.shape}")
        
if __name__ == "__main__":
    model = ResNet50()
    sample_input = torch.randn(1, 3, 224, 224)
    
    # `conv1`만 실행
    x = model.conv1(sample_input)
    x = model.bn1(x)
    x = F.relu(x)
    x = model.maxpool(x)

    print(f"After conv1 & maxpool: {x.shape}")  # 예상: torch.Size([1, 64, 56, 56])

if __name__ == "__main__":
    model = ResNet50()
    sample_input = torch.randn(1, 3, 224, 224)
    
    x = model.conv1(sample_input)
    x = model.bn1(x)
    x = F.relu(x)
    x = model.maxpool(x)

    x = model.layer1(x)  # conv2_x
    print(f"After conv2_x: {x.shape}")  # 예상: torch.Size([1, 256, 56, 56])

    x = model.layer2(x)  # conv3_x
    print(f"After conv3_x: {x.shape}")  # 예상: torch.Size([1, 512, 28, 28])

    x = model.layer3(x)  # conv4_x
    print(f"After conv4_x: {x.shape}")  # 예상: torch.Size([1, 1024, 14, 14])

    x = model.layer4(x)  # conv5_x
    print(f"After conv5_x: {x.shape}")  # 예상: torch.Size([1, 2048, 7, 7])
