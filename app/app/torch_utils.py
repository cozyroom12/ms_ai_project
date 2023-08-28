import torch
import torchvision.transforms as transforms
from torchvision.models.efficientnet import efficientnet_b0
import torch.nn.functional as F
import torch.nn as nn

from PIL import Image


# 모델
# class ConvNet(nn.Module):
#     def __init__(self):  # layer 정의
#         super(ConvNet, self).__init__()

#         # input size = 28x28
#         self.conv1 = nn.Conv2d(1, 10, kernel_size=5)  # input channel = 1, filter = 10, kernel size = 5, zero padding = 0, stribe = 1
#         # ((W-K+2P)/S)+1 공식으로 인해 ((28-5+0)/1)+1=24 -> 24x24로 변환
#         # maxpooling하면 12x12

#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)  # input channel = 1, filter = 10, kernel size = 5, zero padding = 0, stribe = 1
#         # ((12-5+0)/1)+1=8 -> 8x8로 변환
#         # maxpooling하면 4x4

#         self.drop2D = nn.Dropout2d(p=0.25, inplace=False)  # 랜덤하게 뉴런을 종료해서 학습을 방해해 학습이 학습용 데이터에 치우치는 현상을 막기 위해 사용
#         self.mp = nn.MaxPool2d(2)  # 오버피팅을 방지하고, 연산에 들어가는 자원을 줄이기 위해 maxpolling
#         self.fc1 = nn.Linear(320, 100)  # 4x4x20 vector로 flat한 것을 100개의 출력으로 변경
#         self.fc2 = nn.Linear(100, 10)  # 100개의 출력을 10개의 출력으로 변경

#     def forward(self, x):
#         x = self.conv1(x)  # convolution layer 1번에 relu를 씌우고 maxpool, 결과값은 12x12x10
#         x = self.mp(x)  # convolution layer 1번에 relu를 씌우고 maxpool, 결과값은 12x12x10
#         x = F.relu(x)  # convolution layer 1번에 relu를 씌우고 maxpool, 결과값은 12x12x10
#         # x = F.relu(self.mp(self.conv1(x)))  # convolution layer 1번에 relu를 씌우고 maxpool, 결과값은 12x12x10

#         x = F.relu(self.mp(self.conv2(x)))  # convolution layer 2번에 relu를 씌우고 maxpool, 결과값은 4x4x20
#         x = self.drop2D(x)
#         x = x.view(x.size(0), -1)  # flat
#         x = self.fc1(x)  # fc1 레이어에 삽입
#         x = self.fc2(x)  # fc2 레이어에 삽입
#         return F.log_softmax(x)  # fully-connected layer에 넣고 logsoftmax 적용

# 모델 eval
PATH = r"C:\Users\labadmin\Desktop\app\pth\01_ex_best.pt"
model = efficientnet_b0(pretrained=True)
model.classifier[1] = nn.Linear(1280, 5)
model.eval()


# 예측
def get_prediction(image_tensor):
    outputs = model(image_tensor)

    _, predicted = torch.max(outputs.data, 1)
    return predicted


def preprocess_upload(image_bytes):
    input_transforms = [transforms.ToTensor(),]
    my_transforms = transforms.Compose(input_transforms)
    image = Image.open(image_bytes)
    timg = my_transforms(image)
    timg.unsqueeze_(0) # 차원 추가

    return timg