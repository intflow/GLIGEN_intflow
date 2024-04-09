import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from torch.optim.lr_scheduler import StepLR
import cv2

# 데이터셋 로드 및 전처리 정의
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet 기대 입력 크기에 맞춰 조정
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 학습 및 테스트 데이터셋 로드
train_dataset = datasets.ImageFolder(root='./DATA/standard_crop_img', transform=transform)
test_dataset = datasets.ImageFolder(root='./DATA/standard_crop_img', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 모델 정의
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 56 * 56)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = Net()

# 손실 함수와 옵티마이저 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# 손실 함수와 옵티마이저, 스케줄러 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = StepLR(optimizer, step_size=7, gamma=0.1)  # 7 에포크마다 학습률을 0.1배 감소

# 모델 저장 함수
def save_model(model, save_path='best_model.pth'):
    torch.save(model.state_dict(), save_path)

# 모델 훈련 및 검증 함수
def train_model(model, train_loader, criterion, optimizer, num_epochs=25):
    best_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        scheduler.step()
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        
        print(f'Epoch {epoch+1}/{num_epochs} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
        # 가장 좋은 모델 저장
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            save_model(model)
            
    print('Training complete. Best accuracy: {:4f}'.format(best_acc))

# 모델 훈련
train_model(model, train_loader, criterion, optimizer, num_epochs=50)


image_path = "./DATA/standard_crop_img/brown/brown_3.jpg"
# 모델 eval
def eval_model(model, image_path):
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()  # 평가 모드로 설정
    
    color_dic = {0:"black", 1:"brown", 2:"white"}
    
    target_img = cv2.imread(image_path)
    target_img = cv2.resize(target_img,(224,224))
    target_img = torch.tensor(target_img,dtype=torch.float)
    target_img = target_img.permute(2,0,1)
    target_img = target_img.unsqueeze(0)
    output = model(target_img)
    # center_crop = torch.tensor(center_crop,dtype=torch.float)
    # center_crop = center_crop.permute(2,0,1)
    # center_crop = center_crop.unsqueeze(0)
    # output = model(center_crop)
    values, pred = torch.max(output,1)
    obj_color = color_dic[pred.item()]
    print(obj_color, pred, values)

eval_model(model, image_path)