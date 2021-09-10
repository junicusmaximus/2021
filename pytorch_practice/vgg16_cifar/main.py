import torch
from torchvision import transforms, datasets
import torch.optim as optim
import torch.nn as nn


device = 'cuda' if torch.cuda.is_available() else 'cpu'


#hyperparameters

batch_size = 16
epochs = 100
lr = 1e-3


train_dataset = datasets.CIFAR10(root='./data',
                                        train=True,
                                        download=True,
                                        transform = transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(train_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)

test_dataset = datasets.CIFAR10(root='./data',
                                       train=False,
                                       download=True,
                                       transform = transforms.ToTensor())

test_loader = torch.utils.data.DataLoader(test_dataset,
                                         batch_size=batch_size,
                                         shuffle=True)



class VGG16(nn.Module):

    def __init__(self):
        super(VGG16, self).__init__()

        # 첫번째층
        # ImagIn shape=(3,32,32)
        # if the kernel_size = 3, padding = 1 in the VGG paper

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2))

        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2))

        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2))

        self.layer5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2))

        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.reshape(out.shape[0], 512)
        out = self.classifier(out)
        return out


model = VGG16().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=lr)

total_step = len(train_loader)

for epoch in range(epochs):

    for i,(images,labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        #forward pass
        outputs = model(images)
        loss = criterion(outputs,labels)

        #backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 3125 == 0: #step을 3125번 하면
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.3f}'
                  .format(epoch + 1, epochs, i + 1, total_step, loss.item()))

# 모델 evaluate(test)

model.eval()
with torch.no_grad():
    correct = 0
    total = 0

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _,prediction = torch.max(outputs.data,1) #predicition
        total += labels.size(0) #test_loader dataset
        correct += (prediction == labels).sum().item() #predicition과 실제 labels가 맞으면 correct에 횟수 저장

    print('Test Accuracy: {:.3f} %'.format(100 * correct / total))
