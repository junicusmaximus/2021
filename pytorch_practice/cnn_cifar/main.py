import torch
from torchvision import transforms, datasets
import torch.nn as nn

# device configuration

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#hyperparameters

batch_size = 32
epochs = 100
lr = 1e-3

#cifar data download & split between train/test

train_dataset = datasets.CIFAR10(root = '../data/CIFAR_10',
                                 train = True,
                                 download = True,
                                 transform = transforms.ToTensor())

test_dataset = datasets.CIFAR10(root = '../data/CIFAR_10',
                                train = False,
                                transform = transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                           batch_size = batch_size,
                                           shuffle = True)

test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                          batch_size = batch_size,
                                          shuffle = True)

# CNN model

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3,8,3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2,2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(8,16,3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2,2))


        self.layer3 = nn.Sequential(
            nn.Conv2d(16,32,3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2,2))

        self.fc = nn.Linear(4 * 4 * 32, 10)


    def forward(self, x):

        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0),-1)
        out = self.fc(out)
        return out

model = CNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# ?????? training

total_step = len(train_loader)

for epoch in range(epochs):
    for i,(images,labels) in enumerate(train_loader): #i??? ????????????
        images = images.to(device) #train_loader??? ?????????
        labels = labels.to(device) #train_loader??? label
        #forward pass
        outputs = model(images) #train_loader??? ????????? ????????? ?????? ??????
        loss = criterion(outputs,labels) #train_loader??? (?????? ????????? outputs??? ????????? label ??????)

        #backward and optimize
        optimizer.zero_grad()
        loss.backward() #backward ??????
        optimizer.step() #step?????? update??????

        if (i+1) % 1563 == 0: #step??? 1563??? ??????
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.3f}'
                  .format(epoch + 1, epochs, i + 1, total_step, loss.item()))

# ?????? evaluate(test)

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
        correct += (prediction == labels).sum().item() #predicition??? ?????? labels??? ????????? correct??? ?????? ??????

    print('Test Accuracy: {:.3f} %'.format(100 * correct / total))

