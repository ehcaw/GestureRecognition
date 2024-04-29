import time
import pandas as pd
import cv2
from sklearn.metrics import f1_score
import torch
from torch import nn
from torch.utils.data import Dataset
import numpy as np


class LoadDataset(Dataset):
    def __init__(self, csv, train=True):
        # read from the given dataset with pandas
        self.csv = pd.read_csv(csv)
        self.train = train
        # split label and pixels
        self.images = torch.zeros((self.csv.shape[0], 1))
        # for each column of pixels, we turn it into a tensor with dimension 1 and concatenate to images
        for i in range(1, 785):
            temp_text = "pixel" + str(i)
            temp = self.csv[temp_text]
            temp = torch.FloatTensor(temp).unsqueeze(1)
            self.images = torch.cat((self.images, temp), 1)
        self.labels = self.csv["label"]
        self.images = self.images[:, 1:]
        # keep all the rows but convert the pixels of each row into 28x28 2D image
        self.images = self.images.view(-1, 28, 28)

    # take row number of the image
    def __getitem__(self, index):
        # grab the image and convert it to a NumPy array
        image = self.images[index]
        image = image.numpy()
        image = cv2.resize(image, (224, 224))
        # convert back into float tensors and add an extra dimension
        tensor_image = torch.FloatTensor(image).unsqueeze(0)
        # keep grayscale values between 0 and 1 to be more efficient
        tensor_image = tensor_image / 255
        # if we are using the training data then we also want to keep what letter the image is supposed to be
        if self.train:
            return tensor_image, self.labels[index]
        else:
            return tensor_image

    def __len__(self):
        return self.images.shape[0]


class SignLanguageCNN(nn.Module):
    def __init__(self):
        super(SignLanguageCNN, self).__init__()
        # first layer
        self.Conv1 = nn.Sequential(nn.Conv2d(1,32,5),nn.MaxPool2d(2),nn.ReLU(),nn.BatchNorm2d(32))
        # second layer
        self.Conv2 = nn.Sequential(nn.Conv2d(32,64,5),nn.MaxPool2d(2),nn.ReLU(),nn.BatchNorm2d(64))
        # third layer
        self.Conv3 = nn.Sequential(nn.Conv2d(64,128,3),nn.MaxPool2d(2),nn.ReLU(),nn.BatchNorm2d(128))
        # fourth layer
        self.Conv4 = nn.Sequential(nn.Conv2d(128,256,3),nn.MaxPool2d(2),nn.ReLU(),nn.BatchNorm2d(256))
        # fifth layer
        self.Conv5 = nn.Sequential(nn.Conv2d(256,512,3),nn.MaxPool2d(2),nn.ReLU(),nn.BatchNorm2d(512))

        self.Linear1 = nn.Linear(512*4*4,256)
        self.dropout = nn.Dropout(0.1)
        self.Linear3 = nn.Linear(256,25)

    def forward(self, x):
        x = self.Conv1(x)
        x = self.Conv2(x)
        x = self.Conv3(x)
        x = self.Conv4(x)
        x=self.dropout(x)
        x = self.Conv5(x)
        x = x.view(x.size(0), -1)
        x = self.Linear1(x)
        x = self.dropout(x)
        x = self.Linear3(x)
        return x

#function to validate and check the accuracy and f1 score of the model
# we take the model and run the test dataset through
def validate(val_loader,model):
    model.eval()
    test_labels = [0]
    test_pred = [0]
    for i, (images,labels) in enumerate(val_loader):
        outputs = model(images.to("cuda"))
        predicted = torch.softmax(outputs,dim=1)
        _,predicted = torch.max(predicted, 1)
        test_pred.extend(list(predicted.data.cpu().numpy()))
        test_labels.extend(list(labels.data.cpu().numpy()))

    test_pred = np.array(test_pred[1:])
    test_labels = np.array(test_labels[1:])
    correct = (test_pred==test_labels).sum()
    accuracy = correct/len(test_labels)
    f1_test = f1_score(test_labels,test_pred,average="weighted")
    model.train()
    return accuracy,f1_test

def train():
    # load data from train and test files
    train_data = LoadDataset("dataset/sign_mnist_train.csv")
    test_data = LoadDataset("dataset/sign_mnist_test.csv")

    # use pytorch dataloader to create images for training
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=128, num_workers=4, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=64, num_workers=0, shuffle=True)

    model=SignLanguageCNN()
    model=model.to("cuda")
    model.train()
    checkpoint=None
    learning_rate=1e-3
    #train for 20 epochs
    start_epoch=0
    end_epoch=20
    criterion = nn.CrossEntropyLoss().to("cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5, min_lr=1e-6)
    if checkpoint:
        model.load_state_dict(torch.load(checkpoint)['state_dict'])
        start_epoch=torch.load(checkpoint)['epoch']
    for epoch in range(start_epoch,end_epoch+1):
        start_time = time.time()
        # loads a group of images from the DataLoader through the model
        for i, (images,labels) in enumerate(train_loader):
            outputs = model(images.to("cuda"))
            loss = criterion(outputs.to("cuda"),labels.to("cuda"))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            predicted = torch.softmax(outputs,dim=1)
            _,predicted = torch.max(predicted, 1)
            f1 = f1_score(labels.cpu().numpy(),predicted.cpu().numpy(),average="weighted")
        end_time = time.time()
        test_accuracy,test_f1 = validate(test_loader,model)
        print("------------------------------------------------------------------------------------------------------")
        print(f"Epoch {epoch}/{end_epoch}, Training F1: {f1:.4f}, Validation Accuracy: {test_accuracy:.4f}, Validation F1: {test_f1:.4f}, Time: {end_time-start_time:.4f}")
        scheduler.step(test_accuracy)

    # Save the model
    #saves all part of the model so you can continue training
    torch.save({
    'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict()},"SignLanguageModel.pt")

if __name__ == '__main__':
    train()

