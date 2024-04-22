import cv2
import numpy as np
import pickle
import torch
from torch import nn

ALPHABET = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
class SignLanguageCNN(nn.Module):
    def __init__(self):
        super(SignLanguageCNN, self).__init__()
        # first layer
        self.Conv1 = nn.Sequential(nn.Conv2d(1,32,5),nn.MaxPool2d(2),nn.ReLU(),nn.BatchNorm2d(32))
        # second layer
        self.Conv2 = nn.Sequential(nn.Conv2d(32,64,5),nn.MaxPool2d(2),nn.ReLU(),nn.BatchNorm2d(64))
        # third layer
        self.Conv3 = nn.Sequential(nn.Conv2d(64,128,3),nn.MaxPool2d(2),nn.ReLU(),nn.BatchNorm2d(128))
        #fourth layer
        self.Conv4 = nn.Sequential(nn.Conv2d(128,256,3),nn.MaxPool2d(2),nn.ReLU(),nn.BatchNorm2d(256))
        #fifth layer
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

class VideoCamera(object):
    def __init__(self):
        self.model = SignLanguageCNN()
        checkpoint = torch.load("./MLModel/SignLanguageModel.pt")
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        # releasing camera
        self.video.release()


    def predict(self, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # grayscale
        resized_frame = cv2.resize(gray_frame, (224, 224))  # resize
        tensor_frame = torch.tensor(resized_frame, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)

        self.model.to(self.device)

        with torch.no_grad():
            outputs=self.model(tensor_frame.to("cuda"))
        
        return outputs
    
    def get_frame(self):
        # extracting frames
        (rval, im) = self.video.read()

        predictions = self.predict(im)
        # print(predictions)
        predicted = torch.softmax(predictions,dim=1)
        _,predicted = torch.max(predicted, 1)
        label = predicted.data.cpu().numpy()[0] 


        ret, jpeg = cv2.imencode('.jpg', im)
        return jpeg.tobytes(), ALPHABET[label]
