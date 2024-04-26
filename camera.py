import cv2
import numpy as np
import pickle
import torch
from torch import nn
from MLModel.model_train import SignLanguageCNN

ALPHABET = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

class VideoCamera(object):
    def __init__(self):
        self.model = SignLanguageCNN()
        checkpoint = torch.load("./MLModel/Model_test.pt")
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        # releasing camera
        self.video.release()


    def predict(self, frame):
        resized_frame = cv2.resize(frame, (28, 28))  # resize
        gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_RGB2GRAY)  # grayscale
        resized_frame = cv2.resize(gray_frame, (224, 224)) / 255.0
        tensor_frame = torch.FloatTensor(resized_frame).unsqueeze(0).unsqueeze(0).to(self.device)

        self.model.to(self.device)

        with torch.no_grad():
            outputs=self.model(tensor_frame.to("cuda"))
        
        return outputs
    
    def get_frame(self):
        # extracting frames
        (rval, im) = self.video.read()
        cv2.rectangle(im, (208, 128), (432, 352), (0, 0, 255), 2)
        cropped = im[208:432, 128:352]
        predictions = self.predict(cropped)
        # print(predictions)
        predicted = torch.softmax(predictions,dim=1)
        _,predicted = torch.max(predicted, 1)
        label = predicted.data.cpu().numpy()[0] 


        ret, jpeg = cv2.imencode('.jpg', im)
        return jpeg.tobytes(), ALPHABET[label]
