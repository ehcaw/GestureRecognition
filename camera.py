import cv2
import numpy as np
import pickle
import torch
from torch import nn

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

        # capturing video
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        # releasing camera
        self.video.release()


    def predict(self, frame):
        # Preprocess the frame
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized_frame = cv2.resize(gray_frame, (32, 32))  # Resize to match the expected input size
        tensor_frame = torch.tensor(resized_frame, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)

        # Move model parameters to the same device as the input tensor
        self.model.to(self.device)

        # Pass the frame through the model's Conv1 layer
        with torch.no_grad():
            output = self.model.Conv1(tensor_frame)

        return output
    
    def get_frame(self):
        # extracting frames
        (rval, im) = self.video.read()

        predictions = self.predict(im)
        print(predictions)
        # im = cv2.flip(im, 1, 1)
        # mini = cv2.resize(im, (im.shape[1] // size, im.shape[0] // size))
        # faces = classifier.detectMultiScale(mini)
        # for f in faces:
        #     (x, y, w, h) = [v * size for v in f] #Scale the shapesize backup
        #     #Save just the rectangle faces in SubRecFaces
        #     face_img = im[y:y+h, x:x+w]
        #     resized=cv2.resize(face_img,(300,300))
        #     normalized=resized/255.0
        #     reshaped=np.reshape(normalized,(1,300,300,3))
        #     reshaped = np.vstack([reshaped])
        #     K.set_session(session1)
        #     with graph1.as_default():
        #         results=loaded_model.predict(reshaped)
        #     if results >.5:
        #         result = np.array([[1]])
        #     else:
        #         result = np.array([[0]])
        #     label = np.argmax(result)
        #     cv2.rectangle(im,(x,y),(x+w,y+h),color_dict[result[label][0]],2)
        #     cv2.rectangle(im,(x,y-40),(x+w,y),color_dict[result[label][0]],-1)
        #     cv2.putText(im, labels_dict[result[label][0]], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        #     # encode OpenCV raw frame to jpg and displaying it
        ret, jpeg = cv2.imencode('.jpg', im)
        return jpeg.tobytes()