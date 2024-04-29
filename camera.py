import cv2
import torch
from MLModel.model_train import SignLanguageCNN

ALPHABET = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

class VideoCamera(object):
    def __init__(self):
        self.model = SignLanguageCNN()
        #checkpoint = torch.load("./MLModel/Model_test.pt")
        checkpoint = torch.load("./MLModel/Model_test.pt", map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        # releasing camera
        self.video.release()


    def predict(self, frame):
        resized_frame = cv2.resize(frame, (28, 28))  # resize
        #resize back up to 224x224 converting all values to be between 0 and 1
        resized_frame = cv2.resize(resized_frame, (224, 224)) / 255.0
        # finally convert to 4D tensors which then get imput into the model
        tensor_frame = torch.FloatTensor(resized_frame).unsqueeze(0).unsqueeze(0).to(self.device)

        self.model.to(self.device)

        with torch.no_grad():
            output=self.model(tensor_frame.to("cuda") if torch.cuda.is_available() else tensor_frame.to("cpu"))
        
        return output
    
    def get_frame(self):
        # extracting frames
        (rval, im) = self.video.read()
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # grayscale
        # crop down to a 224x224 square in the center of the webcam
        #cv2.rectangle(im, (208, 128), (432, 352), (0, 0, 255), 2)
        cv2.rectangle(im, (208, 128), (656, 576), (0, 0, 255), 2)
        cropped = im[208:432, 128:352]
        predictions = self.predict(cropped)
        # print(predictions)
        predicted = torch.softmax(predictions,dim=1)
        _,predicted = torch.max(predicted, 1)
        label = predicted.data.cpu().numpy()[0] 


        ret, jpeg = cv2.imencode('.jpg', im)
        return jpeg.tobytes(), ALPHABET[label]
    
    def get_label(self):
        # extracting frames
        (rval, im) = self.video.read()
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # grayscale
        # crop down to a 224x224 square in the center of the webcam
        #cv2.rectangle(im, (208, 128), (432, 352), (0, 0, 255), 2)
        cv2.rectangle(im, (208, 128), (656, 576), (0, 0, 255), 2)
        cropped = im[208:432, 128:352]
        predictions = self.predict(cropped)
        # print(predictions)
        predicted = torch.softmax(predictions,dim=1)
        _,predicted = torch.max(predicted, 1)
        label = predicted.data.cpu().numpy()[0] 

        return ALPHABET[label]
