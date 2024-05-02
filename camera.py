import cv2
import torch
from MLModel.model_train import SignLanguageCNN

# define the alphabet. This will be used to turn numerical outputs into labels
ALPHABET = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

class VideoCamera(object):
    def __init__(self):
        # load model class
        self.model = SignLanguageCNN() 
        # load the trained model
        checkpoint = torch.load("MLModel/SignLanguageModel.pt", map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.load_state_dict(checkpoint["state_dict"]) 
         # set model to eval mode
        self.model.eval()
         # use gpu if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # use opencv to get video from camera
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        # releasing camera
        self.video.release()


    def predict(self, frame):
        """
        function to predict from SignLanguageCNN()
        """
        # resize frame
        resized_frame = cv2.resize(frame, (28, 28))
        #resize back up to 224x224 converting all values to be between 0 and 1
        resized_frame = cv2.resize(resized_frame, (224, 224)) / 255.0
        # finally convert to 4D tensors which then get imput into the model
        tensor_frame = torch.FloatTensor(resized_frame).unsqueeze(0).unsqueeze(0).to(self.device)
        self.model.to(self.device)
        # pass frame to model to get output
        with torch.no_grad():
            output=self.model(tensor_frame.to("cuda") if torch.cuda.is_available() else tensor_frame.to("cpu"))
        return output # return output

    def get_frame(self):
        """
        function to get a frame from the video camera and return it to be used
        in app.py
        """
        # extracting frames
        (rval, im) = self.video.read()
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # grayscale
        return im #return the frame

    # grabs dimensions of the cropped image
    # outputs the top left point and the bottom right point of the cropped frame
    def get_dimensions(self,image):
        height, width = image.shape
        height //= 2
        width //= 2
        top_left_x = width - height // 2
        top_left_y = height // 2
        bottom_right_x = width + height // 2
        bottom_right_y = height + height // 2
        top_left = (top_left_x, top_left_y)
        bottom_right = (bottom_right_x, bottom_right_y)
        return top_left, bottom_right

    def convert_jpeg(self):
        image = self.get_frame()
        # dynamically setting a square in the middle of the frame
        # adds a rectangle to help user see where to put hand
        top_left, bottom_right = self.get_dimensions(image)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(image, top_left, bottom_right, (0, 0, 255), 2)
        ret, jpeg = cv2.imencode('.jpg', image) # change frame type
        return jpeg.tobytes() # return jpeg


    def get_label(self):
        """
        function to get a labels for an image
        """
        # extracting frames
        image = self.get_frame()
        top_left, bottom_right = self.get_dimensions(image)
        # image[y1:y2,x1:x2]
        cropped = image[top_left[1]:bottom_right[1],top_left[0]:bottom_right[0]] # crop the frame from the camera
        predictions = self.predict(cropped) # predict from a cropped frame
        predicted = torch.softmax(predictions,dim=1) # get predictions vector
        _,predicted = torch.max(predicted, 1) # get the max value predicted
        label = predicted.data.cpu().numpy()[0] # get the numerical prediction
        return ALPHABET[label] # return the letter label
