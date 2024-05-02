# Gesture Recognition for Interpreting Sign Language

## Team SignLang; Schuyler Ng, Gael Gil, Ryan Nguyen

### schuyler.ng@sjsu.edu, gael.gil@sjsu.edu, ryan.c.nguyen@sjsu.edu

#### There aren’t many transcription services available, especially for transcribing sign language. If you wanted to improve your sign language by attempting to process actual sign language users, it may be inefficient as there isn’t something immediately available to compare it to and have it register in your brain that this hand gesture correlates with this word.

### Create software that allows you to transcribe sign language in real time

### Machine learning / Computer Vision / GUI

## Plan:

#### 1. Ensure that webcam works in the program

#### 2. Detect hand region

#### 3. Train model on various signs

#### 4. Create a GUI for showing the transcription

#### 5. Test features

#### 6. Create presentation and present

## MVP:

#### 1. Obtain data set

#### 2. Try to train model on data set

#### 3. Perform data analysis after testing the model

#### 4. Present analysis

## Description<br>
We have created a webapp that takes live images of sign language symbols and tries to read and predict the letter being displayed

## To run:
1. Install required packages from requirements.txt<br>
2. Run app.py and try to put hand in red box

## To train model:

1. Install pytorch with [cuda](https://pytorch.org/)
2. Install required packages from requirements.txt<br>
3. Open model_train.py and set model_file_name
4. Run model_train.py

## Presentation<br>
The presentation can be found [here](https://docs.google.com/presentation/d/1v8K_vxPH-3cagHdaWpYkzuoqBcduskv5j-9c_nP6eZ4/edit?usp=sharing)

## Credits:
Dataset provided from [this](https://www.kaggle.com/datasets/datamunge/sign-language-mnist) competition on Kaggle<br>
Training model adapted from [Vijay Vignesh P](https://www.kaggle.com/code/vijaypro/cnn-pytorch-96/notebook)