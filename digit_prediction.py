import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os,ssl,time

# setting the https context to fetch data from opneml 
if (not os.environ.get("PYTHONHTTPSVERIFY","") and getattr(ssl,"_create_unverified_context",None)):
    ssl._create_default_https_context = ssl._create_unverified_context


# Fetching data from openml
X,y = fetch_openml("mnist_784",version=1,return_X_y=True)
classes = ["0","1","2","3","4","5","6","7","8","9"]
n_classes = len(classes)


# Splitting the data for trainig and testing
x_train,x_test,y_train,y_test = train_test_split(X,y,random_state = 10,train_size = 7500,test_size = 2500)

x_test_scaled = x_test / 255
x_train_scaled = x_train / 255


# Fitting the training data to create Logistic Regression model
lr = LogisticRegression(solver = "saga",multi_class = "multinomial").fit(x_train_scaled,y_train)


# Calculating the accuracy of our Prediction model
y_predict = lr.predict(x_test_scaled)
accuracy = accuracy_score(y_test,y_predict)
print("Accuracy is ",accuracy)

capt = cv2.VideoCapture(0)

while (True):
    try:
        ret,frame = capt.read()

        grey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        # drawing a box in the center of the video
        height,width = grey.shape
        upper_left = (int(width/2 - 56),int(height/2 - 56))
        bottom_right = (int(width/2 + 56),int(height/2 + 56))
        cv2.rectangle(grey,upper_left,bottom_right,(0,255,0),2)
        
        # We will only consider the area inside the box for detecting the digit
        # ROI = Region of Interest
        roi = grey[upper_left[1]:bottom_right[1],upper_left[0]:bottom_right[0]]

        # Converting image to PIL format
        img_pil = Image.fromarray(roi)

        # Convert your grayscale image to L format which means each pixel
        #  is represented by a single value from 0,255
        img_bw = img_pil.convert("L")
        img_resized = img_bw.resize((28,28),Image.ANTIALIAS)
        img_resized_inverted = PIL.ImageOps.invert(img_resized)
        pixel_filter = 20
        minpixel = np.percentile(img_resized_inverted,pixel_filter)
        img_resized_inverted_scaled = np.clip(img_resized_inverted-minpixel,0,255)
        max_pixel = np.max(img_resized_inverted)
        img_resized_inverted_scaled = np.asarray(img_resized_inverted_scaled) / max_pixel
        test_sample = np.array(img_resized_inverted_scaled).reshape(1,784)
        test_predict = lr.predict(test_sample)
        print("Predicted number is ",test_predict)

        cv2.imshow("frame",grey)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    except Exception as e:
        pass


capt.release()
cv2.destroyAllWindows()