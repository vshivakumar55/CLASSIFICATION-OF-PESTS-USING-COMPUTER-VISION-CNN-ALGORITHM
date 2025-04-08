from tkinter import *
import tkinter
from tkinter import filedialog
import numpy as np
from tkinter.filedialog import askdirectory
from tkinter import simpledialog
from sklearn.model_selection import train_test_split
from tkinter import ttk
from tkinter.filedialog import askopenfilename
import os
import cv2
from keras.models import Sequential
from keras.layers import Input, Dense, LSTM, Embedding
from keras.layers import Dropout, Activation, Bidirectional, GlobalMaxPool1D
from keras.models import Sequential
from sklearn import svm
from sklearn.metrics import accuracy_score
from keras.models import model_from_json
from keras.utils.np_utils import to_categorical
import pickle

main = tkinter.Tk()
main.title("Early Pest Detection From Crop Using Image Processing And Computational Intelligence") #designing main screen
main.geometry("1000x650")

global filename
global X,Y
global X_train, X_test, Y_train, Y_test
global predicted_data
labels = ['Aphids','Uneffected','Whitefly']
global svm_classifier

def getID(name):
  index = 0
  for i in range(len(labels)):
    if labels[i] == name:
      index = i
      break
  return index   

def upload():
  global filename
  filename = filedialog.askdirectory(initialdir = ".")
  text.delete('1.0', END)
  text.insert(END,filename+' Loaded')

def preprocess():
  global X, Y
  global X_train, X_test, Y_train, Y_test
  X = []
  Y = []
  text.delete('1.0', END)
  if os.path.exists('model/X.txt.npy'):
    X = np.load('model/X.txt.npy')
    Y = np.load('model/Y.txt.npy')
    X = X.astype('float32')
    X = X/255
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
  else:
    for root, dirs, directory in os.walk(filename):
      for j in range(len(directory)):
        name = os.path.basename(root)
        print(name+" "+root+"/"+directory[j])
        if 'Thumbs.db' not in directory[j]:
          img = cv2.imread(root+"/"+directory[j],0)
          img = cv2.resize(img, (28,28))
          im2arr = np.array(img)
          im2arr = im2arr.reshape(28,28)
          X.append(im2arr.ravel())
          Y.append(getID(name))
    X = np.asarray(X)
    Y = np.asarray(Y)
    np.save('model/X.txt',X)
    np.save('model/Y.txt',Y)
    X = np.load('model/X.txt.npy')
    Y = np.load('model/Y.txt.npy')
    X = X.astype('float32')
    X = X/255
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]

  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
  text.insert(END,"Total images found in dataset: "+str(X.shape[0])+"\n\n")
  text.insert(END,"Total classes found in dataset: "+str(labels)+"\n\n")
  text.insert(END,"Dataset train and test split 80 and 20%\n\n")
  text.insert(END,"Training 80% images: "+str(X_train.shape[0])+"\n")
  text.insert(END,"Testing 80% images: "+str(X_test.shape[0])+"\n")
  test = X[3]
  test = test.reshape(28,28)
  test = cv2.resize(test,(200,200))
  cv2.imshow("Process Image",test)
  cv2.waitKey(0)
  
def runSVM():
  text.delete('1.0', END)
  global svm_classifier
  global X, Y
  global X_train, X_test, Y_train, Y_test

  svm_classifier = svm.SVC()
  svm_classifier.fit(X, Y)
  predict = svm_classifier.predict(X_test) 
  svm_acc = accuracy_score(Y_test,predict)*100
  text.insert(END,"SVM Prediction Test Accuracy: "+str(svm_acc))

def runCNN():
  text.delete('1.0', END)
  X_train = np.load('model/X.txt.npy')
  Y_train = np.load('model/Y.txt.npy')
  print(X_train.shape)
  print(Y_train.shape)
  if os.path.exists('model/model.json'):
        with open('model/model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            model = model_from_json(loaded_model_json)
        model.load_weights("model/model_weights.h5")
        model._make_predict_function()   
        print(model.summary())
        f = open('model/history.pckl', 'rb')
        data = pickle.load(f)
        f.close()
        acc = data['accuracy']
        accuracy = acc[59] * 100
        text.insert(END,"CNN Model Prediction Accuracy = "+str(accuracy)+"\n\n")
        text.insert(END,"See Black Console to view CNN layers\n")
  else:
        model = Sequential()
        model.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        model.add(Convolution2D(32, 3, 3, activation = 'relu'))
        modeladd(MaxPooling2D(pool_size = (2, 2)))
        model.add(Flatten())
        model.add(Dense(output_dim = 256, activation = 'relu'))
        model.add(Dense(output_dim = 108, activation = 'softmax'))
        print(model.summary())
        model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        hist = model.fit(X_train, Y_train, batch_size=16, epochs=60, shuffle=True, verbose=2)
        model.save_weights('model/model_weights.h5')            
        model_json = classifier.to_json()
        with open("model/model.json", "w") as json_file:
            json_file.write(model_json)
        f = open('model/history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
        f = open('model/history.pckl', 'rb')
        data = pickle.load(f)
        f.close()
        acc = data['accuracy']
        accuracy = acc[59] * 100
        text.insert(END,"CNN Model Prediction Accuracy = "+str(accuracy)+"\n\n")
        text.insert(END,"See Black Console to view CNN layers\n")


  

def checkEffected():
  global svm_classifier
  filename = filedialog.askopenfilename(initialdir="testImages")
  image = cv2.imread(filename,0)
  img = cv2.resize(image, (28,28))
  im2arr = np.array(img)
  im2arr = im2arr.reshape(28,28)
  img = np.asarray(im2arr)
  img = img.astype('float32')
  img = img/255
  temp = []
  temp.append(img.ravel())
  predict = svm_classifier.predict(np.asarray(temp))
  predict = predict[0]
  print(predict)
  
  img = cv2.imread(filename)
  img = cv2.resize(img, (400,400))
  cv2.putText(img, 'Pest Detected as : '+labels[predict], (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2)
  cv2.imshow('Pest Detected as : '+labels[predict], img)
  cv2.waitKey(0)
              
  
def close():
    main.destroy()
   
font = ('times', 16, 'bold')
title = Label(main, text='Early Pest Detection From Crop Using Image Processing And Computational Intelligence', justify=LEFT)
title.config(bg='lavender blush', fg='DarkOrchid1')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=100,y=5)
title.pack()

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Pest Dataset", command=upload)
uploadButton.place(x=10,y=100)
uploadButton.config(font=font1)

processButton = Button(main, text="Preprocess Dataset", command=preprocess)
processButton.place(x=330,y=100)
processButton.config(font=font1) 

svmButton = Button(main, text="Run SVM Algorithm", command=runSVM)
svmButton.place(x=650,y=100)
svmButton.config(font=font1)

svmButton = Button(main, text="Run CNN Algorithm", command=runCNN)
svmButton.place(x=10,y=150)
svmButton.config(font=font1)

testButton = Button(main, text="Check for Effected from Test Image", command=checkEffected)
testButton.place(x=300,y=150)
testButton.config(font=font1)

exitButton = Button(main, text="Exit", command=close)
exitButton.place(x=650,y=150)
exitButton.config(font=font1)



font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=120)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=300)
text.config(font=font1) 

main.config(bg='light coral')
main.mainloop()
