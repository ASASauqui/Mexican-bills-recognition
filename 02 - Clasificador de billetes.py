# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 10:45:07 2022

@author: alans
"""

# ----------------Librerias----------------
# Determinadas
import numpy as np
import pandas

# Modelos
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Training and testing sets
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import KFold
# from sklearn.model_selection import RepeatedStratifiedKFold
# from sklearn.model_selection import cross_val_score

# Procesamiento de imágenes
import cv2




def trainModel():
    # Lectura de datos
        df = pandas.read_csv("datasetToUse/dataset.csv")
    
    # Acomodo de variables en X y Y
        X = df.loc[:,'0':'17']
        Y = df.loc[:,'18']
        X = X.values
        Y = Y.values
        
    # Entranamiento del modelo
        Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.30, random_state=0)
        model = LogisticRegression(random_state=0, multi_class='multinomial', solver='lbfgs',penalty='l2')
        model.fit(Xtrain, Ytrain)
        Ypred = model.predict(Xtest)
        
        return model
    
    


def camera(img, model):
    cameraLength, cameraWidth, cameraCh = img.shape
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
  
    ret, imgt = cv2.threshold(gray, 115, 255, cv2.THRESH_BINARY_INV)
  
    cv2.imshow("Image threshold", imgt)
    countours, hierarchy = cv2.findContours(imgt.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rectangles = [cv2.boundingRect(countour) for countour in countours]
    c = 0
    
    
    for rect in rectangles:
      if (rect[2] > 50 and rect[3] > 50) and (rect[2] < cameraWidth and rect[3] < cameraLength):
          rectSize = [rect[2], rect[3]]
          relation = np.min(rectSize) / np.max(rectSize)
          if  relation >= 0.3 and relation <= 0.7:
              imgn = img[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
              imgn = cv2.resize(imgn, (100, 100))
              imgnHSV = cv2.cvtColor(imgn, cv2.COLOR_BGR2HSV)
              
              c += 1
              cv2.imshow("Image rect", imgn)
              
              
              # Mascara
              lower = np.array([0,50,50])
              upper = np.array([180,255,255])
              mask_range = cv2.inRange(imgnHSV, lower, upper)
              img_mask = cv2.bitwise_and(imgn, imgn, mask=mask_range)
              img_mask = cv2.cvtColor(img_mask, cv2.COLOR_BGR2HSV)
              
          
              # Clasificar la imagen imgn
              nrows, ncols, nch = img_mask.shape
      
              Xi = np.reshape(img_mask, (nrows*ncols,3))
              Xi[Xi == 0] = -1
              
              h = np.histogram(Xi[:,0], bins=18, range=[0,180], density=True)[0] * 1000
              h = h.reshape(1,-1)
              print(h)
              
              if not np.any(np.isnan(h)):
                  ans = model.predict(h)
                  
                  # Escribir el resultado
                  if ans == 0:
                      cv2.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (255, 0, 0), 2)
                      cv2.putText(img, str('20 pesos'), (rect[0], rect[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3, cv2.LINE_AA)
                  elif ans == 1:
                      cv2.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (77, 27, 99), 2)
                      cv2.putText(img, str('50 pesos'), (rect[0], rect[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (77, 27, 99), 3, cv2.LINE_AA)
                  elif ans == 2:
                      cv2.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 0, 112), 2)
                      cv2.putText(img, str('100 pesos'), (rect[0], rect[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 122), 3, cv2.LINE_AA)
                  elif ans == 3:
                      cv2.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (4, 112, 0), 2)
                      cv2.putText(img, str('200 pesos'), (rect[0], rect[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (4, 112, 0), 3, cv2.LINE_AA)
    
    return img



# Obtener modelo
model = trainModel()


# Activar camara
cam = cv2.VideoCapture(0) # Indica que hay una camara
while True:
    val, img = cam.read()
    img = camera(img, model)
    cv2.imshow("Image funcion",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()