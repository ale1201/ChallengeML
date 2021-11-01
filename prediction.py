import cv2
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
modelo = './modelo.h5'
pesos_modelo = './pesos.h5'
cnn = load_model(modelo)
cnn.load_weights(pesos_modelo)
def predict(file):
  x = load_img(file, target_size=(180, 180))
  x = img_to_array(x)
  x = np.expand_dims(x, axis=0)
  array = cnn.predict(x)
  result = array[0]
  answer = np.argmax(result)
  if answer == 0:
    answer = 3
    print("     -     pred: crosswalk (3)")
  elif answer == 1:
    answer = 2
    print("     -     pred: speedlimit (2)")
  elif answer == 2:
    answer = 1
    print("     -     pred: stop (1)")
  elif answer == 3:
    answer = 0
    print("     -     pred: trafficlight (0)")
  else:
    print (answer)

  return answer

vc = cv2.VideoCapture(0)

if vc.isOpened():
  is_capturing, frame = vc.read()
  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

else:
  is_capturing = False


predict(frame)