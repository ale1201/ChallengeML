import cv2
import numpy as np
tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
modelo = 'modelo.h5'
pesos_modelo = 'pesos.h5'
cnn = load_model(modelo)
cnn.load_weights(pesos_modelo)
def predict(file):
  x = cv2.resize(file, dsize=(180, 180))
  x = keras.preprocessing.image.img_to_array(x)
  x = np.expand_dims(x, axis=0)
  array = cnn.predict(x)
  result = array[0]
  answer = np.argmax(result)
  if answer == 0:
    answer = 3
    print("pred: crosswalk (3)")
  elif answer == 1:
    answer = 2
    print("pred: speedlimit (2)")
  elif answer == 2:
    answer = 1
    print("pred: stop (1)")
  elif answer == 3:
    answer = 0
    print("pred: trafficlight (0)")
  else:
    print (answer)

  return answer

vc = cv2.VideoCapture(0)

if vc.isOpened():
  is_capturing, frame = vc.read()
  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

else:
  is_capturing = False

while is_capturing:
  try:
    is_capturing, frame = vc.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    predict(frame)
    webcam_preview = plt.imshow(frame)
    try:
      plt.pause(5)
    except Exception:
      pass
  except KeyboardInterrupt:
    vc.release()