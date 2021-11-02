import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test model on random input data.
floating_model = input_details[0]['dtype']
floating_model2  = input_details[0]['dtype'] == np.float32
input_shape = input_details[0]['shape']

def predict(file):
  x = cv2.resize(file, dsize=(input_shape[1], input_shape[2]))
  cv2.imshow('Frame', x)
  image = np.array(x, dtype=floating_model)


  image = np.expand_dims(image, axis=0)

  interpreter.set_tensor(input_details[0]['index'], image)

  interpreter.invoke()

  output_data = interpreter.get_tensor(output_details[0]['index'])
  results = output_data
  print(results)
  answer = np.argmax(results)
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

vc = cv2.VideoCapture(1)

if vc.isOpened():
  is_capturing, frame = vc.read()
  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

else:
  is_capturing = False

while is_capturing:
  try:
    is_capturing, frame = vc.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    webcam_preview = plt.imshow(frame)
    predict(frame)

    try:
      plt.pause(0.2)
    except Exception:
      pass
  except KeyboardInterrupt:
    vc.release()