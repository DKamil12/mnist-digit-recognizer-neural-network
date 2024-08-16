import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


model = tf.keras.models.load_model('handwritten_digits_recognizer.keras')

img_num = 1

while os.path.isfile(f'digits/{img_num}.png'):
    try:
        img = cv.imread(f'digits/{img_num}.png')[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        
        print(f'Predicted digit is {np.argmax(prediction)}')

        plt.imshow(img[0], cmap=plt.cm.binary, )
        plt.show()
    except Exception as e:
        print('Error', e)
    finally:
        img_num += 1



# loss, accuracy = model.evaluate(x_test, y_test)

# print(loss)
# print(accuracy)