import numpy as np
from openmax import *
import glob
import cv2
import os 
import scipy.misc
from PIL import Image  

np.random.seed(12345)
model = load_model(os.path.join(os.getcwd(), 'saved_models/keras_cifar10_trained_model.h5'))
create_model(model)

X_test_new = []
Y_test_new = []

path = glob.glob(os.path.join(os.getcwd(), 'images_MNIST/*.png'))
for imagepath in path:
    n = Image.open(imagepath)
    print(n.size)
    n = np.reshape(n,(28,28))
    print(n.shape)
    X_test_new.append(n)
    
Y_test_new = ['three', 'two', 'unknown','zero']

for i in range(0,4):

    test_x1 = X_test_new[i]
    test_y1 = Y_test_new[i]
    image_show(test_x1, test_y1)

    np_load_old = np.load
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle = True, **k)


    # Compute fc8 activation for the given image
    activation = compute_activation(model, test_x1)

    # Compute openmax 
    softmax,openmax = compute_openmax(model,activation)
    np.load = np_load_old

    print ('Actual Label: ', np.argmax(test_y1))
    print ('Prediction Softmax: ', softmax)
    if openmax == 10:
        openmax = 'Unknown'
    print ('Prediction openmax: ',openmax)
    i = i + 1