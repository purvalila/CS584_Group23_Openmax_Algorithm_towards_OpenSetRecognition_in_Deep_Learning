import numpy as np
# from tensorflow import keras
from openmax import *
import glob
import cv2
import os 
import scipy.misc
from PIL import Image  

np.random.seed(12345)
model = load_model('/Users/pragatikhekale/Desktop/Fall21/CS 584 Machine Learning/Project/OSDN_CIFAR/saved_models/keras_cifar10_trained_model.h5')
create_model(model)

X_test_new = []
Y_test_new = []

path = glob.glob("//Users/pragatikhekale/Desktop/Fall21/CS 584 Machine Learning/Project/OSDN_CIFAR/images_MNIST/*.png")
for imagepath in path:
    n = Image.open(imagepath)
    print(n.size)
    n = np.reshape(n,(28,28))
    print(n.shape)
    X_test_new.append(n)

# X_test_new.append(Image.open(path, 'zero.png'))
# X_test_new.append(Image.open(path, 'one.png'))
# X_test_new.append(Image.open(path, 'two.png'))
# X_test_new.append(Image.open(path, 'three.png'))
# X_test_new.append(Image.open(path, 'unknown.png'))
    
# X_test_new = X_test_new.reshape(X_test_new[0],28,28,1)
Y_test_new = ['three', 'two', 'unknown','zero']

for i in range(0,4):

    # random_char = np.random.randint(0,10000)
    test_x1 = X_test_new[i]
    test_y1 = Y_test_new[i]
    
    # out_images = np.array((test_x1))
    # img_x1 = out_images.reshape(out_images, 32,32,3)
    # test_y1.reshape(-1)

    image_show(test_x1, test_y1)

    np_load_old = np.load
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle = True, **k)


    # Compute fc8 activation for the given image

    activation = compute_activation(model, test_x1)
    #print (activation)


    # Compute openmax 

    softmax,openmax = compute_openmax(model,activation)
    #openmax_unknown_class(model)
    np.load = np_load_old

    print ('Actual Label: ', np.argmax(test_y1))
    print ('Prediction Softmax: ', softmax)
    if openmax == 10:
        openmax = 'Unknown'
    print ('Prediction openmax: ',openmax)
    i = i + 1

