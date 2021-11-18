# !pip install numpy==1.16.1
import numpy as np
from openmax import *
import os
import glob

np.random.seed(12345)
model = load_model(os.path.join(os.getcwd(), "\saved_models\keras_cifar10_trained_model.h5"))
create_model(model)

X_test = [], y_test = []

path = glob.glob("images\*.png")
for imagepath in path:
    n = cv2.imread(imagepath)
    X_test.append(n)
    
y_test = ['CAT', 'CAT', 'DOG','UNKNOWN']
print(X_test)

for i in range(0,4): # Since we have 4 testing images

    test_x1 = X_test[i]
    test_y1 = y_test[i]

    image_show(test_x1, test_y1)

    np_load_old = np.load
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle = True, **k)

    # Compute fc8 activation for the given image
    activation = compute_activation(model, test_x1)

    # Compute openmax activation
    softmax,openmax = compute_openmax(model,activation)
    np.load = np_load_old

    print ('ACTUAL LABEL OF THE INPUT IMAGE : ', np.argmax(test_y1))
    print ('PREDICTED LABEL USING SOFTMAX ALGORITHM : ', softmax)
    if openmax == 10:
        openmax = 'UNKNOWN'
    print ('PREDICTED LABEL USING OPENMAX ALGORITHM : ',openmax)
    i = i + 1
