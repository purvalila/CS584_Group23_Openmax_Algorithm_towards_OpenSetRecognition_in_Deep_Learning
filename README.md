# CS584_Group23_Openmax_Algorithm_towards_OpenSetRecognition_in_Deep_Learning
This is a Machine Learning CS584 Project done by Purva and Pragati

Steps to run the code:

Step 1: Train a CNN model for the dataset of your choice

Step 2: Load the trained model

Step 3: Load the training data you trained the DNN model

Step 4: Create a mean activation vector (MAV) and perform weibull fit model

Step 5: Pass the sample to compute openmax and evaluate the output from openmax, original label, and softmax

Step 6: Test the trained openmax to images from different distribution


Python libraries required :-

matplotlib
tensorflow
keras
scipy
opencv-python
cython
cmake
libmr


Execution Steps :-
1. Execute the CIFAR10_image_classification.py
2. Build libmr using cmake
3. Execute openmax_mnist_test.py
