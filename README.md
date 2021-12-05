# CS584_Group23_Openmax_Algorithm_towards_OpenSetRecognition_in_Deep_Learning
This is a Machine Learning CS584 Project done by Purva and Pragati

**Steps to run the code:**

1. Train a CNN model for the dataset of your choice
2. Load the trained model
3. Load the training data you trained the DNN model
4. Create a mean activation vector (MAV) and perform weibull fit model
5. Pass the sample to compute openmax and evaluate the output from openmax, original label, and softmax
6. Test the trained openmax to images from different distribution


**Python libraries required :-**
       _**Use command :- python -m pip install library_name**_
1. matplotlib
2. tensorflow
3. keras
4. scipy
5. opencv-python
6. cython
7. cmake
8. libmr


**Execution Steps :-**
1. Execute the CIFAR10_image_classification.py
2. Build libmr using cmake
3. After the libmr is built, two files are generated in examples folder of libmr.
  - a. Execute python_examples.py to test if your libmr is built properly.
  - b. Execute python_exaple_plot.py to get the tailfitting plot graph for weibull distribution with different tail sizes.
5. Execute openmax_mnist_test.py
