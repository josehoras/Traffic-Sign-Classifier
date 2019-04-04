## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
This project creates and train a deep convolutional neural network to classify traffic signs. It uses the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). Additionally the model is tested on images of German traffic signs found on the web and from pictures taken in my neighbourhood.

The deliverables for the project are:

-     [Ipython notebook](./Traffic_Sign_Classifier.ipynb) with code 
-     [HTML output](./Traffic_Sign_Classifier.html) of the code
-     [A writeup report](./writeup.md) (markdown)

Check out the [writeup](./writeup.md) for a detailed discussion on steps, challenges and results encountered in this project.


The Project
---
The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

### Dependencies
This project requires Python3 will the following dependencies:

- [TensorFlow](http://tensorflow.org)
- [NumPy](http://www.numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/)
- [PIL](http://www.pythonware.com/products/pil/)
- [OpenCV](http://opencv.org/)
- [Pickle](https://docs.python.org/3.5/library/pickle.html)
- [cvs](https://docs.python.org/3/library/csv.html)
- [Jupyter](http://jupyter.org/)

### Dataset

A pickled version of the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset)  with images resized to 32x32 is available [here](https://d17h27t6h515a5.cloudfront.net/topher/2016/October/580d53ce_traffic-sign-data/traffic-sign-data.zip).
For dataset augmentation, run `python data_augmentation.py`

