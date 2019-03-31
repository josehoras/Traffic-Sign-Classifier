### Step 0: Load The Data
# Load pickled data
import pickle

# TODO: Fill this in based on where you saved the training and testing data

training_file = "train.p"
validation_file = "valid.p"
testing_file = "test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

### Step 1: Dataset Summary & Exploration
### Replace each question mark with the appropriate value.
### Use python, pandas or numpy methods rather than hard coding the results
import numpy as np

# Number of training examples
n_train = y_train.shape[0]
# Number of validation examples
n_validation = X_valid.shape[0]
# Number of testing examples.
n_test = X_test.shape[0]
# What's the shape of an traffic sign image?
image_shape = X_train[0].shape
# How many unique classes/labels there are in the dataset.
n_classes = np.unique(train['labels']).shape[0]

print("Number of training examples =", n_train)
print("Number of validation examples =", n_validation)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.

# Open csv and place signnames in dictionary
import csv
reader = csv.reader(open('signnames.csv', mode='r'))
signs_dict = dict((rows[0], rows[1]) for rows in reader)

import matplotlib.pyplot as plt
# Visualizations will be shown in the notebook.
# n = 19000
# plt.imshow(X_train[n])
# plt.axis('off')
# plt.title(signs_dict[str(y_train[n])])
# plt.show()

### Step 2: Design and Test a Model Architecture
### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include
### converting to grayscale, etc.
### Feel free to use as many code cells as needed.

### Define your architecture here.
### Feel free to use as many code cells as needed.
import tensorflow as tf
from tensorflow.contrib.layers import flatten

def LeNet(x):
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1

    conv1_weights = tf.Variable(tf.truncated_normal((5, 5, 3, 6), mean=mu, stddev=sigma))
    conv1_strides = [1, 1, 1, 1]
    conv1_bias = tf.Variable(tf.zeros(6))

    conv2_weights = tf.Variable(tf.truncated_normal((5, 5, 6, 16), mean=mu, stddev=sigma))
    conv2_strides = [1, 1, 1, 1]
    conv2_bias = tf.Variable(tf.zeros(16))

    fc1_weights = tf.Variable(tf.truncated_normal((400, 120)))
    fc1_bias = tf.Variable(tf.zeros(120))
    fc2_weights = tf.Variable(tf.truncated_normal((120, 84)))
    fc2_bias = tf.Variable(tf.zeros(84))
    out_weights = tf.Variable(tf.truncated_normal((84, 43)))
    out_bias = tf.Variable(tf.zeros(43))

    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_layer = tf.nn.conv2d(x, conv1_weights, conv1_strides, 'VALID')
    conv1_layer = tf.nn.bias_add(conv1_layer, conv1_bias)
    # Activation.
    conv1_layer = tf.nn.relu(conv1_layer)
    #     print(conv1_layer.shape)
    # Pooling. Input = 28x28x6. Output = 14x14x6.
    pool1_layer = tf.nn.max_pool(conv1_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    # Layer 2: Convolutional. Output = 10x10x16.
    conv2_layer = tf.nn.conv2d(pool1_layer, conv2_weights, conv2_strides, 'VALID')
    conv2_layer = tf.nn.bias_add(conv2_layer, conv2_bias)
    # Activation.
    conv2_layer = tf.nn.relu(conv2_layer)
    #     print(conv2_layer.shape)
    # Pooling. Input = 10x10x16. Output = 5x5x16.
    pool2_layer = tf.nn.max_pool(conv2_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    #     print(pool2_layer.shape)
    # Flatten. Input = 5x5x16. Output = 400.
    flat = flatten(pool2_layer)
    #     print(flat.shape)
    # Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_layer = tf.add(tf.matmul(flat, fc1_weights), fc1_bias)
    # Activation.
    fc1_layer = tf.nn.relu(fc1_layer)
    # Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_layer = tf.add(tf.matmul(fc1_layer, fc2_weights), fc2_bias)
    # Activation.
    fc2_layer = tf.nn.relu(fc2_layer)
    # Layer 5: Fully Connected. Input = 84. Output = 10.
    logits = tf.add(tf.matmul(fc2_layer, out_weights), out_bias)
    return logits


### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected,
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 43)

### training pipeline
rate = 1e-4

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=rate)
training_operation = optimizer.minimize(loss_operation)

### Model evaluation
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()


def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset + BATCH_SIZE], y_data[offset:offset + BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


### Train the model
from sklearn.utils import shuffle
EPOCHS = 10
BATCH_SIZE = 128
iterations = 0
loss_series = []
train_acc_series = []
val_acc_series = []
val_acc_x = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    print("Num examples: ", num_examples)
    print("Training...")
    print()
    validation_accuracy = evaluate(X_valid, y_valid)
    val_acc_series.append(validation_accuracy)
    val_acc_x.append(iterations)
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            loss, _ = sess.run([loss_operation, training_operation], feed_dict={x: batch_x, y: batch_y})
            loss_series.append(loss)
            train_accuracy = evaluate(batch_x, batch_y)
            train_acc_series.append(train_accuracy)
            iterations += 1

        validation_accuracy = evaluate(X_valid, y_valid)
        val_acc_series.append(validation_accuracy)
        val_acc_x.append(iterations)
        print("EPOCH {} ({} iterations)...".format(i + 1, iterations))
        print("Loss = {}".format(loss))
        print("Training Accuracy = {:.3f}".format(train_accuracy))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()

    # saver.save(sess, './lenet')
    # print("Model saved")

f, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))
ax1.plot(loss_series)
ax1.set_title("Loss")
ax2.plot(train_acc_series, label="Training")
ax2.plot(val_acc_x, val_acc_series, marker='o', label="Validation")
ax2.legend(loc='lower right')
ax2.set_title("Accuracy")
plt.show()