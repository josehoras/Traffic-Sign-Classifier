# Import all needed libraries
import tensorflow as tf
from tensorflow.contrib.layers import flatten
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import csv
import pickle
import numpy as np
from PIL import Image
import os

### Step 0: Load The Data
# Load pickled data
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


# Add augmented training data
def add_data(new_data_file, X_train_old, y_train_old):
    training_new = new_data_file
    with open(training_new, mode='rb') as f:
        train_new = pickle.load(f)
    X_train_new, y_train_new = train_new['features'], train_new['labels']
    X_train_old = np.append(X_train_old, X_train_new, axis=0)
    y_train_old = np.append(y_train_old, y_train_new, axis=0)
    return X_train_old, y_train_old


for filename in ("train_trans.p", "train_rot.p", "train_rect.p"):
    X_train, y_train = add_data(filename, X_train, y_train)

### Step 1: Dataset Summary & Exploration

# Number of training examples
n_train = X_train.shape[0]
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
reader = csv.reader(open('signnames.csv', mode='r'))
signs_dict = dict((rows[0], rows[1]) for rows in reader)
from textwrap import wrap
# Visualizations
n = 40 # number of classes to display
rows = 8
cols = 5
plt.figure(1, figsize=(cols*2, rows*2))
for i in range(n):
    plt.subplot(rows, cols, i + 1)  # sets the number of feature maps to show on each row and column
    plt.title(str(i) + ": " + signs_dict[str(i)], fontsize=12)
    plt.axis('off')
    idx = np.where(y_train == i)[0][50]
    plt.imshow(X_train[idx])
# plt.subplots_adjust(wspace=0.1, hspace=0)
plt.tight_layout()
plt.show()

### Step 2: Design and Test a Model Architecture
# center and normalize
def prepro(x):
    x = x.astype('float32')
    x = (x - 128.) / 128.
    return x

X_train_norm = prepro(X_train)
X_valid_norm = prepro(X_valid)
X_test_norm = prepro(X_test)

### Define your architecture here.
def sign_model(x):
    c1_out = 32
    c2_out = 64
    fc1_in = 5 * 5 * c2_out
    fc1_out = 1024
    fc2_out = 344
    fc3_out = 43
    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x32.
    conv1_weights = tf.get_variable("Wconv1", shape=[5, 5, 3, c1_out])
    conv1_bias = tf.get_variable("bconv1", shape=[c1_out])
    conv1_layer = tf.nn.conv2d(x, conv1_weights, [1, 1, 1, 1], 'VALID') + conv1_bias
        # Activation.
    conv1_layer = tf.nn.relu(conv1_layer)
        # Batch normalization
    beta = tf.get_variable("beta", shape=[c1_out])
    gamma = tf.get_variable("gamma", shape=[c1_out])
    mean, variance = tf.nn.moments(conv1_layer, axes=[0, 1, 2])
    conv1_layer = tf.nn.batch_normalization(conv1_layer, mean, variance, beta, gamma, 1e-8)
        # Pooling. Input = 28x28x6. Output = 14x14x64.
    pool1_layer = tf.nn.max_pool(conv1_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    # Layer 2: Convolutional. Input = 14x14x6. Output = 10x10x16.
    conv2_weights = tf.get_variable("Wconv2", shape=[5, 5, c1_out, c2_out])
    conv2_bias = tf.get_variable("bconv2", shape=[c2_out])
    conv2_layer = tf.nn.conv2d(pool1_layer, conv2_weights, [1, 1, 1, 1], 'VALID') + conv2_bias
        # Activation.
    conv2_layer = tf.nn.relu(conv2_layer)
        # Pooling. Input = 10x10x16. Output = 5x5x16.
    pool2_layer = tf.nn.max_pool(conv2_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    # Flatten. Input = 5x5x16. Output = 400.
    flat = flatten(pool2_layer)
    # Layer 3: Fully Connected. Input = 400. Output = 120
    fc1_weights = tf.get_variable("W1", shape=[fc1_in, fc1_out])
    fc1_bias = tf.get_variable("b1", shape=[fc1_out])
    fc1_layer = tf.add(tf.matmul(flat, fc1_weights), fc1_bias)
        # Activation.
    fc1_layer = tf.nn.relu(fc1_layer)
    # Layer 4: Fully Connected. Input = 120. Output = 84
    fc2_weights = tf.get_variable("W2", shape=[fc1_out, fc2_out])
    fc2_bias = tf.get_variable("b2", shape=[fc2_out])
    fc2_layer = tf.add(tf.matmul(fc1_layer, fc2_weights), fc2_bias)
        # Activation.
    fc2_layer = tf.nn.relu(fc2_layer)
        # Dropout
    fc2_layer = tf.nn.dropout(fc2_layer, keep_prob)
    # Layer 5: Fully Connected. Input = 84. Output = 43
    out_weights = tf.get_variable("Wout", shape=[fc2_out, fc3_out])
    out_bias = tf.get_variable("bout", shape=[fc3_out])
    logits = tf.add(tf.matmul(fc2_layer, out_weights), out_bias)
    return logits, conv1_layer


### Train your model here.
# Data placeholder
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 43)
keep_prob = tf.placeholder(tf.float32)

### training pipeline
lr = tf.placeholder(tf.float32, shape=[])
logits, act_layer = sign_model(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=lr)
training_operation = optimizer.minimize(loss_operation)

### Model evaluation
prediction = tf.argmax(logits, 1)
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset + BATCH_SIZE], y_data[offset:offset + BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


### Train the model
EPOCHS = 10
BATCH_SIZE = 256
rate = 1e-3
rate_decay = 0.96

iterations = 0
loss_series = []
train_acc_series = []
val_acc_series = []
acc_x = []
training = False
if training:
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        num_examples = len(X_train)
        print("Training...")
        print()
        batch_x, batch_y = X_train_norm[0:BATCH_SIZE], y_train[0:BATCH_SIZE]
        train_accuracy = evaluate(batch_x, batch_y)
        train_acc_series.append(train_accuracy)
        validation_accuracy = evaluate(X_valid, y_valid)
        val_acc_series.append(validation_accuracy)
        acc_x.append(iterations)

        for i in range(EPOCHS):
            X_train_norm, y_train = shuffle(X_train_norm, y_train)
            for offset in range(0, num_examples, BATCH_SIZE):
                end = offset + BATCH_SIZE
                batch_x, batch_y = X_train_norm[offset:end], y_train[offset:end]
                loss, _ = sess.run([loss_operation, training_operation],
                                   feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5, lr: rate})
                loss_series.append(loss)
                iterations += 1

            train_accuracy = evaluate(X_train_norm, y_train)
            train_acc_series.append(train_accuracy)
            validation_accuracy = evaluate(X_valid_norm, y_valid)
            val_acc_series.append(validation_accuracy)
            acc_x.append(iterations)
            print("EPOCH {} ({} iterations)...".format(i + 1, iterations))
            print("Learning rate: {:.1e}".format(rate))
            print("Loss = {}".format(loss))
            print("Training Accuracy = {:.3f}".format(train_accuracy))
            print("Validation Accuracy = {:.3f}".format(validation_accuracy))
            print()
            rate *= rate_decay
        saver.save(sess, './traffic_model')
        print("Model saved")

    # Plot loss and accuracy history
    f, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))
    ax1.plot(loss_series)
    ax1.set_title("Loss")
    ax2.plot(acc_x, train_acc_series, marker='o', label="Training")
    ax2.plot(acc_x, val_acc_series, marker='o', label="Validation")
    ax2.legend(loc='lower right')
    ax2.set_title("Accuracy")
    plt.show()

image_files = [image_file for image_file in os.listdir('example_signs')]
images = []
labels = []
for image_file in image_files:
    print(image_file)
    if os.path.isfile('example_signs/' + image_file):
        image = Image.open('example_signs/' + image_file)
        image = image.resize((32, 32), Image.ANTIALIAS)
        image = np.array(image, dtype="int32" )
        images.append(image)
        labels.append(image_file.split('.')[0].split('_')[1])
my_labels = np.array(labels, dtype="int32" )
my_images = np.array(images, dtype="int32" )
my_images_pre = prepro(my_images)


with tf.Session() as sess:
    saver.restore(sess, './traffic_model')
    predictions = sess.run(prediction, feed_dict={x: my_images_pre, keep_prob: 1})
    best_logits = sess.run(tf.nn.top_k(tf.nn.softmax(logits), k=5), feed_dict={x: my_images_pre, keep_prob: 1})
# Plot predictions on validation set after training
f, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3, figsize=(10, 6))
for i, ax in enumerate((ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9)):
    ax.imshow(my_images[i])
    ax.axis('off')
    ax.set_title(str(predictions[i]) + ": " + signs_dict[str(predictions[i])])
plt.show()

print("pred: ", predictions)
print("labels: ", my_labels)
num_correct = np.sum(predictions == my_labels)
print("num_correct: ", num_correct)
my_acc = int(num_correct * 100 / predictions.shape[0])
print("Accuracy: {}%".format(my_acc))


### Print out the top five softmax probabilities for the predictions on the German traffic sign images found on the web.
for i in range(9):
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 2))
    for ax in (ax1, ax2, ax3):
        ax.axis('off')
    ax1.imshow(my_images[i])
    txt = ''
    for j in range(5):
        logit = str(best_logits[1][i][j])
        prob = "%.1f" % (best_logits[0][i][j] * 100)
        txt = txt + logit + ": " + signs_dict[logit] + ' (' + prob + '%)'
        if j < 4: txt = txt + '\n'
    f.text(0.35, 0.87, txt, fontsize=16, va='top', bbox={'facecolor':'None'})
    plt.show()

### Visualize your network's feature maps here.
### Feel free to use as many code cells as needed.

# image_input: the test image being fed into the network to produce the feature maps
# tf_activation: should be a tf variable name used during your training procedure that represents the calculated state of a specific weight layer
# activation_min/max: can be used to view the activation contrast in more detail, by default matplot sets min and max to the actual min and max values of the output
# plt_num: used to plot out multiple different weight feature map sets on the same block, just extend the plt number for each new feature map entry

def outputFeatureMap(image_input, tf_activation, activation_min=-1, activation_max=-1 ,plt_num=1):
    # Here make sure to preprocess your image_input in a way your network expects
    # with size, normalization, ect if needed
    # image_input =
    # Note: x should be the same name as your network's tensorflow data placeholder variable
    # If you get an error tf_activation is not defined it may be having trouble accessing the variable from inside a function
    with tf.Session() as sess:
        saver.restore(sess, './traffic_model')
        activation = tf_activation.eval(session=sess, feed_dict={x: image_input, keep_prob: 1})
        featuremaps = activation.shape[3]
        plt.figure(plt_num, figsize=(14, 8))
        for featuremap in range(featuremaps):
            plt.subplot(4,8, featuremap+1) # sets the number of feature maps to show on each row and column
            plt.title('FeatureMap ' + str(featuremap)) # displays the feature map number
            if activation_min != -1 & activation_max != -1:
                plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin =activation_min, vmax=activation_max, cmap="gray")
            elif activation_max != -1:
                plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmax=activation_max, cmap="gray")
            elif activation_min !=-1:
                plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin=activation_min, cmap="gray")
            else:
                plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", cmap="gray")
        plt.show()

# outputFeatureMap(np.expand_dims(images[5], axis=0), act_layer, activation_min=-1, activation_max=-1 ,plt_num=1)
