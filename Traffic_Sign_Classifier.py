### Step 0: Load The Data
# Load pickled data
import pickle
import numpy as np

training_file = "train.p"
validation_file = "valid.p"
testing_file = "test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train_raw, y_train_raw = train['features'], train['labels']
X_valid_raw, y_valid = valid['features'], valid['labels']
X_test_raw, y_test = test['features'], test['labels']


# Add augmented training data
def add_data(new_data_file, X_train_old, y_train_old):
    training_new = new_data_file
    with open(training_new, mode='rb') as f:
        train_new = pickle.load(f)
    X_train_new, y_train_new = train_new['features'], train_new['labels']
    X_train_old = np.append(X_train_old, X_train_new, axis=0)
    y_train_old = np.append(y_train_old, y_train_new, axis=0)
    return X_train_old, y_train_old


X_train_aug, y_train_aug = add_data("train_rot.p", X_train_raw, y_train_raw)
X_train_aug, y_train_aug = add_data("train_rect.p", X_train_aug, y_train_aug)
X_train_aug, y_train_aug = add_data("train_trans.p", X_train_aug, y_train_aug)

# preprocess to gray
def prepro(x):
    x = x.astype('float32')
    x = (x - 128.) / 128.
    return x

X_train = prepro(X_train_aug)
y_train = y_train_aug
X_valid = prepro(X_valid_raw)
X_test = prepro(X_test_raw)

### Step 1: Dataset Summary & Exploration
### Replace each question mark with the appropriate value.
### Use python, pandas or numpy methods rather than hard coding the results

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
print("Labels: ", y_train.shape, y_train[0])
### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.

# Open csv and place signnames in dictionary
import csv
reader = csv.reader(open('signnames.csv', mode='r'))
signs_dict = dict((rows[0], rows[1]) for rows in reader)

import matplotlib.pyplot as plt
# Visualizations will be shown in the notebook.
# n = 19000 + 34799
# plt.imshow(X_train_aug[n])
# plt.axis('off')
# plt.title(signs_dict[str(y_train_aug[n])])

plt.figure(1, figsize=(18, 7))
for i in range(43):
    plt.subplot(5, 9, i + 1)  # sets the number of feature maps to show on each row and column
    plt.title(str(i) + ": " + signs_dict[str(i)], fontsize=8)
    plt.axis('off')
    idx = np.where(y_train_raw == i)[0][80]
    plt.imshow(X_train_raw[idx])
plt.tight_layout()
plt.show()

### Step 2: Design and Test a Model Architecture
### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include
### converting to grayscale, etc.
### Feel free to use as many code cells as needed.
# Normalize the data: subtract the mean image
# mean_image = np.mean(X_train, axis=0, dtype='uint8')
# X_train -= mean_image
# X_valid -= mean_image
# X_test -= mean_image

### Define your architecture here.
### Feel free to use as many code cells as needed.
import tensorflow as tf
from tensorflow.contrib.layers import flatten


def LeNet5(x):
    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_weights = tf.get_variable("Wconv1", shape=[5, 5, 3, 6])
    conv1_bias = tf.get_variable("bconv1", shape=[6])
    conv1_layer = tf.nn.conv2d(x, conv1_weights, [1, 1, 1, 1], 'VALID') + conv1_bias
        # Activation.
    conv1_layer = tf.nn.relu(conv1_layer)
        # Pooling. Input = 28x28x6. Output = 14x14x6.
    pool1_layer = tf.nn.max_pool(conv1_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    # Layer 2: Convolutional. Input = 14x14x6. Output = 10x10x16.
    conv2_weights = tf.get_variable("Wconv2", shape=[5, 5, 6, 16])
    conv2_bias = tf.get_variable("bconv2", shape=[16])
    conv2_layer = tf.nn.conv2d(pool1_layer, conv2_weights, [1, 1, 1, 1], 'VALID') + conv2_bias
        # Activation.
    conv2_layer = tf.nn.relu(conv2_layer)
        # Pooling. Input = 10x10x16. Output = 5x5x16.
    pool2_layer = tf.nn.max_pool(conv2_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    # Flatten. Input = 5x5x16. Output = 400.
    flat = flatten(pool2_layer)
    # Layer 3: Fully Connected. Input = 400. Output = 120
    fc1_weights = tf.get_variable("W1", shape=[400, 120])
    fc1_bias = tf.get_variable("b1", shape=[120])
    fc1_layer = tf.add(tf.matmul(flat, fc1_weights), fc1_bias)
        # Activation.
    fc1_layer = tf.nn.relu(fc1_layer)
    # Layer 4: Fully Connected. Input = 120. Output = 84
    fc2_weights = tf.get_variable("W2", shape=[120, 84])
    fc2_bias = tf.get_variable("b2", shape=[84])
    fc2_layer = tf.add(tf.matmul(fc1_layer, fc2_weights), fc2_bias)
        # Activation.
    fc2_layer = tf.nn.relu(fc2_layer)
    # Layer 5: Fully Connected. Input = 84. Output = 43
    out_weights = tf.get_variable("Wout", shape=[84, 43])
    out_bias = tf.get_variable("bout", shape=[43])
    logits = tf.add(tf.matmul(fc2_layer, out_weights), out_bias)
    return logits


def LeNet_down(x):
    c1_out = 32
    c2_out = 64
    fc1_in = 5 * 5 * c2_out
    fc1_out = 1024
    fc2_out = 344
    fc3_out = 43
    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
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
        # Pooling. Input = 28x28x6. Output = 14x14x6.
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


def my_model(x):
    # Layer 1: Convolutional. Input = 32x32x1. Output = 26x26x32
    conv1_weights = tf.get_variable("Wconv1", shape=[7, 7, 1, 32])
    conv1_bias = tf.get_variable("bconv1", shape=[32])
    conv1_layer = tf.nn.conv2d(x, conv1_weights, strides=[1, 1, 1, 1], padding='VALID') + conv1_bias
        # Activation
    conv1_layer = tf.nn.relu(conv1_layer)
        # Batch normalization
        # beta = tf.get_variable("beta", shape=[32])
        # gamma = tf.get_variable("gamma", shape=[32])
        # mean, variance = tf.nn.moments(h1, axes=[0, 1, 2])
        # bn = tf.nn.batch_normalization(h1, mean, variance, beta, gamma, 1e-8)
        # Pooling. Input = 26x26x32. Output = 13x13x32
    pool1_layer = tf.nn.max_pool(conv1_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    # Flatten.Input = 13x13x32.Output = 5408
    flat_dim = 5408
    flat = tf.reshape(pool1_layer, [-1, flat_dim])
    # Layer 2: Fully Connected. Input = 5408. Output = 1024
    fc1_weights = tf.get_variable("W1", shape=[flat_dim, 1024])
    fc1_bias = tf.get_variable("b1", shape=[1024])
    fc1_layer = tf.matmul(flat, fc1_weights) + fc1_bias
        # Activation
    fc1_layer = tf.nn.relu(fc1_layer)
    # Layer 3: Fully Connected. Input = 1024. Output = 43
    out_weights = tf.get_variable("W2", shape=[1024, 43])
    out_bias = tf.get_variable("b2", shape=[43])
    logits = tf.matmul(fc1_layer, out_weights) + out_bias
    return logits


### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected,
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 43)
keep_prob = tf.placeholder(tf.float32)

### training pipeline
lr = tf.placeholder(tf.float32, shape=[])
logits, act_layer = LeNet_down(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=lr)
training_operation = optimizer.minimize(loss_operation)

### Model evaluation
pred = tf.argmax(logits, 1)
label = tf.argmax(one_hot_y, 1)
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
from sklearn.utils import shuffle
EPOCHS = 20
BATCH_SIZE = 256
rate = 1e-3
rate_decay = 0.96

batch_x, batch_y = X_train[0:BATCH_SIZE], y_train[0:BATCH_SIZE]
iterations = 0
loss_series = []
train_acc_series = []
val_acc_series = []
acc_x = []
training = True
if training:
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        num_examples = len(X_train)
        print("Num examples: ", num_examples)
        print("Training...")
        print()
        train_accuracy = evaluate(batch_x, batch_y)
        train_acc_series.append(train_accuracy)
        validation_accuracy = evaluate(X_valid, y_valid)
        val_acc_series.append(validation_accuracy)
        acc_x.append(iterations)
        for i in range(EPOCHS):
            X_train, y_train = shuffle(X_train, y_train)
            for offset in range(0, num_examples, BATCH_SIZE):
                end = offset + BATCH_SIZE
                batch_x, batch_y = X_train[offset:end], y_train[offset:end]
                loss, _ = sess.run([loss_operation, training_operation],
                                   feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5, lr: rate})
                loss_series.append(loss)
                iterations += 1

            # corr = sess.run(correct_prediction, feed_dict={x: batch_x, y: batch_y})
            # train_accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
            # logit1 = sess.run(logits, feed_dict={x: X_train[0:1], y: y_train[0:1]})
            # one_hot1 = sess.run(one_hot_y, feed_dict={x: X_train[0:1], y: y_train[0:1]})
            # print("Logit1: ", logit1)
            # print("One hot1: ", one_hot1)
            # prediction = sess.run(pred, feed_dict={x: X_train[0:10], y: y_train[0:10]})
            # labels = sess.run(label, feed_dict={x: X_train[0:10], y: y_train[0:10]})
            # print("Pred: ", prediction)
            # print("Label: ", labels)

            train_accuracy = evaluate(batch_x, batch_y)
            train_acc_series.append(train_accuracy)
            validation_accuracy = evaluate(X_valid, y_valid)
            val_acc_series.append(validation_accuracy)
            acc_x.append(iterations)
            print("EPOCH {} ({} iterations)...".format(i + 1, iterations))
            print("Learning rate: {:.1e}".format(rate))
            print("Loss = {}".format(loss))
            print("Training Accuracy = {:.3f}".format(train_accuracy))
            print("Validation Accuracy = {:.3f}".format(validation_accuracy))
            print()
            rate *= rate_decay
        # Plot predictions on validation set after training
        # f, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3, figsize=(10, 6))
        # for ax in (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9):
        #     n = int(np.random.rand() * 4410)
        #     prediction = sess.run(pred, feed_dict={x: X_valid[n:n + 1], y: y_valid[n:n + 1]})
        #     ax.imshow(np.squeeze(X_valid[n]))
        #     ax.axis('off')
        #     ax.set_title(signs_dict[str(prediction[0])])
        # plt.show()
        saver.save(sess, './traffic_model')
        print("Model saved")

    f, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))
    ax1.plot(loss_series)
    ax1.set_title("Loss")
    ax2.plot(acc_x, train_acc_series, marker='o', label="Training")
    ax2.plot(acc_x, val_acc_series, marker='o', label="Validation")
    ax2.legend(loc='lower right')
    ax2.set_title("Accuracy")
    plt.show()

from PIL import Image
import os
image_files = ['example_signs/' + image_file for image_file in os.listdir('example_signs')]
images = []
for image_file in image_files:
    print(image_file)
    if os.path.isfile(image_file):
        image = Image.open(image_file)
        image = image.resize((32, 32), Image.ANTIALIAS)
        image = np.array(image, dtype="int32" )
        images.append(image)
images = np.array(images, dtype="int32" )
images_pre = prepro(images)
print(images.shape)


with tf.Session() as sess:
    saver.restore(sess, './traffic_model')
    prediction = sess.run(pred, feed_dict={x: images_pre, keep_prob: 1})
    logits = sess.run(logits, feed_dict={x: images_pre, keep_prob: 1})
    best_logits = sess.run(tf.nn.top_k(tf.nn.softmax(logits), k=5), feed_dict={x: images_pre, keep_prob: 1})
    # print(logits)
    # print(best_logits)
    # print(prediction)
# Plot predictions on validation set after training
f, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3, figsize=(10, 6))
for i, ax in enumerate((ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9)):
    ax.imshow(images[i])
    ax.axis('off')
    ax.set_title(str(prediction[i]) + ": " + signs_dict[str(prediction[i])])
plt.show()

for i in range(9):
    axes = ((ax1, ax2, ax3), (ax4, ax5, ax6))
    f, axes = plt.subplots(2, 3, figsize=(6, 4))
    for axs in axes:
        for ax in axs:
            ax.axis('off')
    axes[0][0].imshow(images[i])
    txt = ''
    for j in range(5):
        logit = str(best_logits[1][i][j])
        prob = "%.1f" % (best_logits[0][i][j] * 100)
        txt = txt + logit + ": " + signs_dict[logit] + ' (' + prob + '%)\n'
        if j < 3:
            idx = np.where(y_train_aug == best_logits[1][i][j])[0][20]
            print(idx, X_train[idx].shape)
            axes[1][j].imshow(X_train_aug[idx])
    f.text(0.4, 0.9, txt, fontsize=10, va='top', wrap = True, bbox={'facecolor':'None'})
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
