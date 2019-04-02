import pickle
import numpy as np
import matplotlib.pyplot as plt
import csv
import cv2

training_file = "train.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)

X_train_raw, y_train = train['features'], train['labels']


# Define transformations
def translate(im, translation, w, h):
    x_offset = translation * w * np.random.uniform(-1, 1)
    y_offset = translation * h * np.random.uniform(-1, 1)
    translation_mat = np.array([[1, 0, x_offset], [0, 1, y_offset]])
    im = cv2.warpAffine(im, translation_mat, (w, h))
    return im


def rotate(im, angle, w, h):
    center = (w // 2, h // 2)
    angle_rand = np.random.uniform(-angle, angle)
    rotation_mat = cv2.getRotationMatrix2D(center, angle_rand, 1)
    im = cv2.warpAffine(im, rotation_mat, (w, h))
    return im


def add_rect(image, w, h):
    min_hl = 3
    max_hl = 8
    x1 = int((w - max_hl) * np.random.uniform(0, 1))
    y1 = int((h - max_hl) * np.random.uniform(0, 1))
    x2 = x1 + int(np.random.uniform(min_hl, max_hl))
    y2 = y1 + int(np.random.uniform(min_hl, max_hl))
    cv2.rectangle(image, (x1,y1), (x2,y2), (0,0,0), -1)
    return image

# Open csv and place signnames in dictionary
reader = csv.reader(open('signnames.csv', mode='r'))
signs_dict = dict((rows[0], rows[1]) for rows in reader)


translation = 0.2
angle = 25
# n = 19000
# raw_image = X_train_raw[n]
# h, w, channels = raw_image.shape
# image = np.copy(X_train_raw[n])
# image = add_rect(image, w, h)

width, height = 32, 32
new_X = np.copy(X_train_raw)
print(X_train_raw.shape)
print(new_X.shape)
for i in range(X_train_raw.shape[0]):
    raw_image = X_train_raw[i]
    image = np.copy(X_train_raw[i])
    # image = add_rect(image, width, height)
    image = rotate(image, angle, width, height)
    # image = translate(image, translation, width, height)
    new_X[i] = image

n = 34790
raw_image = X_train_raw[n]
image = new_X[n]
# Visualizations will be shown in the notebook.
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,8))
ax1.imshow(raw_image)
ax1.axis('off')
ax1.set_title(signs_dict[str(y_train[n])])
ax2.imshow(image)
ax2.axis('off')
ax2.set_title('translation')
plt.show()

# Create dict of new data, and write it to disk via pickle file
new_file = "train_rot.p"
new_data = {'features': new_X, 'labels': y_train}
with open(new_file, mode='wb') as f:
    pickle.dump(new_data, f)
