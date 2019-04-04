import pickle
import numpy as np
import matplotlib.pyplot as plt
import csv
import cv2


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


def add_rect(im, w, h):
    min_hl = 3
    max_hl = 8
    x1 = int((w - max_hl) * np.random.uniform(0, 1))
    y1 = int((h - max_hl) * np.random.uniform(0, 1))
    x2 = x1 + int(np.random.uniform(min_hl, max_hl))
    y2 = y1 + int(np.random.uniform(min_hl, max_hl))
    cv2.rectangle(im, (x1,y1), (x2,y2), (0,0,0), -1)
    return im


training_file = "train.p"
with open(training_file, mode='rb') as f:
    train = pickle.load(f)
X_train, y_train = train['features'], train['labels']

# Open csv and place signnames in dictionary
reader = csv.reader(open('signnames.csv', mode='r'))
signs_dict = dict((rows[0], rows[1]) for rows in reader)

translation = 0.1
angle = 15

width, height = 32, 32
X_trans = np.copy(X_train)
X_rot = np.copy(X_train)
X_rect = np.copy(X_train)
for i in range(X_train.shape[0]):
    raw_image = X_train[i]
    # Translation
    image_trans = np.copy(X_train[i])
    X_trans[i] = translate(image_trans, translation, width, height)
    # Rotation
    image_rot = np.copy(X_train[i])
    X_rot[i] = rotate(image_rot, angle, width, height)
    # Add rectangles
    image_rect = np.copy(X_train[i])
    X_rect[i] = add_rect(image_rect, width, height)


# Create dict of new data, and write it to disk via pickle file
for new_file, new_X in zip(("train_trans.p", "train_rot.p", "train_rect.p"),
                           (X_trans, X_rot, X_rect)):
    print(new_file)
    with open(new_file, mode='wb') as f:
        pickle.dump({'features': new_X, 'labels': y_train}, f)

n = 3580
raw_image = X_train[n]
image_trans = X_trans[n]
image_rot = X_rot[n]
image_rect = X_rect[n]
# Visualizations will be shown in the notebook.
f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(10,3))
for ax, image, title in zip((ax1, ax2, ax3, ax4),
                            (raw_image, image_trans, image_rot, image_rect),
                            (signs_dict[str(y_train[n])], 'Translation', 'Rotation', 'Add rectangle')):
    ax.axis('off')
    ax.imshow(image)
    ax.set_title(title)
plt.show()
# f.savefig("writeup_images/data_augmentation.jpg")