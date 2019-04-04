import pickle
import numpy as np
import matplotlib.pyplot as plt
import csv
import cv2


# Define transformations
def perspective(im, w, h):
    l = 5
    pts1 = np.float32([[int(l * np.random.uniform(-1, 1)), int(l * np.random.uniform(-1, 1))],
                       [h - int(l * np.random.uniform(-1, 1)), int(l * np.random.uniform(-1, 1))],
                       [int(l * np.random.uniform(-1, 1)), w-int(l * np.random.uniform(-1, 1))],
                       [h-int(l * np.random.uniform(-1, 1)), w-int(l * np.random.uniform(-1, 1))]])
    pts2 = np.float32([[0, 0], [h, 0], [0, w], [h, w]])

    M = cv2.getPerspectiveTransform(pts1,pts2)
    im = cv2.warpPerspective(im, M, (w,h))
    return im


def add_white_rect(im, w, h):
    min_hl = 3
    max_hl = 8
    x1 = int((w - max_hl) * np.random.uniform(0, 1))
    y1 = int((h - max_hl) * np.random.uniform(0, 1))
    x2 = x1 + int(np.random.uniform(min_hl, max_hl))
    y2 = y1 + int(np.random.uniform(min_hl, max_hl))
    cv2.rectangle(im, (x1,y1), (x2,y2), (255,255,255), -1)
    return im


training_file = "train.p"
with open(training_file, mode='rb') as f:
    train = pickle.load(f)
X_train, y_train = train['features'], train['labels']

# Open csv and place signnames in dictionary
reader = csv.reader(open('signnames.csv', mode='r'))
signs_dict = dict((rows[0], rows[1]) for rows in reader)

translation = 0.2
angle = 25

width, height = 32, 32
X_persp = np.copy(X_train)
X_white = np.copy(X_train)
for i in range(X_train.shape[0]):
    raw_image = X_train[i]
    # Change perspective
    image_persp = np.copy(X_train[i])
    X_persp[i] = perspective(image_persp, width, height)
    # Add white rectangles
    image_white = np.copy(X_train[i])
    X_white[i] = add_white_rect(image_white, width, height)


# Create dict of new data, and write it to disk via pickle file
# for new_file, new_X in zip(["train_persp.p", "train_white.p"],
#                            [X_persp, X_white]):
#     print(new_file)
#     with open(new_file, mode='wb') as f:
#         pickle.dump({'features': new_X, 'labels': y_train}, f)

n = 3580
raw_image = X_train[n]
image_persp = X_persp[n]
image_white = X_white[n]
# Visualizations will be shown in the notebook.
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10,2))
for ax, image, title in zip((ax1, ax2, ax3),
                            (raw_image, image_persp, image_white),
                            (signs_dict[str(y_train[n])], 'Perspective', "White rectangle")):
    ax.axis('off')
    ax.imshow(image)
    ax.set_title(title)
plt.show()
f.savefig("writeup_images/data_augmentation2.jpg")