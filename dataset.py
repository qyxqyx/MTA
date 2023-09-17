import numpy as np
import random
from tensorflow.python.platform import flags
import math
import cv2
import scipy.io as sio

FLAGS = flags.FLAGS


def random_crop(img, scale=(0.7, 1.0), ratio=(3. / 4., 4. / 3.)):
    shape = img.shape
    area = shape[0] * shape[1]
    for attempt in range(10):
        target_area = random.uniform(*scale) * area
        aspect_ratio = random.uniform(*ratio)

        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        if random.random() < 0.5:
            w, h = h, w

        if w <= shape[1] and h <= shape[0]:
            i = random.randint(0, shape[0] - h)
            j = random.randint(0, shape[1] - w)

            croped_img = img[i:i+h,j:j+w,:]
            croped_img = cv2.resize(croped_img, (32, 32))

            return croped_img

    w = min(shape[0], shape[1])
    i = (shape[0] - w) // 2
    j = (shape[1] - w) // 2
    croped_img = img[i:i + w, j:j + w, :]
    croped_img = cv2.resize(croped_img, (32, 32))

    return croped_img


def random_flip(img):
    if random.random() < 0.5:
        img = cv2.flip(img, 1)

    return img



class Dataset_CF(object):
    def __init__(self):
        self.num_classes = FLAGS.num_classes
        self.dim_output = self.num_classes

        root = 'dataset'

        data1 = sio.loadmat(root + '/cifar10/cifar-10-batches-mat/test_batch.mat')
        data2 = sio.loadmat(root + '/cifar10/cifar-10-batches-mat/data_batch_1.mat')
        data3 = sio.loadmat(root + '/cifar10/cifar-10-batches-mat/data_batch_2.mat')
        data4 = sio.loadmat(root + '/cifar10/cifar-10-batches-mat/data_batch_3.mat')
        data5 = sio.loadmat(root + '/cifar10/cifar-10-batches-mat/data_batch_4.mat')
        data6 = sio.loadmat(root + '/cifar10/cifar-10-batches-mat/data_batch_5.mat')

        X_test = data1['data']
        y_test = data1['labels']

        X_train = np.append(data2['data'], data3['data'], axis=0)
        X_train = np.append(X_train, data4['data'], axis=0)
        X_train = np.append(X_train, data5['data'], axis=0)
        X_train = np.append(X_train, data6['data'], axis=0)

        y_train = np.append(data2['labels'], data3['labels'], axis=0)
        y_train = np.append(y_train, data4['labels'], axis=0)
        y_train = np.append(y_train, data5['labels'], axis=0)
        y_train = np.append(y_train, data6['labels'], axis=0)

        X_train = X_train.reshape((50000, 3, 32, 32))
        X_test  = X_test.reshape((10000, 3, 32, 32))
        X_train = X_train.transpose((0, 2, 3, 1))
        X_test  = X_test.transpose((0, 2, 3, 1))

        self.train_images = X_train
        self.train_labels = np.array(y_train, dtype=np.int64)[:,0]
        self.train_labels = make_one_hot(self.train_labels, FLAGS.num_classes)
        self.val_images = X_test
        self.val_labels = np.array(y_test, dtype=np.int64)[:,0][:1000]
        self.val_labels = make_one_hot(self.val_labels, FLAGS.num_classes)

        self.train_indexes = list(range(self.train_labels.shape[0]))
        self.val_indexes = list(range(self.val_labels.shape[0]))

        random.shuffle(self.train_indexes)
        self.num_train = self.train_labels.shape[0]
        self.num_val = self.val_labels.shape[0]
        self.train_pointer = 0
        self.val_pointer = 0


    def get_batch_data(self, batch_size, train=True):
        val_end = False

        if train:
            if self.train_pointer + batch_size >= self.num_train:
                batch_indexes = self.train_indexes[self.train_pointer:]
                self.train_pointer = 0
                random.shuffle(self.train_indexes)
            else:
                batch_indexes = self.train_indexes[self.train_pointer:self.train_pointer + batch_size]
                self.train_pointer += batch_size

            batch_images = self.train_images[batch_indexes]
            batch_labels = self.train_labels[batch_indexes]

            for image_id in range(batch_images.shape[0]):
                image = batch_images[image_id]
                if train and FLAGS.data_aug:
                    if random.random() < 0.6:
                        image = random_crop(image)
                    image = random_flip(image)
                    batch_images[image_id] = image
        else:
            if self.val_pointer + batch_size >= self.num_val:
                batch_indexes = self.val_indexes[self.val_pointer:]
                self.val_pointer = 0
                val_end = True
            else:
                batch_indexes = self.val_indexes[self.val_pointer:self.val_pointer + batch_size]
                self.val_pointer += batch_size

            batch_images = self.val_images[batch_indexes]
            batch_labels = self.val_labels[batch_indexes]

        batch_images = batch_images.astype(np.float32)/255
        batch_labels = batch_labels.astype(np.int64)

        return batch_images, batch_labels, val_end


def make_one_hot(data, classes):
    return (np.arange(classes)==data[:,None]).astype(np.integer)


