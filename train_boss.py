import numpy as np
import os
import tensorflow as tf
import cv2
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
IMAGE_SIZE = 64


def select_boss(boss_name, file_path):
    images = []
    labels = []
    for file_dir in os.listdir(file_path):
        abs_path = os.path.abspath(os.path.join(file_path, file_dir))
        if os.path.isdir(abs_path):
            if file_dir == boss_name:
                for file in os.listdir(abs_path):
                    image = cv2.imread(os.path.join(abs_path, file))
                    images.append(image)
                    labels.append("boss")
            else:
                for file in os.listdir(abs_path):
                    image = cv2.imread(os.path.join(abs_path, file))
                    images.append(image)
                    labels.append("others")
        else:
            pass
    labels = LabelEncoder().fit_transform(labels)[:, None]
    labels = OneHotEncoder().fit_transform(labels).todense() # boss -> (1, 0) others -> (0, 1)
    images = np.asarray(images)
    return images, labels


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def fc_layer(bottom, shape, size):
    W = weight_variable(shape)
    biases = bias_variable(size)
    return tf.matmul(bottom, W) + biases


def conv_layer(bottom, shape, filter, padding='SAME'):
    W = weight_variable(shape)
    biases = bias_variable(filter)
    conv = conv2d(bottom, W, padding) + biases
    return tf.nn.relu(conv)


def conv2d(bottom, weight, padding):
    return tf.nn.conv2d(bottom, weight, strides=[1, 1, 1, 1], padding=padding)


def max_pool(bottom):
    return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def build_model(input, drop=1, n_classes = 2):
    conv1_1 = conv_layer(input, [3, 3, 3, 32], [32])
    conv1_2 = conv_layer(conv1_1, [3, 3, 32, 32], [32])
    pool1 = max_pool(conv1_2)
    pool_drop1 = tf.nn.dropout(pool1, keep_prob=drop)

    conv2_1 = conv_layer(pool_drop1, [3, 3, 32, 64], [64])
    conv2_2 = conv_layer(conv2_1, [3, 3, 64, 64], [64])
    pool2 = max_pool(conv2_2)
    pool2_drop2 = tf.nn.dropout(pool2, keep_prob=drop)

    flatten = tf.reshape(pool2_drop2, [-1, 16*16*64])
    fc1 = fc_layer(flatten, [16*16*64, 512], [512])
    fc1_drop = tf.nn.dropout(tf.nn.relu(fc1), drop)
    fc2 = fc_layer(fc1_drop, [512, n_classes], [n_classes])
    return tf.nn.softmax(fc2)


def get_batch_train(x, y, i, train_size):
    return x[i * train_size: (i + 1) * train_size], y[i * train_size: (i + 1) * train_size]


def train_process(data, labels, n_epoch, batch_size=64, data_augmentation=True):
    data = data.astype('float32')
    data /= 255
    print(data.shape, labels.shape)
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=0)
    total_train_iteration = X_train.shape[0] / batch_size
    print(X_train.shape)

    with tf.device("/gpu:7"):
        x = tf.placeholder(tf.float32, [None, 64, 64, 3])
        y = tf.placeholder(tf.float32, [None, 2])
        drop = tf.placeholder(tf.float32)
        pre = build_model(x, drop=drop)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pre, labels=y))
        acc = 100 * tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(pre, 1), tf.argmax(y, 1))))
        optimizer = tf.train.AdamOptimizer(0.0001)
        train_step = optimizer.minimize(loss)

    config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    print(str(int(total_train_iteration)))

    for i in range(n_epoch):
        for train_batch_step in range(int(total_train_iteration)):
            print("step: " + str(train_batch_step))
            batch_x, batch_y = get_batch_train(X_train, y_train, train_batch_step, batch_size)
            sess.run(train_step, feed_dict={x: batch_x, y: batch_y, drop: 0.75})
        accuracy = sess.run(acc, feed_dict={x: X_test, y: y_test, drop: 1.0})
        print("epoch: " + str(i) + ", accuracy: " + str(accuracy))
        saver = tf.train.Saver()
        save_path = saver.save(sess, "model/", global_step=i)
    sess.close()
    return


if __name__ == '__main__':
    boss_name = "zonghua"
    img_path = "img/process"
    images, labels = select_boss(boss_name, img_path)
    train_process(images, labels, 10)
