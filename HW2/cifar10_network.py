import pickle
import numpy as np
import os
import urllib
import tarfile
import zipfile
import sys
import tensorflow as tf

import numpy as np
from time import time
import math

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

def model():

    _IMAGE_SIZE = 32
    _IMAGE_CHANNELS = 3
    _NUM_CLASSES = 10


    x = tf.placeholder(tf.float32, shape=[None, _IMAGE_SIZE * _IMAGE_SIZE * _IMAGE_CHANNELS], name='Input')
    y = tf.placeholder(tf.float32, shape=[None, _NUM_CLASSES], name='Output')
    x_image = tf.reshape(x, [-1, _IMAGE_SIZE, _IMAGE_SIZE, _IMAGE_CHANNELS], name='images')
    #keep_prob1 = tf.placeholder(tf.float32 , name ="koko1")
    keep_prob2 = tf.placeholder(tf.float32 , name ="koko2")

    w1 = weight_variable([3,3,3,32])
    b1 = bias_variable([32])
    conv1 = tf.nn.relu(conv2d(x_image, w1) + b1)
    w1_1 = weight_variable([3,3,32,64])
    b1_1 = bias_variable([64])
    conv = tf.nn.relu(conv2d(conv1, w1_1) + b1_1)
    pool = max_pool_2x2(conv)
    drop = tf.nn.dropout(pool, keep_prob2)


    w2 = weight_variable([3,3,64,128])
    b2 = bias_variable([128])
    conv2 = tf.nn.relu(conv2d(drop, w2) + b2)
    pool2 = max_pool_2x2(conv2)


    w3 = weight_variable([2,2,128,128])
    b3 = bias_variable([128])
    conv3 = tf.nn.relu(conv2d(pool2, w3) + b3)
    pool3 = max_pool_2x2(conv3)
    drop3 = tf.nn.dropout(pool3, keep_prob2)



    flat = tf.reshape(drop3, [-1, 4 * 4 * 128])

    fc = tf.nn.relu(tf.layers.dense(inputs=flat, units=1500)) #, activation=tf.nn.relu)
    drop4 = tf.nn.dropout(fc, keep_prob2)
    fc2 = tf.nn.relu(tf.layers.dense(inputs=drop4, units=1000))
    drop5 = tf.nn.dropout(fc2, keep_prob2)

    softmax = tf.nn.softmax(tf.layers.dense(inputs=drop5, units=_NUM_CLASSES))

    y_pred_cls = tf.argmax(softmax, axis=1)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=softmax, labels=y))

    optimizer = tf.train.ProximalGradientDescentOptimizer(1e-4  , beta1=0.9,
                                beta2=0.999,
                               epsilon=1e-08).minimize(loss)

    # PREDICTION AND ACCURACY CALCULATION
    correct_prediction = tf.equal(y_pred_cls, tf.argmax(y, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return    x, y, loss ,optimizer , correct_prediction ,accuracy, y_pred_cls   ,keep_prob2


def get_data_set(name="train"):
    x = None
    y = None

    maybe_download_and_extract()

    folder_name = "cifar_10"

    f = open('./data_set/'+folder_name+'/batches.meta', 'rb')
    f.close()

    if name is "train":
        for i in range(5):
            f = open('./data_set/'+folder_name+'/data_batch_' + str(i + 1), 'rb')
            datadict = pickle.load(f)
            f.close()

            _X = datadict["data"]
            _Y = datadict['labels']

            _X = np.array(_X, dtype=float) / 255.0
            _X = _X.reshape([-1, 3, 32, 32])
            _X = _X.transpose([0, 2, 3, 1])
            _X = _X.reshape(-1, 32*32*3)

            if x is None:
                x = _X
                y = _Y
            else:
                x = np.concatenate((x, _X), axis=0)
                y = np.concatenate((y, _Y), axis=0)

    elif name is "test":
        f = open('./data_set/'+folder_name+'/test_batch', 'rb')
        datadict = pickle.load(f)
        f.close()

        x = datadict["data"]
        y = np.array(datadict['labels'])

        x = np.array(x, dtype=float) / 255.0
        x = x.reshape([-1, 3, 32, 32])
        x = x.transpose([0, 2, 3, 1])
        x = x.reshape(-1, 32*32*3)

    return x, dense_to_one_hot(y)


def dense_to_one_hot(labels_dense, num_classes=10):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

    return labels_one_hot


def _print_download_progress(count, block_size, total_size):
    pct_complete = float(count * block_size) / total_size
    msg = "\r- Download progress: {0:.1%}".format(pct_complete)
    sys.stdout.write(msg)
    sys.stdout.flush()


def maybe_download_and_extract():
    main_directory = "./data_set/"
    cifar_10_directory = main_directory+"cifar_10/"
    if not os.path.exists(main_directory):
        os.makedirs(main_directory)

        url = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        filename = url.split('/')[-1]
        file_path = os.path.join(main_directory, filename)
        zip_cifar_10 = file_path
        file_path, _ = urllib.urlretrieve(url=url, filename=file_path, reporthook=_print_download_progress)

        print()
        print("Download finished. Extracting files.")
        if file_path.endswith(".zip"):
            zipfile.ZipFile(file=file_path, mode="r").extractall(main_directory)
        elif file_path.endswith((".tar.gz", ".tgz")):
            tarfile.open(name=file_path, mode="r:gz").extractall(main_directory)
        print("Done.")

        os.rename(main_directory+"./cifar-10-batches-py", cifar_10_directory)
        os.remove(zip_cifar_10)
def train(epoch):
    batch_size = int(math.ceil(len(train_x) / _BATCH_SIZE))
    for s in range(batch_size):
        batch_xs = train_x[s*_BATCH_SIZE: (s+1)*_BATCH_SIZE]
        batch_ys = train_y[s*_BATCH_SIZE: (s+1)*_BATCH_SIZE]

        start_time = time()
        _,batch_loss, batch_acc = sess.run(
            [ optimizer, loss, accuracy],
            feed_dict={x: batch_xs, y: batch_ys ,keep_prob2:0.5})
        duration = time() - start_time

        if s % 10 == 0:
            percentage = int(round((s/batch_size)*100))
            msg = "step: {} , batch_acc = {} , batch loss = {}"
            print(msg.format(s, batch_acc, batch_loss))

    test_and_save( epoch)

def test_and_save( epoch):
    global global_accuracy

    i = 0
    predicted_class = np.zeros(shape=len(test_x), dtype=np.int)
    while i < len(test_x):
        j = min(i + _BATCH_SIZE, len(test_x))
        batch_xs = test_x[i:j, :]
        batch_ys = test_y[i:j, :]
        predicted_class[i:j] = sess.run(
            y_pred_cls,
            feed_dict={x: batch_xs, y: batch_ys ,keep_prob2:1 }
        )
        i = j

    correct = (np.argmax(test_y, axis=1) == predicted_class)
    acc = correct.mean()*100
    correct_numbers = correct.sum()

    mes = "\nEpoch {} - accuracy: {:.2f}% ({}/{})"
    print(mes.format((epoch+1), acc, correct_numbers, len(test_x)))

    if global_accuracy != 0 and global_accuracy < acc:
        mes = "This epoch receive better accuracy: {:.2f} > {:.2f}. Saving session..."
        print(mes.format(acc, global_accuracy))
        global_accuracy = acc

    elif global_accuracy == 0:
        global_accuracy = acc

    print("###########################################################################################################")




sess = tf.Session()

train_x, train_y = get_data_set("train")
test_x, test_y = get_data_set("test")
x, y,loss, optimizer , correct_prediction ,accuracy, y_pred_cls ,keep_prob2= model()
global_accuracy = 0

# PARAMS
_BATCH_SIZE = 128
_EPOCH = 300

sess.run(tf.global_variables_initializer())

total_parameters = 0
for variable in tf.trainable_variables():
    shape = variable.get_shape()
    variable_parameters = 1
    for dim in shape:
        variable_parameters *= dim.value
    total_parameters += variable_parameters
print(total_parameters)
raw_input()
def main():
    for i in range(_EPOCH):
        print("\nEpoch: {0}/{1}\n".format((i+1), _EPOCH))
        train(i)


if __name__ == "__main__":
    main()
