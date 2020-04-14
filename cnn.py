"""
    CMPT 414 Project
    author: Yiming Zhang, Chongyu Yuan
"""

import tensorflow as tf
import data_loader
import matplotlib.pyplot as plt
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

batch_size = 100


class CNN:
    def __init__(self, lr, input_x_size, input_y_size):
        self.input_data = tf.placeholder(shape=[None, input_x_size, input_y_size], dtype=tf.float32)
        self.label = tf.placeholder(shape=[None, 34], dtype=tf.float32)

        # Define Parameters
        kenel_size = (3, 3)
        w1 = tf.get_variable("weight1", [kenel_size[0], kenel_size[1], 1, 32], initializer=tf.contrib.layers.xavier_initializer())
        b1 = tf.Variable(tf.constant(0.1, shape=[32]))
        w2 = tf.get_variable("weight2", [kenel_size[0], kenel_size[1], 32, 64], initializer=tf.contrib.layers.xavier_initializer())
        b2 = tf.Variable(tf.constant(0.1, shape=[64]))
        w3 = tf.get_variable("weight3", [kenel_size[0], kenel_size[1], 64, 8], initializer=tf.contrib.layers.xavier_initializer())
        b3 = tf.Variable(tf.constant(0.1, shape=[8]))
        w4 = tf.get_variable("weight4", [(input_y_size * input_x_size / 16) * 8, 512], initializer=tf.contrib.layers.xavier_initializer())
        b4 = tf.Variable(tf.constant(0.1, shape=[512]))
        w5 = tf.get_variable("weight5", [512, 34], initializer=tf.contrib.layers.xavier_initializer())
        b5 = tf.Variable(tf.constant(0.1, shape=[34]))

        # Connect Layers
        feature_1 = tf.nn.relu(tf.nn.conv2d(tf.reshape(self.input_data, [-1, input_x_size, input_y_size, 1]), w1, strides=[1, 1, 1, 1], padding='SAME') + b1)
        feature_2 = tf.nn.relu(tf.nn.conv2d(feature_1, w2, [1, 1, 1, 1], 'SAME') + b2)
        feature_2_pooling = tf.nn.max_pool2d(feature_2, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
        feature_3 = tf.nn.relu(tf.nn.conv2d(feature_2_pooling, w3, [1, 1, 1, 1], 'SAME') + b3)
        feature_3_pooling = tf.nn.max_pool2d(feature_3, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
        fully_connected = tf.nn.relu(tf.matmul(tf.reshape(feature_3_pooling, [-1, (int(input_y_size * input_x_size / 16) * 8)]), w4) + b4)

        # Dropout to prevent over-fitting
        output = tf.nn.dropout(fully_connected, 0.5)
        self.result = tf.matmul(output, w5) + b5

        # Loss Function
        self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=self.label, logits=self.result))

        self.optimize = tf.train.AdamOptimizer(lr).minimize(self.loss)
        self.accuracy_op = tf.metrics.accuracy(labels=tf.argmax(self.label, axis=1), predictions=tf.argmax(self.result, axis=1))[1]


def train_nn(data_loader):
    lr = 0.003
    acc_list = []
    loss_list = []
    episode_list = []
    cnn = CNN(lr, 20, 20)
    init = tf.global_variables_initializer()
    init2 = tf.local_variables_initializer()
    with tf.Session() as sess:
        # Initialization
        sess.run(init)
        sess.run(init2)
        total_epochs = 1000
        loss = 0
        print("Start training, total epochs: %d" % total_epochs)
        for i in range(1, total_epochs):
            if i % 20 == 0:
                img_list, label_list = data_loader.load_random_data(batch_size, True)
                accuracy = sess.run(cnn.accuracy_op, {cnn.input_data: img_list, cnn.label: label_list})
                acc_list.append(accuracy)
                print("Epoch %d: Accuracy = %.2f%%" % (i, accuracy * 100))
                loss_list.append(loss)
                episode_list.append(i)
            img_list, label_list = data_loader.load_random_data(batch_size)
            _, loss = sess.run([cnn.optimize, cnn.loss], {cnn.input_data: img_list, cnn.label: label_list})
        print("\nFinish training.")
        print("Model saved in ./license_plate_cnn folder.")
        tf.train.Saver().save(sess, "license_plate_cnn/model")
    # Figure 1: Accuracy
    plt.plot(episode_list, acc_list)
    plt.title("Accuracy of the CNN versus Training Epoch (learning rate=%.3f)" % lr)
    plt.xlabel('Epoch #')
    plt.ylabel('Accuracy')
    plt.show()
    # Figure 2: Loss Function
    plt.plot(episode_list, loss_list)
    plt.title("Loss Function of the CNN versus Training Epoch (learning rate=%.3f)" % lr)
    plt.xlabel('Epoch #')
    plt.ylabel('Loss function value')
    plt.show()


def test_nn():
    cnn = CNN(0.0025, 20, 20)
    with tf.Session() as sess:
        tf.train.get_checkpoint_state("./license_plate_cnn")
        tf.train.Saver().restore(sess, "license_plate_cnn/model")
        root = "dataset/test_images/"
        while True:
            img_name = int(input("We provide 34 new images for testing purpose. (They are under ./dataset/test_images/)\n\nPlease enter an integer from 0 to 32 to test an image: "))
            while img_name < 0 or img_name > 33:
                print("Invalid input!")
                img_name = int(input("We provide 34 new images for testing purpose. (They are under ./dataset/test_images/)\n\nPlease enter an integer from 0 to 32 to test an image: "))
            img = data_loader.convert_img_to_list(root + str(img_name) + ".jpg")
            img = [img]
            result = sess.run(tf.argmax(tf.nn.softmax(cnn.result), 1), {cnn.input_data: img})
            if result[0] > 9 and result[0] < 18:
                prediction = chr(result[0] + 55)
            elif result[0] >= 18 and result[0] < 23:
                prediction = chr(result[0] + 56)
            elif result >= 23:
                prediction = chr(result[0] + 57)
            else:
                prediction = str(result[0])
            img = data_loader.Image.open(root + str(img_name) + ".jpg")
            print("\n*************************")
            print("* Recognition result: %s *" % prediction)
            print("*************************\n")
            plt.imshow(img)
            plt.axis('off')
            plt.show()


if __name__ == '__main__':
    print("\n------------------------------")
    print("| License Plates Recognition |")
    print("------------------------------\n")
    print("1. Train Neural Network")
    print("2. Test Neural Network (use a pre-trained network)")
    choice = input("\nPlease enter 1 or 2: ")
    if choice == '1':
        print("\nPlease wait, loading training data...")
        data_loader = data_loader.Data_loader()
        train_nn(data_loader)
    elif choice == '2':
        test_nn()
    else:
        print("Invalid Input!")
