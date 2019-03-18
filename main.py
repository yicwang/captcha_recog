import random

import numpy as np
from PIL import Image
from captcha.image import ImageCaptcha
import tensorflow as tf

CHARSETS = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

class CaptchaGenerator(object):

    def __init__(self, image_length):
        self.image_length = image_length
        self.image_captcha = ImageCaptcha()

    def get_random_text(self):
        retVal = ''
        for _ in range(self.image_length):
            retVal += CHARSETS[random.randint(0, len(CHARSETS)-1)]
        return retVal

    def get_labeled_image(self, samples):
        X, Y = [], []
        for _ in range(samples):
            text = self.get_random_text()
            image = self.image_captcha.generate(text, format='png')
            captcha_image = Image.open(image)
            array = np.array(captcha_image)
            X.append(array)
            Y.append(text)
            # self.image_captcha.write(text, "%s.png" % text)
        return X, Y


class CaptchaRecog(object):
    def __init__(self, captcha_length):
        self.captcha_length = captcha_length
        self.image_height = 60
        self.image_width = 160

        self.captcha_generator = CaptchaGenerator(captcha_length)

    def load_dataset(self, trainset_size, testset_size):
        # Both trainset and dataset will be coming from the same distribution
        self.X_train, self.Y_train = \
            self.captcha_generator.get_labeled_image(trainset_size)
        self.X_test, self.Y_test = \
            self.captcha_generator.get_labeled_image(testset_size)

        # image_arr = self.grey_image(image_arr)
        for i in range(trainset_size):
            self.X_train[i] = self.X_train[i] / 255.
            self.Y_train[i] = self.onehot_encode(self.Y_train[i])
        for i in range(testset_size):
            self.X_test[i] = self.X_test[i] / 255.
            self.Y_test[i] = self.onehot_encode(self.Y_test[i])

        print("Train set size: %d" % len(self.X_train))
        print("Test set size: %d" % len(self.X_test))

    def read_image(self, img_file):
        label = img_file.split('.')[0]
        image = Image.open(img_file)
        array = np.array(image)

        return label, array

    def grey_image(self, img):
        r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        grey = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return grey

    def onehot_encode(self, text):
        vector = np.zeros(self.captcha_length * len(CHARSETS))
        for idx, char in enumerate(text):
            vector[idx * len(CHARSETS) + CHARSETS.index(char)] = 1
        return vector

    def onehot_decode(self, vector):
        v = np.nonzero(vector)[0]
        text = ''
        for i in range(self.captcha_length):
            text += CHARSETS[v[i] % len(CHARSETS)]
        return text

    def get_minibatch(self, batch_number, batch_size):
        minibatch_x = np.zeros([batch_size, self.image_height, self.image_width, 3])
        minibatch_y = np.zeros([batch_size, self.captcha_length * len(CHARSETS)])
        for i in range(batch_size):
            minibatch_x[i, :] = self.X_train[batch_number*batch_size + i]
            minibatch_y[i, :] = self.Y_train[batch_number*batch_size + i]

        return minibatch_x, minibatch_y

    def model(self, rate=0.25):
        self.X = tf.placeholder(tf.float32, [None, self.image_height, self.image_width, 3])
        self.Y = tf.placeholder(tf.float32, [None, self.captcha_length * len(CHARSETS)])
        W1 = tf.get_variable("W1", [3, 3, 3, 32], initializer=tf.initializers.glorot_normal())
        W2 = tf.get_variable("W2", [3, 3, 32, 64], initializer=tf.initializers.glorot_normal())
        W3 = tf.get_variable("W3", [3, 3, 64, 128], initializer=tf.initializers.glorot_normal())

        # Conv1
        Z1 = tf.nn.conv2d(self.X, W1, strides=[1, 1, 1, 1], padding='SAME')
        A1 = tf.nn.relu(Z1)
        P1 = tf.nn.max_pool(A1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv1 = tf.nn.dropout(P1, rate=rate)

        # Conv2
        Z2 = tf.nn.conv2d(conv1, W2, strides=[1, 1, 1, 1], padding='SAME')
        A2 = tf.nn.relu(Z2)
        P2 = tf.nn.max_pool(A2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv2 = tf.nn.dropout(P2, rate=rate)

        # Conv3
        Z3 = tf.nn.conv2d(conv2, W3, strides=[1, 1, 1, 1], padding='SAME')
        A3 = tf.nn.relu(Z3)
        P3 = tf.nn.max_pool(A3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv3 = tf.nn.dropout(P3, rate=rate)

        # FC
        P = tf.layers.flatten(conv3)
        Z4 = tf.contrib.layers.fully_connected(P, 1024)
        Z4 = tf.contrib.layers.fully_connected(Z4, self.captcha_length * len(CHARSETS), activation_fn=None)

        return Z4

    def train(self, learning_rate=0.001, num_epochs=100, minibatch_size=64):
        Z4 = self.model()
        cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Z4, labels=self.Y))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
        init = tf.global_variables_initializer()

        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(init)
            for epoch in range(num_epochs):
                minibatch_cost = 0.
                for idx in range(int(len(self.X_train) / minibatch_size)):
                    minibatch_x, minibatch_y = self.get_minibatch(idx, minibatch_size)
                    _, tmp_cost = sess.run(
                         [optimizer, cost], feed_dict={self.X: minibatch_x, self.Y: minibatch_y}
                    )
                    minibatch_cost += tmp_cost

                # Print the cost every epoch
                print("Cost after epoch %i: %f" % (epoch, minibatch_cost))

            predict_text = tf.reshape(Z4, [-1, self.captcha_length, len(CHARSETS)])
            predict_text = tf.argmax(predict_text, 2)
            labelled_text = tf.reshape(self.Y, [-1, self.captcha_length, len(CHARSETS)])
            labelled_text = tf.argmax(labelled_text, 2)
            correct_prediction = tf.equal(predict_text, labelled_text)
            # character level accuracy rate
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float32"))
            train_accuracy = accuracy.eval({self.X: self.X_train, self.Y: self.Y_train})
            test_accuracy = accuracy.eval({self.X: self.X_test, self.Y: self.Y_test})
            print("Train Accuracy: %.2f" % train_accuracy)
            print("Test Accuracy: %.2f" % test_accuracy)

            saver.save(sess, "/tmp/my_model")

            return train_accuracy, test_accuracy

    def predict(self):
        tf.reset_default_graph()
        Z4 = self.model(rate=0)
        saver = tf.train.Saver()

        while (True):
            filename = 'filename.png'
            captcha_image = Image.open(filename)
            X = np.array(captcha_image)

            with tf.Session() as sess:
                saver.restore(sess, "/tmp/my_model")
                y_predict = tf.reshape(Z4, [-1, self.captcha_length, len(CHARSETS)])
                y_predict = tf.argmax(y_predict, 2)

                yhat = sess.run(y_predict, feed_dict={self.X: [X]})[0].tolist()
                predict_text = np.zeros(self.captcha_length * len(CHARSETS))
                for idx, ch in enumerate(yhat):
                    predict_text[idx * len(CHARSETS) + ch] = 1
                print("%s is predicted as: %s." % (filename, self.onehot_decode(predict_text)))
                break


def main():
    cr = CaptchaRecog(captcha_length=4)
    cr.load_dataset(trainset_size=16384, testset_size=2048)
    cr.train(num_epochs=100)
    cr.predict()

if __name__ == '__main__':
    main()
