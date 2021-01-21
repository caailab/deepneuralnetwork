#-*- coding: utf-8 -*-
_author_ = 'Licheng QU'

import os
import timeit

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()

import configuration as cf
import trafficdata as tfdata

# Leaky ReLU
def leakyrelu(x, alpha=0.1, max_value=None, name=None):
    """
    ReLU.

    alpha: slope of negative section.
    """

    return tf.maximum(alpha * x, x, name=name)


# Create the model
def create_model(num_features, num_labels, num_hidden, keepneuron=0.85):
    tf.set_random_seed(2017)

    x = tf.placeholder(tf.float32, [None, num_features], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, num_labels], name='y-input')

    W = []
    b = []
    h = []

    layer_size = len(num_hidden)
    if layer_size > 0 :
        num_hidden0 = num_hidden[0]
    else:
        num_hidden0 = num_labels

    W.append(tf.Variable(tf.random_uniform([num_features, num_hidden0], minval=-1.0, maxval=1.0, name='W0')))
    b.append(tf.Variable(tf.zeros([num_hidden0]), name='b0'))
    h.append(tf.tanh(tf.matmul(x, W[0]) + b[0], name='input_layer'))
    # h[0] = tf.nn.dropout(h0, keepneuron)

    for i in range(1, layer_size):
        W.append(tf.Variable(tf.random_uniform([num_hidden[i-1], num_hidden[i]], minval=-1.0, maxval=1.0, name='W'+str(i))))
        b.append(tf.Variable(tf.zeros([num_hidden[i]]), name='b'+str(i)))
        h.append(tf.tanh(tf.matmul(h[i-1], W[i]) + b[i], name='hidden_layer'+str(i)))
        # h[i] = tf.nn.dropout(h[i], keepneuron)

    W.append(tf.Variable(tf.random_uniform([num_hidden[-1], num_labels], minval=-1.0, maxval=1.0, name='W' + str(layer_size))))
    b.append(tf.Variable(tf.zeros([num_labels]), name='b' + str(layer_size)))
    h.append(tf.nn.relu(tf.matmul(h[-1], W[layer_size]) + b[layer_size], name='output_layer'))
    # h[layer_size] = tf.nn.dropout(h[layer_size], keepneuron)

    y = h[-1]

    return x, y_, y


def train_model(train_x, train_y, num_features, num_labels, num_hidden, model_file, epochs=1000, batchsize=1000, learningrate=0.01):
    start = timeit.default_timer()

    x, y_, y = create_model(num_features, num_labels, num_hidden)

    # Define loss and optimizer
    loss = tf.reduce_mean(tf.square(y - y_))
    optimizer = tf.train.AdadeltaOptimizer(learningrate) #.GradientDescentOptimizer(learningrate)
    train_op = optimizer.minimize(loss)

    tf.add_to_collection("x", x)
    tf.add_to_collection("y", y)
    tf.add_to_collection("y_", y_)

    saver = tf.train.Saver()

    #tf.summary.scalar('x', x)
    #tf.summary.scalar('y', y)
    #tf.summary.scalar('output', y_)
    #tf.summary.scalar('loss', loss)
    #summary_merge = tf.summary.merge_all()

    with tf.Session() as sess:
        #train_writer = tf.summary.FileWriter(tensor_board_log_dir + '/train', sess.graph)

        # Initialize all the tensor
        tf.global_variables_initializer().run()

        if tf.gfile.Exists(model_file + '.meta'):
            print("Restore Model from " + model_file)
            #tf.reset_default_graph()
            saver.restore(sess, model_file)
            print("Restore Model ... Done")

        loss_epoch = []
        loss_batch = []
        train_data_row = len(train_x) - 1
        for e in range(epochs):
            loss_batch.clear()
            batch_begin = 0
            while (batch_begin < train_data_row):
                # Dynamic batch size (+-10%)
                #batchsize_10 = batchsize // 10
                #batchsize_dynamic = batchsize + random.randint(-batchsize_10, batchsize_10)
                batch_end = batch_begin + batchsize #_dynamic
                # print("Batch begin {}\tend {}".format(batch_begin, batch_end))
                if (batch_end > train_data_row):
                    #print("Batch begin {}\tend {}\tAdjust {}".format(batch_begin, batch_end, batch_end - train_data_size))
                    batch_end = train_data_row
                #print("batch: {}-{}\t train_data_row: {}\t shape: {}".format(batch_begin, batch_end, train_data_row, train_x.shape))
                batch_xs, batch_ys = train_x[batch_begin:batch_end],\
                                     train_y[batch_begin:batch_end]
                #batch_xs = batch_xs.reshape(-1, num_features)
                #batch_ys = batch_ys.reshape(-1, num_labels)
                # print(batch_xs.shape)
                # print(batch_ys.shape)

                #print("{}:\tTraining data set from {} to {}".format(e, batch_begin, batch_end))
                _, error = sess.run([train_op, loss], feed_dict={x: batch_xs, y_: batch_ys})
                loss_batch.append(error)

                # Next batch
                batch_begin = batch_end

            loss_epoch.append(np.array(loss_batch).mean())

            if e % 100 == 0:
                print("Epoch {} :\tLast Batch Loss={}\tEpoch Mean Loss={}".format(e, error, loss_epoch[-1]))
                #summary = sess.run(summary_merge, feed_dict={x: batch_xs, y_: batch_ys})
                #train_writer.add_summary(summary, e)
                #saver.save(sess, tensor_board_log_dir + '/model.ckpt', e)

                if len(loss_epoch) > 1 and np.abs(loss_epoch[-1] - loss_epoch[-100]) < 0.00000001:
                    print("EARLY STOP TRAINING!")
                    break

            if e % 1000 == 0:
                saver.save(sess, model_file)

        end = timeit.default_timer()
        print("Training Time: {}s".format(end - start))

        print("Save Model to " + model_file)
        saver.save(sess, model_file)
        print("Save Model ... Done")

        #train_writer.close()

    return loss_epoch


def evaluate_model(test_x, test_y, num_features, num_labels, num_hidden, model_file):
    start = timeit.default_timer()

    x, y_, y = create_model(num_features, num_labels, num_hidden)

    error = tf.abs(y - y_)
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.003)
    #train_op = optimizer.minimize(loss)

    saver = tf.train.Saver()    #tf.global_variables())
    with tf.Session() as sess:
        # Initialize all the tensor
        if tf.gfile.Exists(model_file + '.meta'):
            print("Restore Model from " + model_file)
            #tf.reset_default_graph()
            saver.restore(sess, model_file)
            print("Restore Model ... Done")
        #else:
        #    init = tf.global_variables_initializer()
        #    sess.run(init)

        test_data_size = len(test_x)
        print("\ntest_data_size=", test_data_size)

        batch_xs = test_x.reshape(-1, num_features)
        batch_ys = test_y.reshape(-1, num_labels)

        pred_y, pred_bias = sess.run([y, error], feed_dict={x: batch_xs, y_: batch_ys})

        end = timeit.default_timer()
        print("Evaluate Time: {}s".format(end - start))

        pred_bias_percent = np.abs(pred_bias / batch_ys) * 100

        # print('\nModel Parameter: W1={}, b1={}, W2={}, b2={}'.format(W1.value(), b1.value(), W2.value(), b2.value()))
        print('Absolute Error: mean={}, min={}, max={}, var={}'.format(np.mean(pred_bias),
                                                                  np.min(pred_bias),
                                                                  np.max(pred_bias),
                                                                  np.var(pred_bias)))
        print('Absolute Percentage Error: mean={}, min={}, max={}, var={}'.format(np.mean(pred_bias_percent),
                                                                           np.min(pred_bias_percent),
                                                                           np.max(pred_bias_percent),
                                                                           np.var(pred_bias_percent)))
        print('Test RMSE = {}'.format(np.mean(np.std(pred_bias))))
        print('Test MAPE = {}'.format(np.mean(pred_bias_percent)))
        print('\n')

        return pred_y[:, 0], pred_bias[:, 0], pred_bias_percent[:, 0]


def main_dnn(_):
    #tensor_board_log_dir = '/tmp/tensorflow/traffic_dnn/logs'
    # tensorboard --logdir=/tmp/tensorflow/traffic_dnn/logs

    #if tf.gfile.Exists(tensor_board_log_dir):
    #    tf.gfile.DeleteRecursively(tensor_board_log_dir)
    #tf.gfile.MakeDirs(tensor_board_log_dir)

    config_file = '../conf-005es18066/traffic_dnn-05min.conf'
    config_file = '../conf-005es18066/traffic_dnn-05min-b1152.conf'
    #config_file = '../conf-005es18066/traffic_dnn-minute10-b288.conf'
    #config_file = '../conf-005es18066/traffic_dnn-minute15-b288.conf'
    #config_file = '../conf-005es18066/traffic_dnn-minute20-b288.conf'
    #config_file = '../conf-005es18066/traffic_dnn-minute30-b288.conf'
    #config_file = '../conf-005es18066/traffic_dnn-minute60-b288.conf'

    #config_file = '../conf/traffic_dnn-05min.conf'
    #config_file = '../conf/traffic_dnn-minute10.conf'
    #config_file = '../conf/traffic_dnn-minute15.conf'
    #config_file = '../conf/traffic_dnn-minute20.conf'
    #config_file = '../conf/traffic_dnn-minute30.conf'
    #config_file = '../conf/traffic_dnn-minute60.conf'

    #hidden_layer = [16, 18, 11, 21, 12, 6]      # minute 9.97%
    #hidden_layer = [60, 60, 60, 60, 60, 60]     # minute 9.76%
    #hidden_layer = [20, 20, 20, 20, 20, 20]     # 9.82%
    hidden_layer = [15, 18, 22, 9, 5]   	     # 7.80% 5min

    conf = cf.Config(config_file)
    train_with_config(conf, hidden_layer, True)


def train_with_config(config, hidden_layer, showwindow):
    # model saving path
    model_name_prefix = config.model_name
    model_path = config.model_path
    model_image_path = model_path + 'images/'
    if not os.path.isdir(model_image_path):
        os.makedirs(model_image_path)

    model_result_path = model_path + 'result/'
    if not os.path.isdir(model_result_path):
        os.makedirs(model_result_path)

    num_features = config.num_features    # 9
    num_labels = config.num_labels        # 1
    label_factor = config.label_factor    # 20000
    interval = config.interval            # 5

    epochs = config.epochs                # 10000
    batchsize = config.batchsize          # 24 * 5
    learningrate = config.learningrate    # 0.1

    # Read Training data
    train_data_file = config.train_data_file  # '../dataset/hour_rain/2014/volume-005es16513-N-2014.csv'
    test_data_file = config.test_data_file    # '../dataset/hour_rain/2015/volume-005es16513-N-201502.csv'

    #train_data_file = '../dataset/005es18066/18066-I-201603.csv'
    #test_data_file = '../dataset/005es18066/18066-I-201603.csv'

    #xtrain, ytrain = tfdata.load_traffic_data(train_data_file)
    #xtest, ytest = tfdata.load_traffic_data(test_data_file)
    xtrain, ytrain, train_time = tfdata.load_traffic_data_resample(train_data_file, interval)
    xtest, ytest, test_time = tfdata.load_traffic_data_resample(test_data_file, interval)

    train_x, train_y = tfdata.normalize_data(xtrain, ytrain, label_factor)
    test_x, test_y = tfdata.normalize_data(xtest, ytest, label_factor)

    train_x[:, 8] = 0.0     # rain
    test_x[:, 8] = 0.0      # rain

    train_x = train_x.reshape(-1, num_features)
    train_y = train_y.reshape(-1, num_labels)
    test_x = test_x.reshape(-1, num_features)
    test_y = test_y.reshape(-1, num_labels)

    print('Train dataset:\tFeatures Shape={}\tLabels Shape={}\tMAX: {}, MIN: {}'.format(train_x.shape, train_y.shape, train_y.max(), train_y.min()))
    #print(train_x, train_y)
    print('Test dataset:\tFeatures Shape={}\tLabels Shape={}\tMAX: {}, MIN: {}'.format(test_x.shape, test_y.shape, test_y.max(), test_y.min()))
    #print(test_x, test_y)

    model_name = model_name_prefix + '-{}'.format(''.join(str(h) for h in hidden_layer))
    traffic_prediction_model_file = model_path + model_name + '/traffic_prediction.model'

    # Let's go
    tf.reset_default_graph()

    # Training
    loss_epoch = train_model(train_x, train_y, num_features, num_labels, hidden_layer, traffic_prediction_model_file, epochs, batchsize, learningrate)

    tf.reset_default_graph()

    # Evaluate the model
    pred_y, _, _ = evaluate_model(test_x, test_y, num_features, num_labels, hidden_layer, traffic_prediction_model_file)

    # print(test_y.shape, pred_y.shape, pred_bias.shape, pred_bias_percent.shape)
    # for i in range(len(test_y)):
    #    print("{},{},{},{}".format(test_y[i], pred_y[i], pred_bias[i], pred_bias_percent[i]))

    # train_writer.close()
    # test_writer.close()
    test_y = test_y.flatten()
    test_y = tfdata.unnormalize_data(test_y, label_factor)
    pred_y = tfdata.unnormalize_data(pred_y, label_factor)
    pred_bias = np.abs(pred_y - test_y)
    pred_bias_percent = np.abs(pred_bias / test_y) * 100

    write_result(test_time, test_y, pred_y, pred_bias, pred_bias_percent, loss_epoch, model_name, model_result_path)

    visualize_model(test_time, test_y, pred_y, pred_bias, pred_bias_percent, loss_epoch, model_name, model_image_path, showwindow)

    return np.mean(pred_bias), np.mean(pred_bias_percent)


def write_result(test_time, test_y, pred_y, pred_bias, pred_bias_percent, loss_epoch, model_name, path = './'):
    result_file = path + 'result.csv'
    f = open(result_file, 'a')
    f.write('{},{},{},{}\n'.format(model_name, np.mean(pred_bias), np.mean(pred_bias_percent), loss_epoch[-1]))
    f.close()

    time = pd.to_datetime(test_time.flatten())

    rf = open(path + model_name + '-result.csv', 'w')
    for i in range(len(test_y)):
        rf.write("{},{},{},{},{}\n".format(time[i], test_y[i], pred_y[i], pred_bias[i], pred_bias_percent[i]))
    rf.close()

    lf = open(path + model_name + '-loss.csv', 'w')
    for i in range(len(loss_epoch)):
        lf.write("{}\n".format(loss_epoch[i]))

    lf.close()


def visualize_model(test_time, test_y, pred_y, pred_bias, pred_bias_percent, loss_epoch, title, path='./', showwindow=False):
    plt.clf()
    plt.figure(num=1, figsize=(8, 6))
    plt.subplots_adjust(hspace=0.4)

    plt.rc('font', family='Times New Roman')

    ax1 = plt.subplot(211)
    plt.sca(ax1)

    plt.title(title + "\nMAE={:.3f}  MAPE {:.3f}%  RMSE={:.3f}".format(np.mean(pred_bias), np.mean(pred_bias_percent), np.mean(np.std(pred_bias))), size=12)
    #plt.text(5, 0.65, 'MAE={}'.format(np.mean(pred_bias)))
    #plt.text(5, 0.60, 'MAPE={}'.format(np.mean(pred_bias_percent)))

    plt.xlabel('Time serials', size=12)
    plt.ylabel('Volume (vehs/5min)', size=12)
    plt.plot(test_y, color='b', linestyle='-', label='actual data')  # marker='o',
    plt.plot(pred_y, color='r', linestyle='-', label='predicted')  # marker='*',
    # plt.plot(pred_bias, color='y', linestyle='-', label='bias')  # marker='+',
    # plt.plot(pred_bias_percent, color='g', linestyle='-', marker='.', label='bias (%)')

    plt.legend(loc='upper right')

    # Plot the loss_epoch
    ax2 = plt.subplot(212)
    plt.sca(ax2)

    plt.title('Last Loss {:.8f}'.format(loss_epoch[-1]), size=12)

    plt.xlabel('Epochs', size=12)
    plt.ylabel('Loss', size=12)
    plt.plot(loss_epoch, color='m', linestyle='-', label='loss')  # marker='o',

    plt.savefig(path + title + '.png', format='png')

    if showwindow:
        plt.show()

    plt.close()


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
    #                    help='Directory for storing input data')
    # FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main_dnn)  # , argv=[sys.argv[0]] + unparsed)
