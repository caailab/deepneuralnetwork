#-*- coding: utf-8 -*-
_author_ = 'Licheng QU'

import os
import timeit
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdate

#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()

import configuration as cf
import trafficdata as tfdata


def evaluate_model(test_x, test_y, num_features, num_labels, factor, model_file):
    start = timeit.default_timer()

    with tf.Session() as sess:
        # Initialize all the tensor
        if tf.gfile.Exists(model_file + '.meta'):
            print('Restore Model from ' + model_file)
            saver = tf.train.import_meta_graph(model_file + '.meta')
            saver.restore(sess, model_file)
            print('Restore Model ... Done')
        else:
            print('\nERROR ! Missing the Model files\t' + model_file)
            print('I can NOT finish the prediction.\n')
            exit(-1)

        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        y_ = tf.get_collection('y_')[0]
        error = tf.abs(y - y_)

        test_data_size = len(test_x)
        print('\ntest_data_size=', test_data_size)

        batch_xs = test_x.reshape(-1, num_features)
        batch_ys = test_y.reshape(-1, num_labels)

        pred_y, pred_bias = sess.run([y, error], feed_dict={x: batch_xs, y_: batch_ys})

        end = timeit.default_timer()
        print("Evaluate Time: {}s".format(end - start))

        pred_bias_percent = pred_bias / batch_ys

        print('Absolute Error: mean={}, min={}, max={}, var={}'.format(np.mean(pred_bias),
                                                                       np.min(pred_bias),
                                                                       np.max(pred_bias),
                                                                       np.var(pred_bias)))
        print('Absolute Percentage Error: mean={}, min={}, max={}, var={}'.format(np.mean(pred_bias_percent),
                                                                                  np.min(pred_bias_percent),
                                                                                  np.max(pred_bias_percent),
                                                                                  np.var(pred_bias_percent)))
        print('\n')
        print('Test MAE = {}'.format(np.mean(pred_bias)))
        print('Test RMSE = {}'.format(np.sqrt(np.var(pred_bias))))
        print('Test SE = {}'.format(np.std(pred_bias)))
        print('Test MAPE = {}'.format(np.mean(pred_bias_percent)))

        return pred_y[:, 0], pred_bias[:, 0], pred_bias_percent[:, 0]


def main_dnn_evaluate(_):
    config_file = '../conf-005es18066/traffic_dnn-05min.conf'
    config_file = '../conf-005es18066/traffic_dnn-05min-b1152.conf'
    # config_file = '../conf-005es18066/traffic_dnn-minute10-b288.conf'
    # config_file = '../conf-005es18066/traffic_dnn-minute15-b288.conf'
    # config_file = '../conf-005es18066/traffic_dnn-minute20-b288.conf'
    # config_file = '../conf-005es18066/traffic_dnn-minute30-b288.conf'
    # config_file = '../conf-005es18066/traffic_dnn-minute60-b288.conf'

    # config_file = '../conf/traffic_dnn-05min.conf'
    # config_file = '../conf/traffic_dnn-minute10.conf'
    # config_file = '../conf/traffic_dnn-minute15.conf'
    # config_file = '../conf/traffic_dnn-minute20.conf'
    # config_file = '../conf/traffic_dnn-minute30.conf'
    # config_file = '../conf/traffic_dnn-minute60.conf'

    config = cf.Config(config_file)

    # 'model-005es18115-minute05-15182295'
    #config.model_name = 'model-005es18066-minute05'
    model_name_postfix = '-15182295'
    evaluate_with_config(config, model_name_postfix, True)


def evaluate_with_config(config, model_name_postfix, showwindow):
    model_name_prefix = config.model_name
    model_path = config.model_path        #'../model-minute/'

    model_name = model_name_prefix + model_name_postfix
    traffic_prediction_model_file = model_path + model_name + '/traffic_prediction.model'

    #model_image_path = model_path + 'evaluate/'
    model_image_path = '../model/evaluate/'
    if not os.path.isdir(model_image_path):
        os.makedirs(model_image_path)

    #model_result_path = model_path + 'evaluate/'
    model_result_path = '../model/evaluate/'
    if not os.path.isdir(model_result_path):
        os.makedirs(model_result_path)

    num_features = config.num_features      # 9
    num_labels = config.num_labels          # 1
    label_factor = config.label_factor      # 20000
    interval = config.interval              # 5

    # Import data
    test_data_file = config.test_data_file  # '../dataset/005es18066/18066-I-201603.csv'
    xtest, ytest, test_time = tfdata.load_traffic_data_resample(test_data_file, interval)
    #xtest, ytest, test_time = tfdata.load_traffic_data_clean(test_data_file, interval)
    # print(xdata, ydata)

    # only evaluate the traffic between 6:00 and 22:00
    data_filter = np.logical_and(xtest[:, 3] >= 6, xtest[:, 3] < 22)
    xtest = xtest[data_filter]
    ytest = ytest[data_filter]
    test_time = test_time[data_filter]

    # test_data_size = len(xdata)
    test_x, test_y = tfdata.normalize_data(xtest, ytest, label_factor)
    test_x[:, 8] = 0.0  # rain

    test_x = test_x.reshape(-1, num_features)
    test_y = test_y.reshape(-1, num_labels)

    print('Test dataset:\tFeatures Shape={}\tLabels Shape={}'.format(test_x.shape, test_y.shape))
    print(test_x, test_y)

    # Evaluate trained model
    pred_y, _, _ = evaluate_model(test_x, test_y, num_features, num_labels, label_factor,
                                                         traffic_prediction_model_file)

    # print(test_y.shape, pred_y.shape, pred_bias.shape, pred_bias_percent.shape)
    # for i in range(len(test_y)):
    #    print("{},{},{},{}".format(test_y[i], pred_y[i], pred_bias[i], pred_bias_percent[i]))

    # train_writer.close()
    # test_writer.close()

    #pred_y = tfdata.unnormalize_data(pred_y, label_factor).flatten()
    #test_y = tfdata.unnormalize_data(test_y, label_factor).flatten()
    #pred_bias = tfdata.unnormalize_data(pred_bias, label_factor)
    test_y = test_y.flatten()
    test_y = tfdata.unnormalize_data(test_y, label_factor)
    pred_y = tfdata.unnormalize_data(pred_y, label_factor)
    pred_bias = np.abs(pred_y - test_y)
    pred_bias_percent = np.abs(pred_bias / test_y) * 100
    loss_evaluate = pred_bias_percent
    #loss_evaluate = loss_evaluate[loss_evaluate[:]< 0.2]

    write_result(test_time, test_y, pred_y, pred_bias, pred_bias_percent, model_name, model_result_path)

    visualize_model(test_time, test_y, pred_y, pred_bias, pred_bias_percent, loss_evaluate, model_name, model_image_path, showwindow)


def write_result(test_time, test_y, pred_y, pred_bias, pred_bias_percent, model_name, path='./'):
    postfix = '' #+ datetime.now().strftime('%Y%m%d') #%H%M%S')

    time = pd.to_datetime(test_time.flatten())

    rf = open(path + model_name + '-evaluate-result' + postfix + '.csv', 'w')
    for i in range(len(test_y)):
        rf.write("{},{},{},{},{}\n".format(time[i], test_y[i], pred_y[i], pred_bias[i], pred_bias_percent[i]))
    rf.close()


def visualize_model(test_time, test_y, pred_y, pred_bias, pred_bias_percent, loss_epoch, title, path='./', showwindow=False):
    plt.clf()
    plt.figure(num=1, figsize=(8, 6))
    plt.rc('font', family='Times New Roman')
    #plt.subplots_adjust(hspace = 0.4)

    ax1 = plt.subplot(211)
    plt.sca(ax1)

    plt.title(title, size=14)
    plt.title(title + "\nMAE={:.3f}  MAPE {:.3f}%  RMSE={:.3f}".format(np.mean(pred_bias), np.mean(pred_bias_percent), np.mean(np.std(pred_bias))), size=12)
    #plt.text(5, 0.65, 'MAE={}'.format(np.mean(pred_bias)))
    #plt.text(5, 0.60, 'MAPE={}'.format(np.mean(pred_bias_percent)))

    #plt.xlabel('Time Serials', size=12)
    plt.ylabel('Volume (vehs/5min)', size=12)
    #ax1.xaxis.set_major_formatter(mdate.DateFormatter('%m-%d')) #'%Y-%m-%d %H:%M:%S'))
    #plt.gcf().autofmt_xdate()
    plt.plot(test_time, test_y, color='b', linestyle='-', label='observed')  # marker='o',
    plt.plot(test_time, pred_y, color='r', linestyle='-', label='predicted')  # marker='*',
    #plt.plot(pred_bias, color='y', linestyle='-', label='bias')  # marker='+',
    #plt.plot(pred_bias_percent, color='g', linestyle='-', marker='.', label='bias (%)')

    #plt.legend(loc='upper right')

    # Plot the loss_epoch
    ax2 = plt.subplot(212)
    plt.sca(ax2)

    plt.xlabel('Time Serials', size=12)
    plt.ylabel('APE (%)', size=12)
    ax2.xaxis.set_major_formatter(mdate.DateFormatter('%m-%d'))  # '%Y-%m-%d %H:%M:%S'))
    plt.gcf().autofmt_xdate()
    plt.plot(test_time, loss_epoch, color='m', linestyle='-', label='loss')  # marker='o',

    plt.savefig(path + title + '-evaluate-' + datetime.now().strftime('%Y%m%d') + '.png', format='png')
    if showwindow:
        plt.show()

    plt.close()


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
    #                    help='Directory for storing input data')
    # FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main_dnn_evaluate)  # , argv=[sys.argv[0]] + unparsed)
