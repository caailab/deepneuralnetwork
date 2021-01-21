#-*- coding: utf-8 -*-
_author_ = 'Licheng QU'

import os

#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()

import configuration as cf
import traffic_dnn_train as tdt
import trafficdata as tfdata


#tensor_board_log_dir = '/tmp/tensorflow/traffic_dnn/logs'
# tensorboard --logdir=/tmp/tensorflow/traffic_dnn/logs

def main_dnn_try(_):
    #if tf.gfile.Exists(tensor_board_log_dir):
    #    tf.gfile.DeleteRecursively(tensor_board_log_dir)
    #tf.gfile.MakeDirs(tensor_board_log_dir)

    config_file = '../conf-005es18066/traffic_dnn-05min.conf'
    conf = cf.Config(config_file)

    # mode saveing path
    model_name_prefix = conf.model_name
    model_path = conf.model_path        #'../model-minute/'
    model_image_path = model_path + 'images/'
    if not os.path.isdir(model_image_path):
        os.makedirs(model_image_path)

    num_features = conf.num_features    # 9
    num_labels = conf.num_labels        # 1
    label_factor = conf.label_factor    # 2000
    interval = conf.interval            # 5

    epochs = conf.epochs                # 10000
    batchsize = conf.batchsize          # 24 * 5
    learningrate = conf.learningrate    # 0.1

    # Read Training data
    train_data_file = conf.train_data_file  # '../dataset/hour_rain/2014/volume-005es16513-N-2014.csv'
    test_data_file = conf.test_data_file    # '../dataset/hour_rain/2015/volume-005es16513-N-201502.csv'

    xtrain, ytrain, train_time = tfdata.load_traffic_data_resample(train_data_file, interval)
    xtest, ytest, test_time = tfdata.load_traffic_data_resample(test_data_file, interval)
    # print(xtrain)
    # print(ytrain)
    # print(xtest)
    # print(ytest)

    # train_data_size = len(xtrain)
    # test_data_size = len(xtest)
    # print('Train Data Size={}\tTest Data Size={}'.format(train_data_size, test_data_size))

    train_x, train_y = tfdata.normalize_data(xtrain, ytrain, label_factor)
    test_x, test_y = tfdata.normalize_data(xtest, ytest, label_factor)

    train_x = train_x.reshape(-1, num_features)
    train_y = train_y.reshape(-1, num_labels)
    test_x = test_x.reshape(-1, num_features)
    test_y = test_y.reshape(-1, num_labels)

    print('Train dataset:\tFeatures Shape={}\tLabels Shape={}'.format(train_x.shape, train_y.shape))
    print(train_x)
    print(train_y)
    print('Test dataset:\tFeatures Shape={}\tLabels Shape={}'.format(test_x.shape, test_y.shape))
    print(test_x)
    print(test_y)

    hidden_layer_init = [15, 18, 22, 9, 5]   # 05min, batch_size=288
    for hl in range(4, 25):
        hidden_layer = hidden_layer_init.copy()
        hidden_layer.append(hl)

        model_name = model_name_prefix + '-{}'.format(''.join(str(h) for h in hidden_layer))
        traffic_prediction_model_file = model_path + model_name + '/traffic_prediction.model'
        #print(title)

        tf.reset_default_graph()

        # Training
        loss_epoch = tdt.train_model(train_x, train_y, num_features, num_labels, hidden_layer,
                                 traffic_prediction_model_file, epochs, batchsize, learningrate)

        tf.reset_default_graph()

        # Evaluate the model
        pred_y, pred_bias, pred_bias_percent = tdt.evaluate_model(test_x, test_y, num_features, num_labels, hidden_layer,
                                                              traffic_prediction_model_file)

        # print(test_y.shape, pred_y.shape, pred_bias.shape, pred_bias_percent.shape)
        # for i in range(len(test_y)):
        #    print("{},{},{},{}".format(test_y[i], pred_y[i], pred_bias[i], pred_bias_percent[i]))

        # train_writer.close()
        # test_writer.close()

        tdt.write_result(test_time, test_y.flatten(), pred_y, pred_bias, pred_bias_percent, loss_epoch, model_name, model_path)

        tdt.visualize_model(test_time, test_y, pred_y, pred_bias, pred_bias_percent, loss_epoch, model_name, model_image_path)


if __name__ == '__main__':
    tf.app.run(main=main_dnn_try)
