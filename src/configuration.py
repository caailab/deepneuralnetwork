# coding: utf-8
_author_ = 'Licheng QU'

import configparser

class Config(object):

    def __init__(self, filename):
        self.configfile = filename

        conf = configparser.ConfigParser()
        conf.read(self.configfile)

        secs = conf.sections()
        print('conf section:', type(secs), secs)

        opts = conf.options('model')
        print('conf options', type(opts), opts)

        kvs = conf.items('model')
        print('model Key:Value', type(kvs), kvs)

        self.model_name = conf.get('model', 'name')
        print('model_name = ', self.model_name)
        self.model_path = conf.get('model', 'path')
        print('model_path = ', self.model_path)

        self.num_labels = conf.getint('data', 'labels')
        print('num_labels = ', self.num_labels)
        self.label_factor = conf.getint('data', 'factor')
        print('label_factor = ', self.label_factor)
        self.interval = conf.getint('data', 'interval')
        print('interval = ', self.interval)
        self.num_features = conf.getint('data', 'features')
        print('num_features = ', self.num_features)
        self.train_data_file = conf.get('data', 'trainfile')
        print('train_data_file = ', self.train_data_file)
        self.test_data_file = conf.get('data', 'testfile')
        print('test_data_file = ', self.test_data_file)

        self.epochs = conf.getint('train', 'epochs')
        print('epochs = ', self.epochs)
        self.batchsize = conf.getint('train', 'batchsize')
        print('batchsize = ', self.batchsize)
        self.learningrate = conf.getfloat('train', 'learningrate')
        print('learningrate = ', self.learningrate)
        self.keepneuron = conf.getfloat('train', 'keepneuron')
        print('keepneuron = ', self.keepneuron)

if __name__ == '__main__' :
    configfile = '../conf/traffic_dnn-60minute.conf'
    conf = Config(configfile)
    print(conf.configfile)
    print(conf.train_data_file)