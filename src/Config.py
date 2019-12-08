import configparser

config = configparser.ConfigParser()
config['PATH'] = {
    'pos_train_path': 'dataset/trainset/faces',
}

with open('config.ini', 'w') as configfile:
    config.write(configfile)


pos_training_path = 'dataset/trainset/faces'
neg_training_path = 'dataset/trainset/non-faces'
pos_testing_path = 'dataset/testset/faces'
neg_testing_path = 'dataset/testset/non-faces'
