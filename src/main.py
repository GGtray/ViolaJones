from functools import partial
import numpy as np
from console_progressbar import ProgressBar

import src.Utils as utils
import cv2 as cv
from src.AdaBoost1 import adaboost, strong_classifier, _get_feature_value
from src.HaarFeature import create_features

pos_training_path = '../dataset/trainset/faces'
neg_training_path = '../dataset/trainset/non-faces'
pos_testing_path = '../dataset/testset/faces'
neg_testing_path = '../dataset/testset/non-faces'

min_feature_height = 6
max_feature_height = 8
min_feature_width = 6
max_feature_width = 8

print('Loading faces...')
faces_training = utils.load_images(pos_training_path)
faces_ii_training = list(map(cv.integral, faces_training))
print('...done. ' + str(len(faces_training)) + ' faces loaded.')

print('Loading non-faces...')
non_faces_training = utils.load_images(neg_training_path)
non_faces_ii_training = list(map(cv.integral, non_faces_training))
print('..done. ' + str(len(non_faces_training)) + ' non-faces loaded.\n')

classifiers1 = adaboost(faces_ii_training, non_faces_ii_training, min_feature_height, max_feature_height, min_feature_width, max_feature_width, 5)

# classifiers3 = adaboost(faces_ii_training, non_faces_ii_training, 3)
# classifiers5 = adaboost(faces_ii_training, non_faces_ii_training, 5)
# classifiers10 = adaboost(faces_ii_training, non_faces_ii_training, 10)

faces_testing = utils.load_images(pos_testing_path)
faces_ii_testing = list(map(cv.integral, faces_testing))
print('..done. ' + str(len(faces_testing)) + ' faces loaded.\n\nLoading test non faces..')
non_faces_testing = utils.load_images(neg_testing_path)
non_faces_ii_testing = list(map(cv.integral, non_faces_testing))
print('..done. ' + str(len(non_faces_testing)) + ' non faces loaded.\n')

print('Testing selected classifiers..')


# Calculating feature values
print('Calculating scores for test face images..')
# Create features for all sizes and locations
num_pos = len(faces_ii_testing)

img_height_face, img_width_face = faces_ii_testing[0].shape
features_face = create_features(img_height_face, img_width_face, min_feature_width, max_feature_width, min_feature_height,
                           max_feature_height)
num_features = len(features_face)

test_face_feature_values = np.zeros((num_pos, num_features))
pb = ProgressBar(total=num_pos, prefix='Computing score', suffix='finished', decimals=1, length=50, fill='X',
                 zfill='-')

for i in range(num_pos):
    pb.print_progress_bar(i)
    test_face_feature_values[i, :] = np.array(list(map(partial(_get_feature_value, image=faces_ii_testing[i]), features_face)))

correct_faces = 0
for i in range(num_pos):
    if strong_classifier(classifiers1, test_face_feature_values[i, :]) == 1:
        correct_faces += 1

false_positive = (num_pos - correct_faces) / num_pos
print('false positive is ', false_positive)

print('Calculating scores for test non face images..')
num_neg = len(non_faces_ii_testing)

img_height_non_face, img_width_non_face = non_faces_ii_testing[0].shape
# features_non_face = create_features(img_height_non_face, img_width_non_face, min_feature_width, max_feature_width, min_feature_height,
#                                 max_feature_height)
features_non_face = features_face
num_non_features = len(features_non_face)

test_non_face_feature_values = np.zeros((num_neg, num_non_features))
pb = ProgressBar(total=num_neg, prefix='Computing score', suffix='finished', decimals=1, length=50, fill='X',
                 zfill='-')

for i in range(num_neg):
    pb.print_progress_bar(i)
    test_non_face_feature_values[i, :] = np.array(list(map(partial(_get_feature_value, image=non_faces_ii_testing[i]), features_non_face)))

correct_non_faces = 0
for i in range(num_neg):
    if strong_classifier(classifiers1, test_non_face_feature_values[i, :]) == -1:
        correct_non_faces += 1

false_neg = (num_neg - correct_non_faces) / num_neg
print('false negative is ', false_neg)


accuracy = (correct_faces + correct_non_faces) / (num_pos + num_neg)
print('accuracy is ', accuracy)

