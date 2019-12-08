import src.Utils as utils
import cv2 as cv

from src.HaarFeature import HaarFeature
from src.HaarFeature import FeatureTypes

from src.AdaBoost import adaboost


pos_training_path = '../dataset/trainset/faces'
neg_training_path = '../dataset/trainset/non-faces'
pos_testing_path = '../dataset/testset/faces'
neg_testing_path = '../dataset/testset/non-faces'

min_feature_height = 1
max_feature_height = 19
min_feature_width = 2
max_feature_width = 8

print('Loading faces...')
faces_training = utils.load_images(pos_training_path)
faces_ii_training = list(map(cv.integral, faces_training))
print('...done. ' + str(len(faces_training)) + ' faces loaded.')

print('Locaing non-faces...')
non_faces_training = utils.load_images(neg_training_path)
non_faces_ii_training = list(map(cv.integral, non_faces_training))
print('..done. ' + str(len(non_faces_training)) + ' non-faces loaded.\n')

adaboost(faces_ii_training, non_faces_ii_training)

faces_testing = utils.load_images(pos_testing_path)
faces_ii_testing = list(map(cv.integral, faces_testing))
print('..done. ' + str(len(faces_testing)) + ' faces loaded.\n\nLoading test non faces..')
non_faces_testing = utils.load_images(neg_testing_path)
non_faces_ii_testing = list(map(cv.integral, non_faces_testing))
print('..done. ' + str(len(non_faces_testing)) + ' non faces loaded.\n')

# print('Testing selected classifiers..')
# correct_faces = 0
# correct_non_faces = 0
# correct_faces = sum(utils.ensemble_vote_all(faces_ii_testing, classifiers))
# correct_non_faces = len(non_faces_testing) - sum(utils.ensemble_vote_all(non_faces_ii_testing, classifiers))
#
# print('..done.\n\nResult:\n      Faces: ' + str(correct_faces) + '/' + str(len(faces_testing))
#       + '  (' + str((float(correct_faces) / len(faces_testing)) * 100) + '%)\n  non-Faces: '
#       + str(correct_non_faces) + '/' + str(len(non_faces_testing)) + '  ('
#       + str((float(correct_non_faces) / len(non_faces_testing)) * 100) + '%)')
