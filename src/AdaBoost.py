# import numpy as np
# from src.HaarFeature import HaarLikeFeature, create_features
# from src.HaarFeature import FeatureTypes
#
# from console_progressbar import ProgressBar
# from functools import partial
#
#
# def _get_feature_vote(feature, image):
#     return feature.get_vote(image)
#
#
# def _create_features(img_height, img_width, min_feature_width, max_feature_width, min_feature_height,
#                      max_feature_height):
#     print('Creating haar-like features..')
#     features = []
#     for feature in FeatureTypes:
#         # FeatureTypes are just tuples
#         feature_start_width = max(min_feature_width, feature[0])
#         for feature_width in range(feature_start_width, max_feature_width + feature[0], feature[0]):
#             feature_start_height = max(min_feature_height, feature[1])
#             for feature_height in range(feature_start_height, max_feature_height + feature[1], feature[1]):
#                 for x in range(img_width - feature_width):
#                     for y in range(img_height - feature_height):
#                         features.append(HaarLikeFeature(feature, (x, y), feature_width, feature_height, 0, 1))
#                         features.append(HaarLikeFeature(feature, (x, y), feature_width, feature_height, 0, -1))
#     print('..done. ' + str(len(features)) + ' features created.\n')
#     return features
#
#
# def adaboost(positive_iis, negative_iis, num_rounds):
#     """
#     :param num_rounds:
#     :rtype: object
#     :param positive_iis: faces_ii_training, list of integral image of each image in the face images set
#     :param negative_iis: non_faces_ii_training, ist of integral image of each image in the non-face images set
#     """
#     # boosting constants
#     min_feature_height = 8
#     max_feature_height = 8
#     min_feature_width = 8
#     max_feature_width = 12
#
#     num_pos = len(positive_iis)
#     num_neg = len(negative_iis)
#     num_imgs = num_pos + num_neg
#     img_height, img_width = positive_iis[0].shape
#
#     images = positive_iis + negative_iis
#
#     # Create features for all sizes and locations
#     features = _create_features(img_height, img_width, min_feature_width, max_feature_width, min_feature_height,
#                                 max_feature_height)
#     num_features = len(features)
#     feature_indexes = list(range(num_features))
#
#     # Calculating scores
#     print('Calculating scores for images..')
#
#     votes = np.zeros((num_imgs, num_features))
#     pb = ProgressBar(total=num_imgs, prefix='Computing score', suffix='finished', decimals=1, length=50, fill='X',
#                      zfill='-')
#
#     for i in range(num_imgs):
#         votes[i, :] = np.array(list(map(partial(_get_feature_vote, image=images[i]), features)))
#         pb.print_progress_bar(i)
#
#     print('\n Score created\n')
#
#     # Boosting
#     print('Selecting classifiers..')
#     # Create initial weights and labels
#     pos_weights = np.ones(num_pos) * 1. / (2 * num_pos)
#     neg_weights = np.ones(num_neg) * 1. / (2 * num_neg)
#     weights = np.hstack((pos_weights, neg_weights))
#     labels = np.hstack((np.ones(num_pos), np.ones(num_neg) * -1))
#
#
#     classifiers = []
#
#     pb = ProgressBar(total=num_rounds, prefix='Computing score', suffix='finished', decimals=1, length=50, fill='X',
#                      zfill='-')
#     for i in range(num_rounds):
#
#         classification_errors = np.zeros(len(feature_indexes))
#
#         # normalize weights
#         weights *= 1. / np.sum(weights)
#
#         # select best classifier based on the weighted error
#         for f in range(len(feature_indexes)):
#             f_idx = feature_indexes[f]
#             # classifier error is the sum of image weights where the classifier
#             # is right
#             error = sum(
#                 map(lambda img_idx:
#                     weights[img_idx]
#                     if labels[img_idx] != votes[img_idx, f_idx]
#                     else 0,
#                     range(num_imgs))
#             )
#             classification_errors[f] = error
#
#         # get best feature, i.e. with smallest error
#         min_error_idx = np.argmin(classification_errors)
#         best_error = classification_errors[min_error_idx]
#         best_feature_idx = feature_indexes[min_error_idx]
#         print('best_feature_idx is', best_feature_idx)
#         print('best_error is', best_error)
#
#         # set feature weight
#         best_feature = features[best_feature_idx]
#         feature_weight = 0.5 * np.log((1 - best_error) / best_error)
#         best_feature.weight = feature_weight
#         classifiers.append(best_feature)
#         print(best_feature.polarity, best_feature.threshold, best_feature.top_left, best_feature.type)
#
#         # update image weights
#         weights = np.array(list(
#             map(lambda img_idx:
#                 weights[img_idx] * np.sqrt((1 - best_error) / best_error)
#                 if labels[img_idx] != votes[img_idx, best_feature_idx]
#                 else weights[img_idx] * np.sqrt(best_error / (1 - best_error)),
#                 range(num_imgs))
#         ))
#
#         # remove feature (a feature can't be selected twice)
#         feature_indexes.remove(best_feature_idx)
#
#         pb.print_progress_bar(i)
#
#     print('\n done with boosting')
#
#     return classifiers
