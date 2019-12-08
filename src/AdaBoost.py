import numpy as np
from src.HaarFeature import HaarFeature, create_features
from src.HaarFeature import FeatureTypes

from console_progressbar import ProgressBar
from functools import partial


def _get_feature_vote(feature, image):
    return feature.get_vote(image)


def adaboost(positive_iis, negative_iis):
    """
    :rtype: object
    :param positive_iis:
    :param negative_iis:
    """

    min_feature_height = 9
    max_feature_height = 10
    min_feature_width = 9
    max_feature_width = 10

    num_pos = len(positive_iis)
    num_neg = len(negative_iis)
    num_imgs = num_pos + num_neg
    img_height, img_width = positive_iis[0].shape

    # Create initial weights and labels
    pos_weights = np.ones(num_pos) * 1. / (2 * num_pos)
    neg_weights = np.ones(num_neg) * 1. / (2 * num_neg)
    weights = np.hstack((pos_weights, neg_weights))
    labels = np.hstack((np.ones(num_pos), np.ones(num_neg) * -1))

    images = positive_iis + negative_iis

    # Create features for all sizes and locations
    features = create_features(img_height, img_width, min_feature_width, max_feature_width, min_feature_height,
                               max_feature_height)
    num_features = len(features)
    feature_indexes = list(range(num_features))

    # Calculating scores
    print('Calculating scores for images..')

    votes = np.zeros((num_imgs, num_features))
    pb = ProgressBar(total=num_imgs, prefix='Computing score', suffix='finished', decimals=1, length=50, fill='X',
                     zfill='-')

    for i in range(num_imgs):
        votes[i, :] = np.array(list(map(partial(_get_feature_vote, image=images[i]), features)))
        pb.print_progress_bar(i)

    print('\n Score created\n')

    # Boosting
    print('Selecting classifiers..')

    classifiers = []

    num_classifiers = 2 ##num_features
    print('the number of classifier is ',num_classifiers)
    pb = ProgressBar(total=num_features, prefix='Computing score', suffix='finished\n', decimals=1, length=50, fill='X',
                     zfill='-')
    for _ in range(num_classifiers):

        classification_errors = np.zeros(len(feature_indexes))

        # normalize weights
        weights *= 1. / np.sum(weights)

        # select best classifier based on the weighted error
        for f in range(len(feature_indexes)):
            f_idx = feature_indexes[f]
            # classifier error is the sum of image weights where the classifier
            # is right
            error = sum(map(lambda img_idx: weights[img_idx] if labels[img_idx] != votes[img_idx, f_idx] else 0, range(num_imgs)))
            classification_errors[f] = error

        # get best feature, i.e. with smallest error
        min_error_idx = np.argmin(classification_errors)
        best_error = classification_errors[min_error_idx]
        best_feature_idx = feature_indexes[min_error_idx]

        # set feature weight
        best_feature = features[best_feature_idx]
        feature_weight = 0.5 * np.log((1 - best_error) / best_error)
        best_feature.weight = feature_weight

        classifiers.append(best_feature)

        # update image weights
        weights = np.array(list(map(lambda img_idx: weights[img_idx] * np.sqrt((1-best_error)/best_error) if labels[img_idx] != votes[img_idx, best_feature_idx] else weights[img_idx] * np.sqrt(best_error/(1-best_error)), range(num_imgs))))

        # remove feature (a feature can't be selected twice)
        feature_indexes.remove(best_feature_idx)
        pb.print_progress_bar(_)

        print('done with boosting')

    return classifiers
