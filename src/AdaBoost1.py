import numpy as np
from src.HaarFeature import HaarFeature, create_features
from src.HaarFeature import FeatureTypes

from console_progressbar import ProgressBar
from functools import partial

from src.classifiers import build_running_sums, find_best_threshold


def _get_feature_value(feature, image):
    return feature.get_value(image)


def weak_classifier(x_feature, polarity: float, theta: float) -> float:
    return 1. if (polarity * x_feature) < (polarity * theta) else -1.
    # return (np.sign((polarity * theta) - (polarity * x_feature)) + 1) // 2


def adaboost(positive_iis, negative_iis, num_rounds=2):
    """
    AdaBoosting step for the face Detection
    :param num_rounds: number of the adaboosting rounds
    :param positive_iis: faces_ii_training, list of integral image of each image in the face images set
    :param negative_iis: non_faces_ii_training, ist of integral image of each image in the non-face images set
    :rtype: classifiers
    """
    # feature extracting constants, which determines how many features will be generated
    min_feature_height = 8
    max_feature_height = 8
    min_feature_width = 8
    max_feature_width = 8

    num_pos = len(positive_iis)
    num_neg = len(negative_iis)
    num_imgs = num_pos + num_neg
    img_height, img_width = positive_iis[0].shape

    images = positive_iis + negative_iis

    # Create features for all sizes and locations
    features = create_features(img_height, img_width, min_feature_width, max_feature_width, min_feature_height,
                               max_feature_height)
    num_features = len(features)
    feature_indexes = list(range(num_features))

    # Calculating feature values
    print('Calculating scores for images..')

    feature_values = np.zeros((num_imgs, num_features))
    pb = ProgressBar(total=num_imgs, prefix='Computing score', suffix='finished', decimals=1, length=50, fill='X',
                     zfill='-')

    for i in range(num_imgs):
        feature_values[i, :] = np.array(list(map(partial(_get_feature_value, image=images[i]), features)))
        pb.print_progress_bar(i)

    print('\n Score created\n')

    # Boosting
    print('Selecting classifiers..')
    # Create initial weights and labels
    pos_weights = np.ones(num_pos) * 1. / (2 * num_pos)
    neg_weights = np.ones(num_neg) * 1. / (2 * num_neg)
    weights = np.hstack((pos_weights, neg_weights))
    labels = np.hstack((np.ones(num_pos), np.ones(num_neg) * -1))

    classifiers = []

    pb = ProgressBar(total=num_rounds, prefix='Computing score', suffix='finished', decimals=1, length=50, fill='X',
                     zfill='-')
    for i in range(num_rounds):
        pb.print_progress_bar(i)
        classification_errors = np.zeros(len(feature_indexes))

        # normalize weights
        weights *= 1. / np.sum(weights)

        # select best classifier based on the weighted error
        min_error_list = []
        threshold_list = []
        polarity_list = []
        for featureColidx in range(len(feature_values[0])):
            featureCol = feature_values[:, featureColidx]
            p = np.argsort(featureCol)
            featureColp, labelsp, weightsp = featureCol[p], labels[p], weights[p]
            t_minus, t_plus, s_minuses, s_pluses = build_running_sums(labelsp, weightsp)
            currentThreshold, currentPolarity, min_error = find_best_threshold(featureColp, t_minus, t_plus, s_minuses,
                                                                               s_pluses)
            min_error_list.append(min_error)
            threshold_list.append(currentThreshold)
            polarity_list.append(currentPolarity)

        min_error_idx = np.argmin(min_error_list)
        best_error = min_error_list[min_error_idx]
        best_feature_idx = min_error_idx
        # print(min_error_idx)
        print('\n error rate is ', min_error_list[np.argmin(min_error_list)])
        print('feature type is ', features[min_error_idx].type)
        print(' feature position ', features[min_error_idx].top_left)
        print(' feature width ', features[min_error_idx].width)
        print(' feature height ', features[min_error_idx].height)

        best_feature = features[best_feature_idx]
        feature_weight = 0.5 * np.log((1 - best_error) / best_error)
        best_feature.weight = feature_weight
        classifiers.append(best_feature)

        # update image weights
        weights = np.array(list(
            map(lambda img_idx:
                weights[img_idx] * np.sqrt((1 - best_error) / best_error)
                if labels[img_idx] !=
                   weak_classifier(feature_values[img_idx, best_feature_idx], polarity_list[min_error_idx],
                                   threshold_list[min_error_idx])
                else weights[img_idx] * np.sqrt(best_error / (1 - best_error)),
                range(num_imgs))
        ))

        # remove feature (a feature can't be selected twice)
        feature_indexes.remove(best_feature_idx)

    print('\n done with boosting')

    return classifiers
