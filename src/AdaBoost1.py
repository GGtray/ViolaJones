import numpy as np

from src.HaarFeature import create_features
from console_progressbar import ProgressBar
from functools import partial
from src.classifiers import WeakClassifier

from src.classifiers import build_running_sums, find_best_threshold


def _get_feature_value(feature, image):
    return feature.get_value(image)


def weak_classifier(x_feature: float, polarity: float, threshold: float) -> int:
    """
    naive classifier in boosting
    :param threshold: threshold
    :param x_feature:
    :param polarity:
    :return:
    :rtype: int
    """
    return 1 if (polarity * x_feature) < (polarity * threshold) else -1
    # return (np.sign((polarity * theta) - (polarity * x_feature)) + 1) // 2


def strong_classifier(weak_classifiers: list, x) -> int:
    weak_vote = []
    alpha_list = []
    for i in range(len(weak_classifiers)):
        weak_vote.append(weak_classifiers[i].get_vote(x))
        alpha_list.append(weak_classifiers[i].alpha)

    return np.sign(sum(np.multiply(weak_vote, alpha_list)))


def adaboost(positive_iis, negative_iis, min_feature_height, max_feature_height, min_feature_width, max_feature_width, num_rounds=2) -> list:
    """
    AdaBoosting step for the face Detection, return all weak classifiers
    :param max_feature_width:
    :param min_feature_width:
    :param max_feature_height:
    :param min_feature_height:
    :param num_rounds: number of the adaboosting rounds
    :param positive_iis: faces_ii_training, list of integral image of each image in the face images set
    :param negative_iis: non_faces_ii_training, ist of integral image of each image in the non-face images set
    :rtype: a list of classifiers
    """

    num_pos = len(positive_iis)
    num_neg = len(negative_iis)
    num_imgs = num_pos + num_neg
    img_height, img_width = positive_iis[0].shape

    images = positive_iis + negative_iis

    # Create features for all sizes and locations
    features = create_features(img_height, img_width, min_feature_width, max_feature_width, min_feature_height,
                               max_feature_height)
    num_features = len(features)
    # feature_indexes = list(range(num_features))

    # Calculating feature values
    print('Calculating scores for images..')
    feature_values = np.zeros((num_imgs, num_features))
    pb = ProgressBar(total=num_imgs, prefix='Computing score', suffix='finished', decimals=1, length=50, fill='X',
                     zfill='-')

    for i in range(num_imgs):
        pb.print_progress_bar(i)
        feature_values[i, :] = np.array(list(map(partial(_get_feature_value, image=images[i]), features)))

    print('\n Score created\n')

    # Boosting
    print('Selecting classifiers..')
    # Create initial weights and labels
    pos_weights = np.ones(num_pos) * 1. / (2 * num_pos)
    neg_weights = np.ones(num_neg) * 1. / (2 * num_neg)
    weights = np.hstack((pos_weights, neg_weights))
    labels = np.hstack((np.ones(num_pos), np.ones(num_neg) * -1))

    weak_classifiers = []

    pb = ProgressBar(total=num_rounds, prefix='Computing score', suffix='finished', decimals=1, length=50, fill='X',
                     zfill='-')
    for i in range(num_rounds):
        pb.print_progress_bar(i)

        # normalize weights
        weights *= 1. / np.sum(weights)

        # select best classifier based on the weighted error
        classification_errors = []
        error_each_feature_list = []
        threshold_list = []
        polarity_list = []
        featureColidx: int  # index of the feature
        for featureColidx in range(len(feature_values[0])):
            sampleCol = feature_values[:, featureColidx]
            p = np.argsort(sampleCol)
            sampleCol_p, labels_p, weights_p = sampleCol[p], labels[p], weights[p]
            t_minus, t_plus, s_minuses, s_pluses = build_running_sums(labels_p, weights_p)
            currentThreshold, currentPolarity, min_error = find_best_threshold(sampleCol_p, t_minus, t_plus, s_minuses,
                                                                               s_pluses)
            error_each_feature_list.append(min_error)
            threshold_list.append(currentThreshold)
            polarity_list.append(currentPolarity)
            classification_errors.append(min_error)

        best_feature_idx = np.argmin(error_each_feature_list)
        best_feature_idx.astype(np.int)
        best_error = error_each_feature_list[best_feature_idx]
        print('\n feature number ', best_feature_idx)
        print('feature type is ', features[best_feature_idx].type)
        print('feature position ', features[best_feature_idx].top_left)
        print('feature width ', features[best_feature_idx].width)
        print('feature height ', features[best_feature_idx].height)

        best_feature = features[best_feature_idx]
        feature_weight = 0.5 * np.log((1 - best_error) / best_error)

        # update image weights
        beta = best_error / (1 - best_error)
        alpha = np.log(1 / beta)
        weights = np.array(list(
            map(lambda img_idx:
                weights[img_idx] * np.sqrt((1 - best_error) / best_error)
                if labels[img_idx] !=
                   weak_classifier(feature_values[img_idx, best_feature_idx], polarity_list[best_feature_idx],
                                   threshold_list[best_feature_idx])
                else weights[img_idx] * np.sqrt(best_error / (1 - best_error)),
                range(num_imgs))
        ))

        # # remove feature (a feature can't be selected twice)
        # feature_indexes.remove(best_feature_idx)
        weak_classifiers.append(
            WeakClassifier(threshold_list[best_feature_idx], polarity_list[best_feature_idx], best_feature_idx, alpha))

    print('\n done with boosting')

    return weak_classifiers
