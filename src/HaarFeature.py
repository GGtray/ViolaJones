from src.Utils import sum_region


def enum(**enums):
    return type('Enum', (), enums)


FeatureType = enum(TWO_VERTICAL=(1, 2), TWO_HORIZONTAL=(2, 1), THREE_HORIZONTAL=(3, 1), THREE_VERTICAL=(1, 3),
                   FOUR=(2, 2))
FeatureTypes = [FeatureType.TWO_VERTICAL, FeatureType.TWO_HORIZONTAL, FeatureType.THREE_VERTICAL,
                FeatureType.THREE_HORIZONTAL, FeatureType.FOUR]


def create_features(img_height, img_width, min_feature_width, max_feature_width, min_feature_height,
                    max_feature_height):
    print('Creating features ...')
    features = []
    for feature in FeatureTypes:
        feature_size_prefix = len(features)  # record current size before computing
        feature_start_width = max(min_feature_width, feature[0])
        for feature_width in range(feature_start_width, max_feature_width, feature[0]):
            feature_start_height = max(min_feature_height, feature[1])
            for feature_height in range(feature_start_height, max_feature_height, feature[1]):
                for x in range(img_width - feature_width):
                    for y in range(img_height - feature_height):
                        features.append(HaarFeature(feature, (x, y), feature_width, feature_height, 0, 1))
                        features.append(HaarFeature(feature, (x, y), feature_width, feature_height, 0, -1))
        feature_size = len(features) - feature_size_prefix
        print(str(feature_size) + ' ' + str(feature) + ' features created')
    print('..done. ' + str(len(features)) + ' features created.\n')
    return features
    # print('Creating haar-like features..')
    # features = []
    # for feature in FeatureTypes:
    #     # FeatureTypes are just tuples
    #     feature_start_width = max(min_feature_width, feature[0])
    #     for feature_width in range(feature_start_width, max_feature_width, feature[0]):
    #         feature_start_height = max(min_feature_height, feature[1])
    #         for feature_height in range(feature_start_height, max_feature_height, feature[1]):
    #             for x in range(img_width - feature_width):
    #                 for y in range(img_height - feature_height):
    #                     features.append(HaarFeature(feature, (x, y), feature_width, feature_height, 0, 1))
    #                     features.append(HaarFeature(feature, (x, y), feature_width, feature_height, 0, -1))
    # print('..done. ' + str(len(features)) + ' features created.\n')
    # return features



class HaarFeature(object):
    def __init__(self, feature_type, position, width, height, threshold, polarity):
        self.type = feature_type
        self.top_left = position
        self.bottom_right = (position[0] + width, position[1] + height)
        self.width = width
        self.height = height
        self.threshold = threshold
        self.polarity = polarity
        self.weight = 1

    def get_feature_value(self, integral_img):
        """
        Get score for given integral image array.
        :param integral_img: Integral image array
        :type integral_img: numpy.ndarray
        :return: Score for given feature
        :rtype: float
        """
        score = 0
        if self.type == FeatureType.TWO_VERTICAL:
            first = sum_region(integral_img, self.top_left,
                               (self.top_left[0] + self.width, int(self.top_left[1] + self.height / 2)))
            second = sum_region(integral_img, (self.top_left[0], int(self.top_left[1] + self.height / 2)),
                                self.bottom_right)
            return first - second
        elif self.type == FeatureType.TWO_HORIZONTAL:
            first = sum_region(integral_img, self.top_left,
                               (int(self.top_left[0] + self.width / 2), self.top_left[1] + self.height))
            second = sum_region(integral_img, (int(self.top_left[0] + self.width / 2), self.top_left[1]),
                                self.bottom_right)
            score = first - second
        elif self.type == FeatureType.THREE_HORIZONTAL:
            first = sum_region(integral_img, self.top_left,
                               (int(self.top_left[0] + self.width / 3), self.top_left[1] + self.height))
            second = sum_region(integral_img, (int(self.top_left[0] + self.width / 3), self.top_left[1]),
                                (int(self.top_left[0] + 2 * self.width / 3), self.top_left[1] + self.height))
            third = sum_region(integral_img, (int(self.top_left[0] + 2 * self.width / 3), self.top_left[1]),
                               self.bottom_right)
            score = first - second + third
        elif self.type == FeatureType.THREE_VERTICAL:
            first = sum_region(integral_img, self.top_left,
                               (self.bottom_right[0], int(self.top_left[1] + self.height / 3)))
            second = sum_region(integral_img, (self.top_left[0], int(self.top_left[1] + self.height / 3)),
                                (self.bottom_right[0], int(self.top_left[1] + 2 * self.height / 3)))
            third = sum_region(integral_img, (self.top_left[0], int(self.top_left[1] + 2 * self.height / 3)),
                               self.bottom_right)
            score = first - second + third
        elif self.type == FeatureType.FOUR:
            # top left area
            first = sum_region(integral_img, self.top_left,
                               (int(self.top_left[0] + self.width / 2), int(self.top_left[1] + self.height / 2)))
            # top right area
            second = sum_region(integral_img, (int(self.top_left[0] + self.width / 2), self.top_left[1]),
                                (self.bottom_right[0], int(self.top_left[1] + self.height / 2)))
            # bottom left area
            third = sum_region(integral_img, (self.top_left[0], int(self.top_left[1] + self.height / 2)),
                               (int(self.top_left[0] + self.width / 2), self.bottom_right[1]))
            # bottom right area
            fourth = sum_region(integral_img,
                                (int(self.top_left[0] + self.width / 2), int(self.top_left[1] + self.height / 2)),
                                self.bottom_right)
            score = first - second - third + fourth

        return score

    def get_vote(self, integral_img):
        """
        Get vote of this feature for given integral image.
        :param integral_img:
        :param int_img: Integral image array
        :type int_img: numpy.ndarray
        :return: 1 iff this feature votes positively, otherwise -1
        :rtype: int
        """
        score = self.get_feature_value(integral_img)
        return self.weight * (1 if score < self.polarity * self.threshold else -1)
