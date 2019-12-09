from src.Utils import sum_region


def enum(**enums):
    return type('Enum', (), enums)


FeatureType = enum(TWO_VERTICAL=(2, 1), TWO_HORIZONTAL=(1, 2), THREE_HORIZONTAL=(3, 1), THREE_VERTICAL=(1, 3),
                   FOUR=(2, 2))  # (width, height)
FeatureTypes = [FeatureType.TWO_VERTICAL, FeatureType.TWO_HORIZONTAL, FeatureType.THREE_VERTICAL,
                FeatureType.THREE_HORIZONTAL, FeatureType.FOUR]


def create_features(img_height: int, img_width: int, min_feature_width: int, max_feature_width: int,
                    min_feature_height: int,
                    max_feature_height: int) -> list:
    """
    create features given image parameter
    :param img_height: height of the image, constant
    :param img_width: width of the image, constant
    :param min_feature_width: the minimum of
    :param max_feature_width:
    :param min_feature_height:
    :param max_feature_height:
    :return:
    """
    print('Creating features ...')
    features = []
    for feature in FeatureTypes:
        feature_size_prefix = len(features)  # record current size before computing
        feature_start_width = max(min_feature_width, feature[1])
        for feature_width in range(feature_start_width, max_feature_width + feature[1], feature[1]):
            feature_start_height = max(min_feature_height, feature[0])
            for feature_height in range(feature_start_height, max_feature_height + feature[0], feature[0]):
                for x in range(img_width - feature_width):
                    for y in range(img_height - feature_height):
                        features.append(HaarFeature(feature, (x, y), feature_width, feature_height))
        feature_size = len(features) - feature_size_prefix
        print(str(feature_size) + ' ' + str(feature) + ' features created')
    print('..done. ' + str(len(features)) + ' features created.\n')
    return features


class HaarFeature(object):
    def __init__(self, feature_type, position, width, height):
        self.type = feature_type
        self.top_left = position
        self.bottom_right = (position[0] + width, position[1] + height)
        self.width = width
        self.height = height

    def get_feature_value(self, integral_img):
        """
        Get score for given integral image array.
        :param integral_img: Integral image array
        :type integral_img: numpy.ndarray
        :return: Score for given feature
        :rtype: float
        """
        feature_value = 0
        if self.type == FeatureType.TWO_VERTICAL:
            first = sum_region(integral_img, self.top_left,
                               (self.top_left[0] + self.width, int(self.top_left[1] + self.height / 2)))
            second = sum_region(integral_img, (self.top_left[0], int(self.top_left[1] + self.height / 2)),
                                self.bottom_right)
            feature_value = first - second
        elif self.type == FeatureType.TWO_HORIZONTAL:
            first = sum_region(integral_img, self.top_left,
                               (int(self.top_left[0] + self.width / 2), self.top_left[1] + self.height))
            second = sum_region(integral_img, (int(self.top_left[0] + self.width / 2), self.top_left[1]),
                                self.bottom_right)
            feature_value = first - second
        elif self.type == FeatureType.THREE_HORIZONTAL:
            first = sum_region(integral_img, self.top_left,
                               (int(self.top_left[0] + self.width / 3), self.top_left[1] + self.height))
            second = sum_region(integral_img, (int(self.top_left[0] + self.width / 3), self.top_left[1]),
                                (int(self.top_left[0] + 2 * self.width / 3), self.top_left[1] + self.height))
            third = sum_region(integral_img, (int(self.top_left[0] + 2 * self.width / 3), self.top_left[1]),
                               self.bottom_right)
            feature_value = first - second + third
        elif self.type == FeatureType.THREE_VERTICAL:
            first = sum_region(integral_img, self.top_left,
                               (self.bottom_right[0], int(self.top_left[1] + self.height / 3)))
            second = sum_region(integral_img, (self.top_left[0], int(self.top_left[1] + self.height / 3)),
                                (self.bottom_right[0], int(self.top_left[1] + 2 * self.height / 3)))
            third = sum_region(integral_img, (self.top_left[0], int(self.top_left[1] + 2 * self.height / 3)),
                               self.bottom_right)
            feature_value = first - second + third
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
            feature_value = first - second - third + fourth

        return feature_value

    def get_value(self, integral_img):
        """
        Get vote of this feature for given integral image.
        :param integral_img:
        :param int_img: Integral image array
        :type int_img: numpy.ndarray
        :return: 1 iff this feature votes positively, otherwise -1
        :rtype: int
        """
        value = self.get_feature_value(integral_img)
        return value
