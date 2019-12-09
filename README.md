# ViolaJones
violajones final project
# Haar Feature

in this project the range for width and height is as such

    min_feature_height = 1
    max_feature_height = 10
    min_feature_width = 2
    max_feature_width = 8

thus there are 

    7350 TWO_VERTICAL features created
    8700 TWO_HORIZONTAL features created
    6090 THREE_HORIZONTAL features created
    5250 THREE_VERTICAL features created
    4200 FOUR features created

the way to compute the number of features are as follow

<a href="https://www.codecogs.com/eqnedit.php?latex=\(n(width_{feature})&space;=&space;width_{window}&space;-&space;width_{feature}&space;&plus;&space;1&space;\\&space;n(height_{feature})&space;=height_{window}&space;-&space;height_{feature}&space;&plus;&space;1\\&space;N_{width}&space;=&space;\sum_{i&space;=&space;min\&space;width}^{max\&space;width}&space;n(i)&space;\\&space;N_{height}&space;=&space;\sum_{i&space;=&space;min\&space;height}^{max&space;\&space;height}&space;n(i)&space;\\&space;N_{feature}&space;=&space;N_{width}&space;\times&space;N_{height}\)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\(n(width_{feature})&space;=&space;width_{window}&space;-&space;width_{feature}&space;&plus;&space;1&space;\\&space;n(height_{feature})&space;=height_{window}&space;-&space;height_{feature}&space;&plus;&space;1\\&space;N_{width}&space;=&space;\sum_{i&space;=&space;min\&space;width}^{max\&space;width}&space;n(i)&space;\\&space;N_{height}&space;=&space;\sum_{i&space;=&space;min\&space;height}^{max&space;\&space;height}&space;n(i)&space;\\&space;N_{feature}&space;=&space;N_{width}&space;\times&space;N_{height}\)" title="\(n(width_{feature}) = width_{window} - width_{feature} + 1 \\ n(height_{feature}) =height_{window} - height_{feature} + 1\\ N_{width} = \sum_{i = min\ width}^{max\ width} n(i) \\ N_{height} = \sum_{i = min\ height}^{max \ height} n(i) \\ N_{feature} = N_{width} \times N_{height}\)" /></a>

# AdaBoost
## Round 1

    error rate is  0.12808216432865743
    feature type is  (1, 2)
    feature position  (8, 3)
    feature width  2
    feature height  8

## Round 3

    error rate is  0.22950280548561788
    feature type is  (2, 1)
    feature position  (3, 4)
    feature width  2
    feature height  4

## Round 5

    error rate is  0.25430753901334124
    feature type is  (3, 1)
    feature position  (10, 15)
    feature width  2
    feature height  4

## Round 10

    error rate is  0.2614991947946847
    feature type is  (2, 2)
    feature position  (0, 0)
    feature width  2
    feature height  6
