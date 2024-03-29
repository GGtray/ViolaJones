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
## Concept and Implementation

### Weak Classifiers

adaboost need weak classifiers that are easily generated. In this project, the classifiers we are dealing with is so called stump classifiers:

<a href="https://www.codecogs.com/eqnedit.php?latex=$$h(x,&space;p,&space;\theta,&space;f)&space;=&space;\begin{cases}&space;1&space;&&space;px(f)&space;<&space;p\theta\&space;-1&space;&&space;otherwise\end{cases}$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$h(x,&space;p,&space;\theta,&space;f)&space;=&space;\begin{cases}&space;1&space;&&space;px(f)&space;<&space;p\theta\&space;-1&space;&&space;otherwise\end{cases}$$" title="$$h(x, p, \theta, f) = \begin{cases} 1 & px(f) < p\theta\ -1 & otherwise\end{cases}$$" /></a>

the implementation of this classifier is a class

    class WeakClassifier:
        def __init__(self, threshold, polarity, feature_idx, alpha):
            self.feature_idx = feature_idx
            self.threshold = threshold
            self.polarity = polarity
            self.alpha = alpha
    
        def get_vote(self, x: np.ndarry) -> int:
            return 1 if (self.polarity * x[self.feature_idx]) < (self.polarity * self.threshold) else -1

take threshold, polarity, feature as state, while the example as input
### Boosting

while boosting, we need to select the best classifier each round, naively, you have infinite choice, then boosting will take infinite time. An more advanced brute force method is using the data sample value as threshold, which is iterate over all features, for each feature, using each value as threshold, find best feature which minimize the total error. But this will cause O(NK) time for selecting a weak classifier in each boosting round.

Fortunately, there is one way to do this over a single pass, by ranking.

For each feature, sorted the examples based on feature value. Find the optimal threshold for that feature as follow. For each element in the sorted list, four sums are maintained and evaluated: 

1. the total sum of positive example weights T + , 
2. the total sum of negative example weights T − , 
3. the sum of positive weights below the current example S+ 
4. the sum of negative weights below the current example S−. 

The error for a threshold which splits the range between the current and previous example in the sorted list is:

<a href="https://www.codecogs.com/eqnedit.php?latex=$$e&space;=&space;min(S^&plus;&space;&plus;(T^-&space;−S^-&space;),\&space;S^-&space;&plus;(T^&plus;&space;−S^&plus;&space;))$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$e&space;=&space;min(S^&plus;&space;&plus;(T^-&space;−S^-&space;),\&space;S^-&space;&plus;(T^&plus;&space;−S^&plus;&space;))$$" title="$$e = min(S^+ +(T^- −S^- ),\ S^- +(T^+ −S^+ ))$$" /></a>

And then, we iterate over all feature, find the best feature that minimize the error.

the correctness of this method can be explained by exploring the semantic meaning of T and S

S_plus + (T_minus - S_minus) represents the total weighted error that if you label all the sample negative before current example. because the S_plus means the error this hypothesis made before current example, while the (T_minus - S_minus) means the error after current example. In this case, polarity is 1

S_minus + (T_plus - S_plus) is the opposite case, which is when you label all the sample positive before current example. In this case, polarity is -1.

After each sucessful selection of the weak classifier, we updated the weighted error vector based on this error. then goes to anthoer round. Finally we assemble these weak classifiers to build a strong classifier.

In my program, the weak classifier is modeled as a class, becase each weak classifier has its own state: polarity, threshold, and coeffient alpha.

## Experiment
### Round 1

    weighted error rate is  0.12808216432865743
    feature type is  (1, 2)
    feature position  (8, 3)
    feature width  2
    feature height  8
    
    
    
 ![round1](https://github.com/GGtray/ViolaJones/blob/master/img/Round1.png)

### Round 3

    weighted error rate is  0.22950280548561788
    feature type is  (2, 1)
    feature position  (3, 4)
    feature width  2
    feature height  4
 ![Round3](https://github.com/GGtray/ViolaJones/blob/master/img/Round3.png)

### Round 5

    weighted error rate is  0.25430753901334124
    feature type is  (3, 1)
    feature position  (10, 15)
    feature width  2
    feature height  4
   
 ![Round5](https://github.com/GGtray/ViolaJones/blob/master/img/Round5.png)

### Round 10

    weighted error rate is  0.2614991947946847
    feature type is  (2, 2)
    feature position  (0, 0)
    feature width  2
    feature height  6
    
 ![Round10](https://github.com/GGtray/ViolaJones/blob/master/img/Round10.png)
 
 ## The Boosting process
 
 ## Experiment
 for 

    min_feature_height = 6
    max_feature_height = 8
    min_feature_width = 6
    max_feature_width = 8

it has 4654 features for total

the boosting result is 

### round 1

    false positive is  0.3177966101694915
    false negative is  0.20589705147426288
    accuracy is  0.772745653052972

### round 3

    false positive is  0.6483050847457628
    false negative is  0.11294352823588207
    accuracy is  0.7848766680145572

### round 5

    false positive is  0.7309322033898306
    false negative is  0.09145427286356822
    accuracy is  0.7864941366761019

### round 10

    false positive is  0.6546610169491526
    false negative is  0.05347326336831584
    accuracy is  0.831783259199353

## Reason of performance Improvement
We can see the accuracy is improved along with the round going. the reason is Adaboosting.
For each round, we selected a decision stump classifier which classified all samples with minimum weighted error, since the weighted error has been updated, this classifier put more attention on the samples which mis-classified by the exact previous classifier. thus we can see that the weighted error has no improvement, even became larger, becasue the classifiers are bouncing on the samples which is very hard to be classified.

But we can notice another fact, that is the accuracy is improved, why? becasue there is a constant associated with each classifier: alpha, which is smaller when the weighted error is larger, since the accuracy is measured by the strong classifier that is build by assembling these weak classifier, we can see that, if a weak classifier made a lot of mistake, we lower the weight of its vote, ensure that the accuracy can be improved.

Talking about another phenomeon: the false positive is hardly imporved. I think the reason is because the training feature is not enough. Although we do not need that many features to recognize a face, these features still need to be boosted from a large set of features, which is time consuming. As an experiment, I pick a small set of features to boost, which has some side-effects: the Adaboost may pick sub-optimal features, since we only pick one best feature each round, and give these feature low threshold. Thus the feature may be mistakely found in a non-face picture, since the threshold is not hight.
 
 # Justification of the threshold

from my perspective, to replace the empirical error with the false positive, or the false negative, we just need a little change when we are conducting the boosting.

That change takes place in the process when we select the best threshold for each feature. If we are using the empirical error, we want to minimize the total error, and we do as follow:

    def find_best_threshold(samples: np.ndarray, t_minus: float, t_plus: float, s_minuses: List[float],
                            s_pluses: List[float]):
        min_e = float('inf')
        min_z, polarity = 0, 0
        for z, s_m, s_p in zip(samples, s_minuses, s_pluses):
            error_1 = s_p + (t_minus - s_m)
            error_2 = s_m + (t_plus - s_p)
            if error_1 < min_e:
                min_e = error_1
                min_z = z
                polarity = -1
            elif error_2 < min_e:
                min_e = error_2
                min_z = z
                polarity = 1
        return [min_z, polarity, min_e]

and for the false positive, we change the code to 

    def find_best_threshold_false_positive(samples: np.ndarray, t_minus: float, t_plus: float, s_minuses: List[float],
                                            s_pluses: List[float]):
        min_false_positive = float('inf')
        best_threshold, polarity = 0, 0
        for sample_value, s_m, s_p in zip(samples, s_minuses, s_pluses):
            error_1 = s_p
            error_2 = s_m
            if error_1 < min_false_positive:
                min_false_positive = s_p
                best_threshold = sample_value
                polarity = -1
            elif error_2 < min_false_positive:
                min_false_positive = s_m
                best_threshold = sample_value
                polarity = 1
        return [best_threshold, polarity, min_false_positive]

since for now, we are searching the minimum false positive

Similarly, for boosting based on false negative, we find the best threshold as below

    def find_best_threshold_false_negative(samples: np.ndarray, t_minus: float, t_plus: float, s_minuses: List[float],
                                            s_pluses: List[float]):
        min_false_negative = float('inf')
        best_threshold, polarity = 0, 0
        for sample_value, s_m, s_p in zip(samples, s_minuses, s_pluses):
            error_1 = t_minus - s_m
            error_2 = t_plus - s_p
            if error_1 < min_false_negative:
                min_false_negative = error_1
                best_threshold = sample_value
                polarity = -1
            elif error_2 < min_false_negative:
                best_threshold = sample_value
                min_z = z
                polarity = 1
        return [best_threshold, polarity, min_false_negative]

## Experiment

for 

    min_feature_height = 6
    max_feature_height = 8
    min_feature_width = 6
    max_feature_width = 8

boosting on empirical error

    false positive is  0.7309322033898306
    false negative is  0.09145427286356822
    accuracy is  0.7864941366761019
    
boosting on false positivie 

    false positive is  0.0
    false negative is  1.0
    accuracy is  0.19086130206227253

boosting on false negative

    false positive is  1.0
    false negative is  0.0009995002498750624
    accuracy is  0.8083299636069551
