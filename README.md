## A study on benefits of deep learning model calibration toward improving performance in class-imbalanced medical image classification tasks

In medical image classification tasks, it is common to find that the number of normal samples far exceeds the number of abnormal samples. In such class-imbalanced situations, reliable training of deep neural networks continues to be a major challenge. Under these circumstances, the predicted class confidence may be biased toward the majority class. Calibration has been suggested to alleviate some of these effects. However, there is insufficient analysis explaining when and whether calibrating a model would be beneficial in improving performance. 

In this study, we perform a systematic analysis of the effect of model calibration on its performance on two medical image modalities, namely, chest X-rays (CXRs) and fundus images, using various deep learning classifier backbones. For this, we study the following variations: (i) the degree of imbalances in the dataset used for training; (ii) calibration methods; and, (iii) two classification thresholds, namely,  default decision threshold of 0.5, and optimal threshold from precision-recall (PR) curves. 

Our results indicate that at the default operating threshold of 0.5, the performance achieved through calibration is significantly superior (p < 0.05) to an uncalibrated model. However, at the PR-guided threshold, these gains were not significantly different (p > 0.05). This finding holds for both image modalities and at varying degrees of imbalance.


## Code description
This repository contains the Jupyter notebook showing how the train the models, calibrate their probabiliites, measure performance metrics, compute optimal threshold using ROC curves and PR curves using the calibrated and unclibrated probabilites and compare their performance.

