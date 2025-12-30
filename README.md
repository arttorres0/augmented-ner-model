# Augmented NER with BI-LSTM+CRF and BERT

This project contains several Jupyter Notebooks that create, train and evaluate BI-LSTM+CRF and BERT models for NER task. It was developed using 3 low-resource domain (highly-specialized language fields) datasets:  BioCreative V CDR Task, i2b2 2010, MaSciP. It was executed using Google Colab, but different datasets and platforms can be used, if proper adjustments to the notebook are made.

In this project, training data is augmented to generate more examples and verify the impact of data augmentation on the model overall eficiency. Random points from the training data can be removed as well, if desired.

## Requirements

- Google Colab
- High-end GPU (highly recommended for BERT models)
- Python3
- Tensorflow 1.x
- GoogleDrive and GoogleAuth (not mandatory, only used to store files)
- Pandas
- Numpy
- Keras
- Sklearn
- Transformers (HuggingFace)

## Setup

For all three datasets, if a train-dev-test split was not originally provided, we used 15% of the original data for testing and 85% for training, and we used 15% of the training data as validation data. To assess the impact of data augmentation in various low-resource scenarios, this study selected different sentence subsets from each training dataset: 50, 150, and 500 sentences. Additionally, subsets amounting to 25%, 50%, and 75% of the original dataset size are chosen, provided these subsets exceed 500 sentences. The validation and testing datasets remain unaltered throughout the evaluation.

We evaluate the efficacy of two text augmentation techniques — Mention Replacement (MR) and Contextual Word Replacement (CWR) — on two widely-used NER models, Bi-LSTM+CRF and BERT. MR replaces randomly-selected mentions, i.e., one or more contiguous tokens with a uniform label type, with other mentions from the original training set that have the same label type. The number of tokens in the replacement mention may differ from the original (e.g. "I took a medicine today for acute colitis." -> "I took a medicine today for **cancer**."). CWR, on the other hand, employs BERT models to replace words contextually within a sentence (e.g. "I took a medicine today for acute colitis." -> "I **got** a medicine **now** for acute colitis.").
 
For each dataset and its subsets, we gradually increased the amount of augmented data in the training, ranging from 0% - original data without augmented data - to 500% - original data plus 5 times the amount of original data as augmented data. Finally, we observed validation loss as a parameter to interrupt model training, if it stopped decreasing for 5 epochs. This setup yields 784 groups of models, considering the combination of all datasets and subsets, architectures, and augmentation techniques and amounts. Each group consists of 10 models trained from scratch, which we evaluated in the testing data, in order to check model variance. F1-score of each model was calculated on the respective test dataset, and we took the average for each of the groups. You can see these results in files "results_MR.xslx" and "results_CWR.xsls".

For more details please check our paper (on review).

## Results

Please check files "results_MR.xslx" and "results_CWR.xsls" for all model results. We observe a modest increase in model quality for lower-sized subsets before the model starts to reduce the F1-score value, when analyzing models augmented with MR. However, for the full dataset and other larger subsets, the average F1-score has not increased, maintaining or losing quality. The only exception to this was MaSciP (BERT), in which every subset benefits from data augmentation. This reduction in F1-score may be a consequence of the introduction of invalid augmented data, which had their original ground-truth labels altered by the augmentation technique. This suggests that data augmentation should be used with caution, as the injection of high noise into the training dataset may cause models to overfit these invalid instances.
Models augmented with CWR show similar results. However, we notice that more models benefit from data augmentation. This is an indicator that this technique produces more data variability than MR in the studied datasets, which makes sense, as BERT can suggest several different tokens. We also have more possible target replacements ("O" tokens) than when applying MR, which targets non-"O" tokens. On the other hand, we observe a more inconsistent pattern of improvement in F1-score across different subset sizes. This is also a consequence of the higher variability introduced by BERT, but on the negative side, where there is more noise introduction than actual valid augmented examples in the datasets.
We also picked the best augmentated model, in terms of average F1-score, for each dataset subset (if it exists), and took T-student tests to assess the improvement. We note that of the 112 best models, 11, or about 10%, were found to present statistically significant improvements in the average F1-score. The conclusions above are also evident here, that is, first, smaller subsets improve with data augmentation, unlike larger subsets, and, second, in spite of these promising results, there is not an augmentation amount that yields the best results.
