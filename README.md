# NER with BI-LSTM+CRF

This project contains a Jupyter Notebook that creates, trains and evaluates BI-LSTM+CRF models. It was originally developed using a medical dataset from BioCreative[1] and executed using Google Colab, but different datasets and platforms can be used, if proper adjustments to the notebook are made.

In this notebook, training data can be augmented to generate more examples and verify the impact of data augmentation on the model overall eficiency. Random points from the training data can be removed as well, if desired.

## Requirements

- Google Colab
- High-end GPU (highly recommended)
- Python3
- Tensorflow 1.x
- GoogleDrive and GoogleAuth (not mandatory, only used to store files)
- Pandas
- Numpy
- Keras
- Sklearn

## Results

This notebook was executed to verify how effective data augmentation is for BI-LSTM+CRF models when running a NER task [1, 2, 3]. We used a combination of two text augmentation techniques: entity replacement, where we replace words of some tag with other words of the same tag (e.g: I want to see a *movie* -> I want to see a *tv show*); and random token removal, where we randomly remove words from a set of words of same tag (e.g: My name is *Arthur Torres* -> My name is *Torres*) [4, 5, 6].

The dataset has 3000 examples (2400 for training and 600 for testing, using our train/test split, which can be changed) [7]. The whole original training data was used, plus some variable amount of augmented examples, which ranged from 0% to 100% of the original data (e.g.: 50% means training data consisted of 2400 original examples and 1200 augmented ones). Around 10 models for each of these amounts were trained with the same training data, to check model variance. The table below contains the F1-score(%) for each model, then the average and standard deviation for each amount.

| Amount of augmentation | Model 1 | Model 2 | Model 3 | Model 4 | Model 5 | Model 6 | Model 7 | Model 8 | Model 9 | Model 10 | avg | std |
| ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| 0% | 30.3 | 42.1 | 49.1 | 39.2 | 39.4 | 20.2 | 46.4 | 48.1 | 46.9 | - | 40.19 | 9.54 |
| 10% | 50 | 43.7 | 59.4 | 69.5 | 72.6 | 43.5 | 48.2 | 40.8 | 54.9 | 50.4 | 53.62 | 11.48 |
| 20% | 59.3 | 55.9 | 52.2 | 64.1 | 53.9 | 54.6 | 72.7 | 63.6 | - | - | 59.54 | 6.91 |
| 30% | 66.4 | 72.5 | 61.3 | 27.5 | 43.4 | 63.9 | 64.4 | 60.1 | 48.6 | 49.7 | 55.78 | 13.43 |
| 40% | 76.8 | 74.4 | 77.3 | 76.8 | 76.6 | 73.8 | 77.6 | 79 | 67.3 | 75.7 | 75.53 | 3.26 |
| 50% | 75.6 | 74.8 | 55.7 | 68.9 | 70.4 | 70 | 74.9 | 76.8 | 76 | - | 71.46 | 6.59 |
| 60% | 75.6 | 76.7 | 76.7 | 74.7 | 77.2 | 74.5 | 78.3 | 77.9 | 78.1 | 76.1 | 76.58 | 1.35 |
| 70% | 67.7 | 76.5 | 76.9 | 74.9 | 78.7 | 79 | 74.5 | 80.1 | 77.5 | 68.7 | 75.45 | 4.2 |
| 80% | 77 | 77.6 | 78.4 | 78.1 | 74 | 50.5 | 76.4 | 79.8 | 75.3 | - | 74.12 | 9.02 |
| 90% | 76.4 | 77.4 | 77.2 | 76.3 | 80 | 76.9 | 78.4 | 75.2 | 71.2 | 77.5 | 76.65 | 2.31 |
| 100% | 77.2 | 75.4 | 76.2 | 77.6 | 75.8 | 76.9 | 76.6 | 77.4 | 75.2 | 77.9 | 76.62 | 0.94 |

We observed a significant increase in F1-score when we increased training data using augmented examples, from 40.19% when no augmentation is used, to 76.62% when 100% of augmented data is used. However, this improvement reached a plateau with way less augmented data (around 40%), which means there may be a limit of how much augmentation can be used to improve the model.

Another interesting result from our experiments so far is that the standard deviation of those F1-scores reduced when increasing augmented data, from 9.54 when no augmentation was used, to 0.94 when adding 100% augmented examples to the training dataset. This means augmenting data may produce more uniform models.

## References

[1] Huang Z., Xu W., Yu K. Bidirectional LSTM-CRF Models for Sequence Tagging. Em: arXiv preprint arXiv:1508.01991, 2015.
[2] Panchendrarajan R., Amaresan A. Bidirectional LSTM-CRF for Named Entity Recognition. Em: Proceedings of the 32nd Pacific Asia Conference on Language, Information and Computation, pp. 531 – 540, 2018.
[3] Lee C. LSTM-CRF Models for Named Entity Recognition. Em: IEICE Transactions on Information and Systems, pp. 882 – 887, 2017.
[4] Zhang X., Zhao J., Lecun Y. Character-level Convolutional Networks for Text Classification. Em: Proceedings of the 28th International Conference on Neural Information Processing Systems, Volume 1, pp. 649 – 657, 2016.
[5] Wei J., Zou K. EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks. Em: arXiv preprint arXiv:1901.11196, 2019.
[6] Vieira, H. Recognition and Linking of Product Mentions in User-generated Contents. 127 p. Thesis (Doutorado em Informática) - Universidade Federal do Amazonas, Manaus, 2018.
[7] Li J, Sun Y, Johnson RJ, Sciaky D, Wei CH, Leaman R, Davis AP, Mattingly CJ, Wiegers TC, and Lu Z. Annotating chemicals, diseases and their interactions in biomedical literature. In: _Proceedings of the Fifth BioCreative Challenge Evaluation Workshop_, pp 173-182. Dataset Download link: https://biocreative.bioinformatics.udel.edu/media/store/files/2016/CDR_Data.zip
