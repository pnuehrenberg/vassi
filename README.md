![vassi logo](docs/source/vassi_logo.svg)

# Verifiable, automated scoring of social interactions in animal groups using trajectory data


### *vassi* can help you to

- organize trajectory and posture data in datasets with groups of multiple individuals
- extract individual and dyadic spatiotemporal features to describe movement and posture
- sample behavioral datasets to train machine-learning algorithms
- post-process behavioral classification results for down-stream analyses
- interactively visualize and validate behavioral sequences

## Installation

Install the package and all dependencies from this repository:

```
pip install git+https://github.com/pnuehrenberg/vassi.git
```

Refer to the documentation for more information.

## Getting started

You can use *vassi* to implement a full behavioral scoring pipeline in Python, train a machine-learning model, and use it to predict behavioral sequences.

```python
# load training dataset
dataset_train = load_dataset("train", ...)

# configure feature extractor
extractor = FeatureExtractor().read_yaml("feature_config.yaml")

# extract samples from dataset
X, y = dataset_train.subsample(extractor, size=0.1)

# train classifier
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()
classifier.fit(X, dataset_train.encode(y))

# load test dataset and predict
dataset_test = load_dataset("test",  ...)
classification_result = predict(dataset_test, classifier, extractor)

# postprocessing
processed_result = classification_result.smooth(
    lambda result: sliding_mean(result, window_size=5)
).threshold(
    [0.1, 0.8]  # assuming two categories
)

# save for downstream behavioral analyses
processed_result.predictions.to_csv("predictions.csv")
```

You can find more examples, including the two case studies presented in our paper, in the online documentation. The respective jupyter notebooks are also available here at [examples/CALMS21](https://github.com/pnuehrenberg/vassi/tree/main/examples/CALMS21) and [examples/social_cichlids](https://github.com/pnuehrenberg/vassi/tree/main/examples/social_cichlids).

## Issues and contributions

If you encounter any issues or have suggestions for improvements, please open an issue on our [GitHub repository](https://github.com/pnuehrenberg/vassi/issues). We welcome contributions in the form of pull requests as well.

If you need help to format/import your own tracking data (or from some other software), please first have a look at the example [conversion](https://github.com/pnuehrenberg/vassi/blob/main/examples/CALMS21/conversion.py) script.  If that does not help, please open an [issue](https://github.com/pnuehrenberg/vassi/issues) and we can find a solution and add more data loading examples.

## Cite *vassi*

If you use this software in your research, please cite our preprint/paper (details will be added soon):

> *vassi*: Verifiable, automated scoring of social interactions in animal groups, P. Nuehrenberg, A. Bose, A. Jordan (2025)

## License

*vassi* is released under the [MIT License](https://github.com/pnuehrenberg/vassi/blob/main/LICENSE).
