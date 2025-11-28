from collections.abc import Callable, Mapping
from functools import partial
from typing import Concatenate, Literal, TypedDict, final

import numpy as np
import optuna

from ..dataset import AnnotatedDataset
from ..features import BaseExtractor, Shaped
from ..utils import to_int_seed
from ._predict import Classifier, k_fold_predict
from ._results import AnnotatedDatasetClassification


def without_postprocessing(
    dataset_classification: AnnotatedDatasetClassification,
) -> AnnotatedDatasetClassification:
    return dataset_classification.discretize()


def _scorer(
    result: AnnotatedDatasetClassification,
    *,
    on: Literal["timestamp", "prediction", "annotation"] | None,
    foreground_only: bool,
) -> float:
    if on is None:
        scores = result.score()
    else:
        scores = result.f1_score(on=on)
    if foreground_only:
        scores = scores[list(result.foreground_categories)]
    return float(np.mean(scores))


def _make_scorer(
    on: Literal["timestamp", "prediction", "annotation"] | None = None,
    foreground_only: bool = False,
) -> Callable[[AnnotatedDatasetClassification], float]:
    return partial(_scorer, on=on, foreground_only=foreground_only)


macro_f1_all_levels = _make_scorer()
macro_f1_foreground_all_levels = _make_scorer(foreground_only=True)
macro_f1_timestamp = _make_scorer(on="timestamp")
macro_f1_foreground_timestamp = _make_scorer(on="timestamp", foreground_only=True)


@final
class KFoldExperiment:
    def __init__[F: Shaped](
        self,
        dataset: AnnotatedDataset,
        extractor: BaseExtractor[F],
        classifier: Classifier | type[Classifier],
        *,
        k: int,
        classifier_kwargs: Mapping[str, object] | None = None,
        balance_sample_weights: bool,
        sampling_function: Callable[
            Concatenate[AnnotatedDataset, BaseExtractor[F], ...], tuple[F, np.ndarray]
        ],
        sampling_function_kwargs: Mapping[str, object] | None = None,
        postprocessing_function: Callable[
            [AnnotatedDatasetClassification], AnnotatedDatasetClassification
        ] = without_postprocessing,
        postprocessing_function_kwargs: Mapping[str, object] | None = None,
        scoring_function: Callable[[AnnotatedDatasetClassification], float],
        random_state: np.random.Generator | int | None = None,
    ):
        self.dataset = dataset
        self.extractor = extractor
        if (
            not isinstance(classifier, type)
            and classifier_kwargs
            and len(classifier_kwargs) > 0
        ):
            raise ValueError(
                "classifier_kwargs can only be used when classifier is passed as its class"
            )
        self.classifier = (
            classifier(**classifier_kwargs if classifier_kwargs is not None else {})
            if isinstance(classifier, type)
            else classifier
        )
        self.k = k
        self.balance_sample_weights = balance_sample_weights
        self.sampling_function = sampling_function
        self.sampling_function_kwargs = sampling_function_kwargs
        self.postprocessing_function = postprocessing_function
        self.postprocessing_function_kwargs = postprocessing_function_kwargs
        self.scoring_function = scoring_function
        self.random_state = np.random.default_rng(random_state)

    def run(self) -> float:
        k_fold_result = k_fold_predict(
            self.dataset,
            self.extractor,
            self.classifier,
            k=self.k,
            sampling_function=self.sampling_function,
            balance_sample_weights=self.balance_sample_weights,
            random_state=self.random_state,
            **self.sampling_function_kwargs
            if self.sampling_function_kwargs is not None
            else {},
        )
        k_fold_result = self.postprocessing_function(
            k_fold_result,
            **self.postprocessing_function_kwargs
            if self.postprocessing_function_kwargs is not None
            else {},
        )
        return self.scoring_function(k_fold_result)


class Parameters(TypedDict):
    balance_sample_weights: bool
    classifier_kwargs: Mapping[str, object]
    sampling_function_kwargs: Mapping[str, object]
    postprocessing_function_kwargs: Mapping[str, object]


@final
class ParameterSpace:
    def __init__(
        self,
        balance_sample_weights: Callable[[optuna.trial.Trial], bool],
        classifier_kwargs: Mapping[str, Callable[[optuna.trial.Trial], object]]
        | None = None,
        sampling_function_kwargs: Mapping[str, Callable[[optuna.trial.Trial], object]]
        | None = None,
        postprocessing_function_kwargs: Mapping[
            str, Callable[[optuna.trial.Trial], object]
        ]
        | None = None,
    ):
        self.balance_sample_weights = balance_sample_weights
        self.classifier_kwargs = classifier_kwargs or {}
        self.sampling_function_kwargs = sampling_function_kwargs or {}
        self.postprocessing_function_kwargs = postprocessing_function_kwargs or {}

    def suggest(self, trial: optuna.trial.Trial) -> Parameters:
        balance_sample_weights = self.balance_sample_weights(trial)
        classifier_kwargs = {
            key: parameter(trial) for key, parameter in self.classifier_kwargs.items()
        }
        sampling_function_kwargs = {
            key: parameter(trial)
            for key, parameter in self.sampling_function_kwargs.items()
        }
        postprocessing_function_kwargs = {
            key: parameter(trial)
            for key, parameter in self.postprocessing_function_kwargs.items()
        }
        return Parameters(
            balance_sample_weights=balance_sample_weights,
            classifier_kwargs=classifier_kwargs,
            sampling_function_kwargs=sampling_function_kwargs,
            postprocessing_function_kwargs=postprocessing_function_kwargs,
        )


def _run_k_fold_experiment[F: Shaped](
    dataset: AnnotatedDataset,
    extractor: BaseExtractor[F],
    classifier: Classifier | type[Classifier],
    trial: optuna.trial.Trial,
    *,
    k: int,
    sampling_function: Callable[
        Concatenate[AnnotatedDataset, BaseExtractor[F], ...], tuple[F, np.ndarray]
    ],
    postprocessing_function: Callable[
        [AnnotatedDatasetClassification], AnnotatedDatasetClassification
    ] = without_postprocessing,
    scoring_function: Callable[[AnnotatedDatasetClassification], float],
    parameter_space: ParameterSpace,
    random_state: int,
) -> float:
    return KFoldExperiment(
        dataset,
        extractor,
        classifier,
        k=k,
        sampling_function=sampling_function,
        postprocessing_function=postprocessing_function,
        scoring_function=scoring_function,
        random_state=random_state + trial.number,
        **parameter_space.suggest(trial),
    ).run()


def run_optuna_hyperparameter_search[F: Shaped](
    dataset: AnnotatedDataset,
    extractor: BaseExtractor[F],
    classifier: Classifier | type[Classifier],
    parameter_space: ParameterSpace,
    *,
    num_trials: int,
    k: int,
    sampling_function: Callable[
        Concatenate[AnnotatedDataset, BaseExtractor[F], ...], tuple[F, np.ndarray]
    ],
    postprocessing_function: Callable[
        [AnnotatedDatasetClassification], AnnotatedDatasetClassification
    ] = without_postprocessing,
    scoring_function: Callable[[AnnotatedDatasetClassification], float],
    random_state: np.random.Generator | int | None = None,
) -> optuna.study.Study:
    random_state = np.random.default_rng(random_state)
    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(seed=to_int_seed(random_state)),
        direction="maximize",
    )
    study.optimize(
        partial(
            _run_k_fold_experiment,
            dataset,
            extractor,
            classifier,
            k=k,
            sampling_function=sampling_function,
            postprocessing_function=postprocessing_function,
            scoring_function=scoring_function,
            random_state=to_int_seed(random_state),
            parameter_space=parameter_space,
        ),
        n_trials=num_trials,
    )
    return study
