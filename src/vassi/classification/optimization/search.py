import os
import tempfile
from collections.abc import Callable, Mapping
from functools import partial
from typing import Concatenate, final

import numpy as np
import optuna
import optuna.storages.journal
import pandas as pd

from ...dataset import AnnotatedDataset
from ...distributed import Environment, get_process_state, limited_process_pool
from ...features import BaseExtractor, Shaped
from ...io.yaml import to_yaml
from ...utils import to_int_seed
from .._predict import Classifier, k_fold_predict
from .._results import AnnotatedDatasetClassification
from .utils import ParameterSpace, without_postprocessing


@final
class KFoldExperiment:
    def __init__[F: Shaped](
        self,
        dataset: AnnotatedDataset,
        extractor: BaseExtractor[F],
        classifier: type[Classifier],
        *,
        k: int,
        classifier_kwargs: Mapping[str, object] | None = None,
        balance_sample_weights: bool,
        sampling_function: Callable[
            Concatenate[AnnotatedDataset, BaseExtractor[F], ...], tuple[F, np.ndarray]
        ],
        sampling_function_kwargs: Mapping[str, object] | None = None,
        postprocessing_function: Callable[
            Concatenate[AnnotatedDatasetClassification, ...],
            AnnotatedDatasetClassification,
        ] = without_postprocessing,
        postprocessing_function_kwargs: Mapping[str, object] | None = None,
        scoring_function: Callable[[AnnotatedDatasetClassification], float],
        random_state: np.random.Generator | int | None = None,
    ):
        self.dataset = dataset
        self.extractor = extractor
        self.classifier = classifier(
            **classifier_kwargs if classifier_kwargs is not None else {}
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
        with get_process_state():
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


def _run_k_fold_experiment[F: Shaped](
    dataset: AnnotatedDataset,
    extractor: BaseExtractor[F],
    classifier: type[Classifier],
    trial: optuna.trial.Trial,
    *,
    k: int,
    sampling_function: Callable[
        Concatenate[AnnotatedDataset, BaseExtractor[F], ...], tuple[F, np.ndarray]
    ],
    postprocessing_function: Callable[
        Concatenate[AnnotatedDatasetClassification, ...], AnnotatedDatasetClassification
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


def _run_optuna_hyperparameter_search_linear[F: Shaped](
    dataset: AnnotatedDataset,
    extractor: BaseExtractor[F],
    classifier: type[Classifier],
    parameter_space: ParameterSpace,
    *,
    num_trials: int,
    k: int,
    sampling_function: Callable[
        Concatenate[AnnotatedDataset, BaseExtractor[F], ...], tuple[F, np.ndarray]
    ],
    postprocessing_function: Callable[
        Concatenate[AnnotatedDatasetClassification, ...], AnnotatedDatasetClassification
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


def _optuna_worker[F: Shaped](
    study_name: str,
    storage: str,
    random_state: int,
    num_trials: int,
    dataset: AnnotatedDataset,
    extractor: BaseExtractor[F],
    classifier: type[Classifier],
    parameter_space: ParameterSpace,
    k: int,
    sampling_function: Callable[
        Concatenate[AnnotatedDataset, BaseExtractor[F], ...], tuple[F, np.ndarray]
    ],
    postprocessing_function: Callable[
        Concatenate[AnnotatedDatasetClassification, ...], AnnotatedDatasetClassification
    ],
    scoring_function: Callable[[AnnotatedDatasetClassification], float],
) -> None:
    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(
        study_name=study_name,
        storage=optuna.storages.JournalStorage(
            optuna.storages.journal.JournalFileBackend(storage)
        ),
        sampler=sampler,
        direction="maximize",
        load_if_exists=True,
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
            random_state=random_state,
            parameter_space=parameter_space,
        ),
        n_trials=num_trials,
    )


def _pool_helper[F: Shaped](
    worker_args: tuple[
        str,  # study_name,
        str,  # storage,
        int,  # int_seed,
        int,  # num trials for this worker
        AnnotatedDataset,  # dataset,
        BaseExtractor[F],  # extractor,
        type[Classifier],  # classifier,
        ParameterSpace,  # parameter_space,
        int,  # k,
        Callable[
            Concatenate[AnnotatedDataset, BaseExtractor[F], ...], tuple[F, np.ndarray]
        ],  # sampling_function,
        Callable[
            Concatenate[AnnotatedDatasetClassification, ...],
            AnnotatedDatasetClassification,
        ],  # postprocessing_function,
        Callable[[AnnotatedDatasetClassification], float],  # scoring_function,
    ],
) -> None:
    return _optuna_worker(*worker_args)


def _run_optuna_hyperparameter_search_parallel[F: Shaped](
    n_jobs: int,
    num_trials_node: int,
    int_seed_node: int,
    study_name: str,
    storage: str,
    dataset: AnnotatedDataset,
    extractor: BaseExtractor[F],
    classifier: type[Classifier],
    parameter_space: ParameterSpace,
    k: int,
    sampling_function: Callable[
        Concatenate[AnnotatedDataset, BaseExtractor[F], ...], tuple[F, np.ndarray]
    ],
    postprocessing_function: Callable[
        Concatenate[AnnotatedDatasetClassification, ...],
        AnnotatedDatasetClassification,
    ],
    scoring_function: Callable[[AnnotatedDatasetClassification], float],
) -> None:
    trials_per_process = num_trials_node // n_jobs
    remainder = num_trials_node % n_jobs

    num_trials_workers = [
        trials_per_process + (1 if n < remainder else 0) for n in range(n_jobs)
    ]

    worker_args = [
        (
            study_name,
            storage,
            int_seed_node + worker_idx,
            count,
            dataset,
            extractor,
            classifier,
            parameter_space,
            k,
            sampling_function,
            postprocessing_function,
            scoring_function,
        )
        for worker_idx, count in enumerate(num_trials_workers)
        if count > 0
    ]

    if not worker_args:
        return

    with limited_process_pool(n_jobs) as pool:
        _ = pool.map(_pool_helper, worker_args)


def run_optuna_hyperparameter_search[F: Shaped](
    dataset: AnnotatedDataset,
    extractor: BaseExtractor[F],
    classifier: type[Classifier],
    parameter_space: ParameterSpace,
    *,
    num_trials: int,
    k: int,
    sampling_function: Callable[
        Concatenate[AnnotatedDataset, BaseExtractor[F], ...], tuple[F, np.ndarray]
    ],
    postprocessing_function: Callable[
        Concatenate[AnnotatedDatasetClassification, ...], AnnotatedDatasetClassification
    ] = without_postprocessing,
    scoring_function: Callable[[AnnotatedDatasetClassification], float],
    random_state: np.random.Generator | int | None = None,
    n_jobs: int = 1,
) -> optuna.study.Study:
    env = Environment()

    if env.size == 1 and n_jobs == 1:
        return _run_optuna_hyperparameter_search_linear(
            dataset,
            extractor,
            classifier,
            parameter_space,
            num_trials=num_trials,
            k=k,
            sampling_function=sampling_function,
            postprocessing_function=postprocessing_function,
            scoring_function=scoring_function,
            random_state=random_state,
        )

    int_seed = 0
    if env.is_root:
        rng = np.random.default_rng(random_state)
        int_seed = to_int_seed(rng)
    int_seed = env.bcast(int_seed)

    storage = ""
    study_name = ""
    if env.is_root:
        handle, study_name_file = tempfile.mkstemp(
            dir=".", prefix="optuna_study_", suffix=".log"
        )
        os.close(handle)
        storage = os.path.abspath(study_name_file)
        study_name = os.path.basename(study_name_file).split(".")[0]
    storage = env.bcast(storage)
    study_name = env.bcast(study_name)
    if env.is_root:
        _ = optuna.create_study(
            study_name=study_name,
            storage=optuna.storages.JournalStorage(
                optuna.storages.journal.JournalFileBackend(storage)
            ),
            sampler=optuna.samplers.TPESampler(seed=int_seed),
            load_if_exists=True,
            direction="maximize",
        )
    env.barrier()

    trials_for_node = num_trials // env.size
    remainder = num_trials % env.size

    num_trial_current_node = trials_for_node + (1 if env.rank < remainder else 0)
    int_seed_current_node = int_seed + (env.rank * n_jobs)

    if num_trial_current_node > 0:
        if n_jobs > 1:
            # hybrid parallelism (MPI node + multiprocessing)
            _run_optuna_hyperparameter_search_parallel(
                n_jobs=n_jobs,
                num_trials_node=num_trial_current_node,
                int_seed_node=int_seed_current_node,
                study_name=study_name,
                storage=storage,
                dataset=dataset,
                extractor=extractor,
                classifier=classifier,
                parameter_space=parameter_space,
                k=k,
                sampling_function=sampling_function,
                postprocessing_function=postprocessing_function,
                scoring_function=scoring_function,
            )
        else:
            # MPI-only Parallelism (single process per node)
            _optuna_worker(
                study_name=study_name,
                storage=storage,
                random_state=int_seed_current_node,
                num_trials=num_trial_current_node,
                dataset=dataset,
                extractor=extractor,
                classifier=classifier,
                parameter_space=parameter_space,
                k=k,
                sampling_function=sampling_function,
                postprocessing_function=postprocessing_function,
                scoring_function=scoring_function,
            )

    env.barrier()

    return optuna.load_study(
        study_name=study_name,
        storage=optuna.storages.JournalStorage(
            optuna.storages.journal.JournalFileBackend(storage)
        ),
    )


def summarize_study(
    study: optuna.study.Study,
    *,
    summary_file: str = "optimization-summary.yaml",
    trials_file: str = "optimization-trials.csv",
) -> dict[str, object]:
    trials = pd.DataFrame(
        [
            {
                **{f"param_{param}": value for param, value in trial.params.items()},
                "value": trial.value,
            }
            for trial in study.trials
        ]
    )
    trials.to_csv(trials_file, index=False)
    to_yaml(
        {
            **{f"param_{param}": value for param, value in study.best_params.items()},
            "trial": study.best_trial.number,
        },
        file_name=summary_file,
    )
    return {f"param_{param}": value for param, value in study.best_params.items()}
