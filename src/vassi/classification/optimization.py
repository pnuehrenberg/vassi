import os
import tempfile
from collections.abc import Callable, Mapping
from concurrent.futures import ProcessPoolExecutor
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
            Concatenate[AnnotatedDatasetClassification, ...],
            AnnotatedDatasetClassification,
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
    classifier: Classifier | type[Classifier],
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
    storage: str | optuna.storages.BaseStorage,
    random_state: int,
    num_trials: int,
    dataset: AnnotatedDataset,
    extractor: BaseExtractor[F],
    classifier: Classifier | type[Classifier],
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
        storage=storage,
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
        Classifier | type[Classifier],  # classifier,
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
        Concatenate[AnnotatedDatasetClassification, ...], AnnotatedDatasetClassification
    ] = without_postprocessing,
    scoring_function: Callable[[AnnotatedDatasetClassification], float],
    random_state: np.random.Generator | int | None = None,
    n_jobs: int = 1,
) -> optuna.study.Study:
    rank = 0
    comm = None
    try:
        from mpi4py import MPI

        if MPI.COMM_WORLD.Get_size() > 1:
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
    except ImportError:
        pass

    # sync random state as int seed
    int_seed: int
    if comm is not None:
        if rank == 0:
            rng = np.random.default_rng(random_state)
            int_seed = to_int_seed(rng)
        else:
            # will be overwritten by rank 0
            int_seed = 0
        int_seed = comm.bcast(int_seed, root=0)
    else:
        # no MPI
        rng = np.random.default_rng(random_state)
        int_seed = to_int_seed(rng)

    if comm is not None or n_jobs > 1:
        if comm is None or rank == 0:
            handle, study_name_file = tempfile.mkstemp(
                dir=".", prefix="optuna_study_", suffix=".db"
            )
            os.close(handle)
            db_path = os.path.abspath(study_name_file)
            storage = f"sqlite:///{db_path}".replace(os.sep, "/")
            study_name = os.path.basename(study_name_file).split(".")[0]
        else:
            storage = ""
            study_name = ""

        if comm is not None:
            storage = str(comm.bcast(storage, root=0))
            study_name = str(comm.bcast(study_name, root=0))

        if comm is None or rank == 0:
            _ = optuna.create_study(
                study_name=study_name,
                storage=optuna.storages.RDBStorage(url=storage),
                sampler=optuna.samplers.TPESampler(seed=int_seed),
                load_if_exists=True,
                direction="maximize",
            )

        if comm is not None:
            # wait for DB creation
            comm.Barrier()

        trials_per_process = num_trials // (n_jobs if comm is None else comm.Get_size())
        remainder = num_trials % (n_jobs if comm is None else comm.Get_size())

        if comm is not None:
            # distributed MPI case

            _optuna_worker(
                study_name=study_name,
                storage=storage,
                random_state=int_seed + rank,
                num_trials=trials_per_process + (1 if rank < remainder else 0),
                dataset=dataset,
                extractor=extractor,
                classifier=classifier,
                parameter_space=parameter_space,
                k=k,
                sampling_function=sampling_function,
                postprocessing_function=postprocessing_function,
                scoring_function=scoring_function,
            )

            # sync and return on all nodes (even if we only continue downstream on root node)
            comm.Barrier()
            return optuna.load_study(study_name=study_name, storage=storage)

        # single node with true multiprocessing case (n_jobs > 1)
        worker_num_trials = [
            trials_per_process + (1 if n < remainder else 0) for n in range(n_jobs)
        ]

        # Prepare arguments for starmap/map
        worker_args = [
            (
                study_name,
                storage,
                int_seed + worker_idx,
                count,  # trials for this worker
                dataset,
                extractor,
                classifier,
                parameter_space,
                k,
                sampling_function,
                postprocessing_function,
                scoring_function,
            )
            for worker_idx, count in enumerate(worker_num_trials)
            if count > 0
        ]

        if not worker_args:
            # num_trials was 0
            return optuna.create_study(
                storage=storage, study_name=study_name, load_if_exists=True
            )

        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            _ = pool.map(_pool_helper, worker_args)

        return optuna.load_study(study_name=study_name, storage=storage)

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
    )
