# from abc import ABC, abstractmethod

# import numpy as np
# import pandas as pd
#

# from .utils import (
#     EncodingFunction,
#     to_predictions,
# )


# class ResultMixin(ABC):
#     categories: tuple[str, ...]
#     background_category: str

#     @property
#     @abstractmethod
#     def timestamps(self) -> np.ndarray[np.int64]: ...

#     @property
#     @abstractmethod
#     def predictions(self) -> pd.DataFrame: ...

#     @property
#     @abstractmethod
#     def y_proba(self) -> np.ndarray[np.float64]: ...

#     @property
#     @abstractmethod
#     def y_pred(self) -> np.ndarray[np.int64]: ...


# class AnnotatedResultMixin(ResultMixin):
#     @abstractmethod
#     def encode(self, y: np.ndarray) -> np.ndarray[np.int64]: ...

#     @property
#     @abstractmethod
#     def annotations(self) -> pd.DataFrame: ...

#     @property
#     @abstractmethod
#     def y_true(self) -> np.ndarray[np.integer]: ...

#     # def score_category_counts(self) -> np.ndarray:
#     #     return score_category_counts(
#     #         self.annotations,
#     #         self.predictions,
#     #         self.categories,
#     #     )

#     # def f1_score(
#     #     self,
#     #     on: Literal["timestamp", "annotation", "prediction"],
#     #     *,
#     #     average: Optional[Literal["micro", "macro", "weighted"]] = None,
#     #     encode_func: Callable[[np.ndarray], np.ndarray[np.integer]],
#     # ) -> float | tuple[float, ...]:
#     #     categories: tuple[str, ...] = tuple(self.categories)  # type: ignore
#     #     if on == "timestamp":
#     #         y_true = self.y_true
#     #         y_pred = self.y_pred
#     #     elif on == "annotation":
#     #         y_true = encode_func(self.annotations["category"].to_numpy())
#     #         y_pred = encode_func(self.annotations["predicted_category"].to_numpy())
#     #     elif on == "prediction":
#     #         y_true = encode_func(self.predictions["true_category"].to_numpy())
#     #         y_pred = encode_func(self.predictions["category"].to_numpy())
#     #     else:
#     #         raise ValueError(
#     #             f"'on' should be one of 'timestamp', 'annotation', 'prediction' and not '{per}'"
#     #         )
#     #     return f1_score(
#     #         y_true,
#     #         y_pred,
#     #         labels=range(len(categories)),
#     #         average=average,  # type: ignore
#     #         zero_division=np.nan,  # type: ignore
#     #     )

#     # @overload
#     # def score(
#     #     self,
#     #     encode_func: Callable[[np.ndarray], np.ndarray[np.integer]],
#     #     *,
#     #     macro: Literal[False] = False,
#     # ) -> Mapping[str, np.ndarray]: ...

#     # @overload
#     # def score(
#     #     self,
#     #     encode_func: Callable[[np.ndarray], np.ndarray[np.integer]],
#     #     *,
#     #     macro: Literal[True],
#     # ) -> Mapping[str, float]: ...

#     # def score(
#     #     self,
#     #     encode_func: Callable[[np.ndarray], np.ndarray[np.integer]],
#     #     *,
#     #     macro: bool = False,
#     # ) -> Mapping[str, np.ndarray | float]:
#     #     category_count_scores = self.score_category_counts()
#     #     f1_per_timestamp = self.f1_score("timestamp", encode_func=encode_func)
#     #     f1_per_annotation = self.f1_score("annotation", encode_func=encode_func)
#     #     f1_per_prediction = self.f1_score("prediction", encode_func=encode_func)
#     #     scores = {
#     #         score_name: np.asarray(values)
#     #         for score_name, values in zip(
#     #             [
#     #                 "category_count_score",
#     #                 "f1_per_timestamp",
#     #                 "f1_per_annotation",
#     #                 "f1_per_prediction",
#     #             ],
#     #             [
#     #                 category_count_scores,
#     #                 f1_per_timestamp,
#     #                 f1_per_annotation,
#     #                 f1_per_prediction,
#     #             ],
#     #         )
#     #     }
#     #     if macro:
#     #         return {
#     #             score_name: float(values.mean())
#     #             for score_name, values in scores.items()
#     #         }
#     #     return scores


# class BaseClassificationResult(ResultMixin):
#     def __init__(
#         self,
#         y_proba: np.ndarray[np.float64],
#         *,
#         timestamps: np.ndarray[np.int64],
#         categories: tuple[str, ...],
#         background_category: str,
#         encode_func: EncodingFunction,
#     ):
#         self.categories = categories
#         self.background_category = background_category
#         self._timestamps = timestamps
#         self._y_pred: np.ndarray[np.int64] = np.argmax(y_proba, axis=1)
#         self._y_proba = y_proba
#         self._encode = encode_func
#         self._predictions = to_predictions(
#             self.y_pred,
#             self.y_proba,
#             self.categories,
#             self.timestamps,
#         )

#     @property
#     def timestamps(self) -> np.ndarray[np.int64]:
#         return self._timestamps

#     @property
#     def y_pred(self) -> np.ndarray[np.int64]:
#         return self._y_pred

#     @property
#     def y_proba(self) -> np.ndarray[np.float64]:
#         return self._y_proba

#     @property
#     def predictions(self) -> pd.DataFrame:
#         return self._predictions

#     def encode(self, y: np.ndarray) -> np.ndarray[np.int64]:
#         return self._encode(y)


# class AnnotatedBaseClassificationResult(BaseClassificationResult, AnnotatedResultMixin):
#     def __init__(
#         self,
#         y_proba: np.ndarray[np.float64],
#         *,
#         timestamps: np.ndarray[np.int64],
#         categories: tuple[str, ...],
#         background_category: str,
#         encode_func: EncodingFunction,
#         y_true: np.ndarray,  # as labels
#         annotations: pd.DataFrame,
#     ):
#         BaseClassificationResult.__init__(
#             self,
#             y_proba,
#             timestamps=timestamps,
#             categories=categories,
#             background_category=background_category,
#             encode_func=encode_func,
#         )
#         self._y_true = self.encode(y_true)
#         self._annotations = annotations

#     @property
#     def y_true(self) -> np.ndarray[np.int64]:
#         return self._y_true

#     @property
#     def annotations(self) -> pd.DataFrame:
#         return self._annotations


# BaseClassificationResult(
#     np.zeros((100, 4)),
#     timestamps=np.arange(100, dtype=int),
#     categories=("a", "b", "c", "d"),
#     background_category="c",
#     encode_func=lambda y: np.repeat("c", len(y)),
# )


# AnnotatedBaseClassificationResult(
#     np.zeros((100, 4)),
#     timestamps=np.arange(100, dtype=int),
#     categories=("a", "b", "c", "d"),
#     background_category="c",
#     encode_func=lambda y: np.repeat(3, len(y)),
#     y_true=np.repeat("c", 100),
#     annotations=pd.DataFrame(
#         {
#             "start": [0],
#             "stop": [99],
#             "category": ["c"],
#         }
#     ),
# )


# # TODO
