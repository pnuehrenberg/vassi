# from typing import Iterable


# class MyClass:
#     def __init__(self, categories: Iterable[str], predictions: pd.DataFrame):
#         self.categories = categories
#         self.predictions = predictions

#     def _repr_html_(self):
#         html = "<span><h3>Classification result</h1></span>"
#         html += f"Categories ({len(self.categories)}): {', '.join(self.categories)}"

#         # Predictions with expandable table
#         html += "<details><summary>Predictions</summary>"

#         # Create summary table
#         summary_df = pd.DataFrame(index=self.categories)
#         summary_df[('count', "")] = self.predictions.groupby('category')['category'].count()
#         for value in ["mean_probability", "max_probability"]:
#             if value not in self.predictions.columns:
#                 continue
#             summary_df[(f"{value.replace("_", " ")}", "mean")] = self.predictions.groupby('category')[value].mean()
#             summary_df[(f"{value.replace("_", " ")}", "std")] = self.predictions.groupby('category')[value].std()

#         summary_df.columns = pd.MultiIndex.from_tuples(summary_df.columns)

#         html += summary_df._repr_html_()
#         html += "</details>"
#         return html


# pipeline stuff
# if isinstance(feature_extractor, DataFrameFeatureExtractor):
#     pipeline.set_output(transform="pandas")
# if fit_pipeline:
#     return pipeline.fit_transform(X), y
# return pipeline.transform(X), y


from typing import Generic, Iterable, TypeVar

import numpy as np
import pandas as pd

T = TypeVar("T")  # Generic type variable


class BaseProcessor(Generic[T]):
    def generate(self) -> list[T]:
        """Method to be implemented by subclasses"""
        raise NotImplementedError

    def concatenate(self, data: Iterable[T]) -> T:
        """Method to be implemented by subclasses"""
        raise NotImplementedError


class NumpyProcessor(BaseProcessor[np.ndarray]):
    def generate(self) -> list[np.ndarray]:
        return [np.array([1, 2, 3]), np.array([4, 5, 6])]

    def concatenate(self, data: Iterable[np.ndarray]) -> np.ndarray:
        return np.concatenate(list(data))


class PandasProcessor(BaseProcessor[pd.DataFrame]):
    def generate(self) -> list[pd.DataFrame]:
        return [pd.DataFrame({"A": [1, 2]}), pd.DataFrame({"A": [3, 4]})]

    def concatenate(self, data: Iterable[pd.DataFrame]) -> pd.DataFrame:
        return pd.concat(data, ignore_index=True)


def process_data[P](processor: BaseProcessor[P]) -> P:
    generated = processor.generate()  # Pyright knows generated is list[P]
    return processor.concatenate(
        generated
    )  # Pyright knows concatenate accepts Iterable[P]


# Usage
numpy_processor = NumpyProcessor()
result_np = process_data(numpy_processor)  # np.ndarray

pandas_processor = PandasProcessor()
result_pd = process_data(pandas_processor)  # pd.DataFrame
