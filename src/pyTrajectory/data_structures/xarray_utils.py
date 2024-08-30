# from typing import Any, Mapping, Optional

# import numpy as np
# import xarray as xr
# from numpy.dtypes import StringDType  # type: ignore
# from numpy.typing import NDArray

# from .. import config


# def to_xarray_dataset(
#     data: Mapping[str, NDArray],
#     cfg: config.Config,
# ) -> xr.Dataset:
#     timestamps = None
#     if cfg.key_timestamp is not None and cfg.key_timestamp in data:
#         timestamps = data[cfg.key_timestamp]
#         if timestamps.ndim != 1:
#             raise ValueError("timestamps are not one-dimensional")
#     identities = None
#     if cfg.key_identity is not None and cfg.key_identity in data:
#         identities = data[cfg.key_identity]
#         if identities.ndim != 1:
#             raise ValueError("identities are not one-dimensional")
#         if not np.issubdtype(identities.dtype, np.integer) and not isinstance(
#             identities.dtype, StringDType
#         ):
#             raise ValueError(
#                 "identities are not of integer of numpy.dtypes.StringDType type"
#             )
#     data_vars = {}
#     for key, value in data.items():
#         data_vars[key] = xr.DataArray(value)
#         if key in [cfg.key_timestamp, cfg.key_identity]:
#             data_vars[f"index_{key}"] = xr.DataArray(value)
#     attrs: dict[str, Any] = {
#         "cfg": cfg,
#         "values_as_index": {
#             cfg.key_timestamp: None,
#             cfg.key_identity: None,
#         },
#         "multiindex": None,
#     }
#     if timestamps is not None:
#         attrs["values_as_index"][cfg.key_timestamp] = f"index_{cfg.key_timestamp}"
#     if identities is not None:
#         attrs["values_as_index"][cfg.key_identity] = f"index_{cfg.key_identity}"
#     multiindex = [
#         name for name in attrs["values_as_index"].values() if name is not None
#     ]
#     if len(multiindex) > 1:
#         attrs["multiindex"] = tuple(multiindex)
#     return (
#         xr.Dataset(
#             data_vars=data_vars,
#             attrs=attrs,
#         )
#         .set_coords(multiindex)
#         .set_xindex(multiindex)
#     )


# def to_data(dataset: xr.Dataset) -> dict[str, NDArray]:
#     dataset = dataset.drop_vars(dataset.coords)
#     single_entry = any([dataset[var].ndim == 0 for var in dataset.data_vars])
#     data = {}
#     for key in dataset.data_vars:
#         value = dataset[key].to_numpy()
#         if single_entry:
#             value = value[np.newaxis, ...]
#         data[str(key)] = value
#     return data


# def update_index(
#     dataset: xr.Dataset,
#     *,
#     key: Optional[str] = None,
# ) -> xr.Dataset:
#     cfg = dataset.cfg
#     if key is not None and key not in [cfg.key_timestamp, cfg.key_identity]:
#         # no need to update index
#         return dataset
#     if (multiindex := dataset.multiindex) is not None:
#         dataset = dataset.reset_index(list(dataset.indexes.keys()))
#     if (index_timestamp := dataset.values_as_index[cfg.key_timestamp]) is not None:
#         dataset[index_timestamp] = dataset[cfg.key_timestamp]
#     if (index_identity := dataset.values_as_index[cfg.key_identity]) is not None:
#         dataset[index_identity] = dataset[cfg.key_identity]
#     if multiindex is not None:
#         dataset = dataset.set_xindex(list(multiindex))
#     return dataset
