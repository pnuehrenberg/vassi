import numpy as np

import pyTrajectory.config
import pyTrajectory.series_operations
import pyTrajectory.instance

from multimethod import multimethod

from typing import TypeVar, Self, Any
from numpy.typing import ArrayLike
from math import isclose
from copy import deepcopy


class OutOfTrajectoryRange(Exception):
	# TODO rename to OutOfTimeRange
	pass


Value = ArrayLike | None
Data = dict[str, Value]
Keys = list[str] | tuple[str, ...]
TrajectoryType = TypeVar('Trajectory')
IntType = int | np.integer
FloatType = float | np.floating

class Trajectory:

	def __init__(
			self,
			data: Data | None = None,
			cfg: pyTrajectory.config.Config | None = None) -> None:
		self._cfg = cfg
		self._data = {key: None for key in self.keys()}
		if data is not None:
			self.data = data

	@property
	def cfg(self) -> pyTrajectory.config.Config:
		if self._cfg is None:
			return pyTrajectory.config.cfg
		return self._cfg

	@cfg.setter
	def cfg(self, cfg: pyTrajectory.config.Config | None):
		self._cfg = cfg

	def keys(self, exclude: list[str] | None = None) -> list[str]:
		if exclude is None:
			exclude = []
		return [key for key in self.cfg.trajectory_keys if key not in exclude]

	def values(
			self,
			exclude: list[str] | None = None,
			copy: bool = True) -> list[Value]:
		return [self._get_value(key, copy=copy) for key in self.keys(exclude=exclude)]

	def items(
			self,
			exclude: list[str] | None = None,
			copy: bool = True) -> list[tuple[str, Value]]:
		keys = self.keys(exclude=exclude)
		values = self.values(exclude=exclude, copy=copy)
		return [(key, value) for key, value in zip(keys, values)]

	@property
	def length(self) -> int:
		length = 0
		for key in self.cfg.trajectory_keys:
			value = self._get_value(key)
			if value is None:
				continue
			return len(value)
		return length

	def __len__(self) -> int:
		return self.length

	def verify_data(self, data: Data) -> None:
		if data is None:
			return
		complete_keys = set(data.keys()) == set(self.keys())
		lengths = set()
		for key, value in data.items():
			if value is None:
				continue
			data[key] = np.asarray(value)
			lengths.add(len(data[key]))
		if not len(lengths) == 1:
			raise ValueError(f'non-None values mave mismatched lengths')
		timestamps = None
		try:
			timestamps = data[self.cfg.key_timestamp]
			if timestamps is not None:
				if timestamps.ndim != 1:
					raise ValueError(f'data timestamps is not 1-dimensional (key: {self.cfg.key_timestamp})')
				counts = np.unique(timestamps, return_counts=True)[1]
				if np.any(counts > 1):
					raise ValueError(f'data contains duplicated timestamps (key: {self.cfg.key_timestamp})')
		except KeyError:
			pass
		length = self.length
		for key, value in data.items():
			if value is None:
				continue
			if not complete_keys and length > 0 and len(value) != length:
				raise ValueError(f'could not set value (key: {key}) with length: {len(value)} on {type(self)} with length: {length}')

	@property
	def data(self) -> Data:
		return {key: value for key, value in self.items()}

	@data.setter
	def data(self, data: Data) -> None:
		if not set(self.keys()) == set(data.keys()):
			raise KeyError('could not set data due to mismatched keys')
		self.verify_data(data)
		self._data = {key: None for key in self.keys()}
		if data is None:
			return
		for key, value in data.items():
			self._set_value(key, value)

	def _set_value(self, key: str, value: Value, verify: bool = True) -> None:
		if verify:
			self.verify_data({key: value})
		if key not in self.keys():
			raise KeyError(f'could not set value for undefined key: {key}')
		if value is not None:
			value = np.asarray(value)
		if not value.flags.owndata:
			value = value.copy()
		self._data[key] = value
		self._data[key].flags.writeable = False

	def _set_value_slice(self, slice_key: slice, key: str, value: Value, verify: bool = True) -> None:
		is_value = value is not None
		has_value = self._get_value(key) is not None
		if is_value is not has_value:
			raise ValueError('value should match slice (either None or appropriate array-like)')
		data = self.data
		data[key][slice_key] = value
		if verify:
			self.verify_data(data)
		if not value.flags.owndata:
			value = value.copy()
		self._data[key].flags.writeable = True
		self._data[key][slice_key] = value
		self._data[key].flags.writeable = False

	def _get_value(self, key: str, copy: bool = False) -> Value:
		value = self._data[key]
		if value is None:
			return value
		if copy:
			return value.copy()
		return value

	def _get_value_slice(self, slice_key: slice, key: str, copy: bool = False) -> Value:
		value = self._get_value(key)
		if value is None:
			return value
		if copy:
			value[slice_key].copy()
		return value[slice_key]

	@multimethod
	def __getitem__(self, key: str) -> Value:
		return self._get_value(key)

	@__getitem__.register
	def _get_multiple_keys(self, key: Keys) -> list[Value]:
		return [self._get_value(key) for key in key]

	@__getitem__.register
	def _get_slice(self, key: slice) -> Self:
		return type(self)(data={_key: self._get_value_slice(key, _key) for _key in self.keys()})

	@__getitem__.register
	def _get_slice_single_key(self, key: tuple[slice, 'str']) -> Value:
		return self._get_value_slice(*key)

	@__getitem__.register
	def _get_slice_multiple_keys(self, key: tuple[slice, Keys]) -> Data:
		return {_key: self._get_value_slice(key[0], _key) for _key in key[1]}

	@__getitem__.register
	def _get_instance_at(self, key: IntType) -> pyTrajectory.instance.Instance:
		instance_data = {_key: (deepcopy(value[key]) if value is not None else None)
						 for _key, value in self.items(copy=False)}
		return pyTrajectory.instance.Instance(**instance_data)

	@multimethod
	def __setitem__(self, key: str, value: Value) -> None:
		self._set_value(key, value)

	@__setitem__.register
	def _set_multiple_keys(self, key: Keys, value: Value) -> None:
		self.verify_data({_key: value for _key in key})
		for _key in key:
			self._set_value(_key, value, verify=False)

	@__setitem__.register
	def _set_multiple_keys_multiple_values(self, key: Keys, value: list[Value]) -> None:
		self.verify_data({_key: _value for _key, _value in zip(key, value)})
		for _key, _value in zip(key, value):
			self._set_value(_key, _value, verify=False)

	@__setitem__.register
	def _set_slice(self, key: slice, value: TrajectoryType) -> None:
		if not set(self.keys()) == set(value.keys()):
			raise KeyError('could not set data due to mismatched keys')
		data = self.data
		for _key in value.keys():
			data[_key][key] = value[_key]
		self.verify_data(data)
		for _key in value.keys():
			self._set_value_slice(key, _key, value[_key], verify=False)

	@__setitem__.register
	def _set_slice_single_key(self, key: tuple[slice, str], value: Value) -> None:
		data = {key[1]: self._get_value(key[1], copy=True)}
		data[key[1]][key[0]] = value
		self.verify_data(data)
		self._set_value_slice(*key, value, verify=False)

	@__setitem__.register
	def _set_slice_multiple_keys(self, key: tuple[slice, Keys], value: Value) -> None:
		data = {_key: self._get_value(key[1], copy=True) for _key in key[1]}
		for _key in key[1]:
			data[_key][key[0]] = value
		self.verify_data(data)
		for _key in key[1]:
			self._set_value_slice(key[0], _key, value, verify=False)

	@__setitem__.register
	def _set_slice_multiple_keys_multiple_values(self, key: tuple[slice, Keys], value: list[Value]) -> None:
		data = {_key: self._get_value(key[1], copy=True) for _key in key[1]}
		for _key, _value in zip(key[1], value):
			data[_key][key[0]] = _value
		self.verify_data(data)
		for _key, _value in zip(key[1], value):
			self._set_value_slice(key[0], _key, _value, verify=False)

	@property
	def is_sorted(self) -> bool:
		return (np.diff(self[self.cfg.key_timestamp]) > 0).all()

	@property
	def is_complete(self) -> bool:
		timestamps = self[self.cfg.key_timestamp]
		first = timestamps.min()
		last = timestamps.max()
		step = np.diff(timestamps).min()
		length = 1 + (last - first) / step
		return isclose(self.length, length)

	def sort(self, copy: bool = True) -> TrajectoryType:
		if self.is_sorted:
			if not copy:
				return self
			return type(self)(data=self.data)
		sort_idx = np.argsort(self[self.cfg.key_timestamp])
		data = {key: value[sort_idx] for key, value in self.items()}
		if not copy:
			self.data = data
			return self
		return type(self)(data=data)

	def sample(self, timestamps: Value, copy: bool = True) -> TrajectoryType:
		timestamps = np.asarray(timestamps)
		if not self.is_sorted:
			raise AssertionError(f'can only sample sorted {type(self)}')
		data = {key: pyTrajectory.series_operations.sample_series(
					self[key], self[self.cfg.key_timestamp], timestamps)
				for key in self.keys(exclude=[self.cfg.key_timestamp])}
		data[self.cfg.key_timestamp] = timestamps
		if not copy:
			self.data = data
			return self
		return type(self)(data=data)

	def interpolate(self, step: IntType | float | None = None, copy: bool = True) -> TrajectoryType:
		timestamps = self[self.cfg.key_timestamp]
		first = timestamps.min()
		last = timestamps.max()
		if step is None:
			step = np.diff(timestamps).min()
		length = 1 + (last - first) / step
		if not isclose(length, np.round(length)):
			raise ValueError(f'steps should result in an integer {type(self)} length and not: {length}')
		timestamps = np.linspace(first, last, int(np.round(length)))
		if isclose(step, 1) and isclose(first, np.round(first)):
			timestamps = np.round(timestamps).astype(int) 
		return self.sample(timestamps, copy=copy)

	def slice_window(self, start: IntType | float, stop: IntType | float, interpolate: bool = True) -> TrajectoryType:
		key_timestamp = self.cfg.key_timestamp
		timestamps = self[self.cfg.key_timestamp]
		first = timestamps.min()
		last = timestamps.max()
		if interpolate and start < first:
			raise OutOfTrajectoryRange(f'start: {start} not in trajectory range: [{first} {last}]')
		if interpolate and stop > last:
			raise OutOfTrajectoryRange(f'stop: {stop} not in trajectory range: [{first} {last}]')
		try:
			slice_key = slice(np.argwhere(timestamps >= start).ravel()[0],
							  np.argwhere(timestamps <= stop).ravel()[-1] + 1)
		except IndexError:
			raise OutOfTrajectoryRange(f'slice: {slice_key} not in trajectory range: [{first} {last}]')
		if not interpolate:
			return self[slice_key]
		if self[slice_key.start][key_timestamp] > start:
			slice_key = slice(max(0, slice_key.start - 1), slice_key.stop)
		if self[slice_key.stop - 1][key_timestamp] < stop:
			slice_key = slice(slice_key.start, min(len(self), slice_key.stop + 1))
		trajectory_window = type(self)(data=self[slice_key].data)
		if not trajectory_window.is_complete:
			trajectory_window = trajectory_window.interpolate()
		if isclose(trajectory_window[0][key_timestamp], start) and isclose(trajectory_window[-1][key_timestamp], stop):
			return trajectory_window
		return trajectory_window.slice_window(start, stop)



