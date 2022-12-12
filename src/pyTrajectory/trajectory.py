import numpy as np
import deepdish as dd

import pyTrajectory.config as config
from .instance import Instance
from .series_operations import interpolate_series


is_ipython = False
try:
    import IPython
    import io
    import matplotlib.pyplot as plt
    from .visualization import get_trajectory_range, plot_trajectory
    is_ipython = True
except ModuleNotFoundError:
    pass


class OutOfTrajectoryRange(Exception):
    pass


class Trajectory(list):

    def __init__(self, instances=None, data=None):
        super().__init__()
        if instances is not None:
            if type(instances) == np.ndarray:
                instances = instances.tolist()
            self.extend(instances)
            return
        if data is not None:
            self.load(data=data)
            return

    def init_new_trajectory(self, instances):
        return type(self)(instances)

    @property
    def data(self):
        return {key: value for (key, value) in self.items()}

    def save(self, file_name):
        dd.io.save(file_name, self.data)

    def load(self, file_name=None, data=None, condition=None):
        if file_name is not None:
            data = dd.io.load(file_name)
        assert data is not None, \
            'specify either file_name or data input arguments'
        num_instances = max([len(data[key]) for key in self.keys()
                             if key in data and data[key] is not None])
        for key in self.keys():
            if key not in data:
                data[key] = None
            if data[key] is not None:
                assert len(data[key]) == num_instances, \
                    'all data values must have the same length'
                continue
            data[key] = [None] * num_instances
        selection = np.repeat(True, num_instances)
        if condition is not None:
            selection = condition(data)
        instances = [Instance(**{k: v for k, v in zip(self.keys(), instance_data)})
                     for instance_data in zip(*[np.asarray(data[key])[selection]
                     for key in self.keys()])]
        self.__init__(instances)
        return self

    # implement some dictionary functionality

    def keys(self, exclude=None):
        exclude = exclude or []
        return [key for key in config.KEYS if key not in exclude]

    def values(self, exclude=None):
        return [self.get_values(key) for key in self.keys(exclude)]

    def items(self, exclude=None):
        return [(key, value) for key, value in zip(self.keys(exclude), self.values(exclude))]

    # implement some list functionality

    def copy(self):
        return self.init_new_trajectory(super().copy())

    def append(self, instance):
        self.reset_values()
        super().append(instance)

    def extend(self, trajectory):
        self.reset_values()
        super().extend(trajectory)

    def __add__(self, trajectory):
        self.reset_values()
        return self.init_new_trajectory(super().__add__(trajectory))

    def __iadd__(self, trajectory):
        self.reset_values()
        return self.init_new_trajectory(super().__iadd__(trajectory))

    def reset_values(self):
        for key in self.keys():
            if not hasattr(self, f'_{key}'):
                continue
            delattr(self, f'_{key}')

    def get_values(self, key):
        if key in self.keys() and hasattr(self, f'_{key}'):
            return getattr(self, f'_{key}')
        values = [np.asarray(getattr(instance, key)) for instance in self]
        is_none = np.any([value.all() is None for value in values])
        is_empty = len(self) == 0
        if is_none or is_empty:
            values = None
        else:
            values = np.array(values)
        if key in self.keys() and not hasattr(self, f'_{key}'):
            setattr(self, f'_{key}', values)
        return values

    def __getitem__(self, key):
        if type(key) == tuple:
            return [self.__getitem__(key) for key in key]
        if type(key) == str and key not in self.keys():
            raise NotImplementedError('trajectory has no key {}.'.format(key))
        if key in self.keys():
            return self.get_values(key)
        item = super().__getitem__(key)
        if type(item) == Instance:
            return item
        return self.init_new_trajectory(item)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)

    def __delitem__(self, key):
        super().__delitem__(key)

    # trajectory functionality

    def sort(self, copy=False, key='time_stamp'):
        sort_idx = np.argsort(self[key])
        instances = np.array(self)[sort_idx]
        if copy:
            return self.init_new_trajectory(instances)
        self.reset_values()
        self.__init__(instances)
        return self

    def select(self, condition, copy=True):
        instances = np.array(self)[condition]
        if copy:
            return self.init_new_trajectory(instances)
        self.reset_values()
        self.__init__(instances)
        return self

    def interpolate(self, copy=True):
        self_sorted = self.sort(copy=True)
        time_stamps = self_sorted['time_stamp']
        data_interpolated = {'time_stamp': np.arange(time_stamps.min(),
                                                     time_stamps.max() + 1)}
        for key in self.keys(exclude=['time_stamp']):
            value = self_sorted[key]
            if value is None:
                data_interpolated[key] = [None] * data_interpolated['time_stamp'].size
                continue
            data_interpolated[key] = interpolate_series(
                value, time_stamps, data_interpolated['time_stamp'])
        instances = [Instance(**{k: v for k, v in zip(self.keys(), instance_data)})
                     for instance_data in zip(*data_interpolated.values())]
        if copy:
            return self.init_new_trajectory(instances)
        self.reset_values()
        self.__init__(instances)
        return self

    def is_complete(self):
        if len(self) == 0:
            return True
        return len(self) == self[-1].time_stamp - self[0].time_stamp + 1

    def slice_window(self, start, stop, check_completeness=True):
        if start < self[0].time_stamp:
            raise OutOfTrajectoryRange
        if stop > self[-1].time_stamp:
            raise OutOfTrajectoryRange
        selection = slice(np.argwhere(self['time_stamp'] >= start).ravel()[0],
                          np.argwhere(self['time_stamp'] <= stop).ravel()[-1] + 1)
        if self[selection.start].time_stamp > start:
            selection = slice(max(0, selection.start - 1), selection.stop)
        if self[selection.stop - 1].time_stamp < stop:
            selection = slice(selection.start, min(len(self) - 1, selection.stop + 1))
        trajectory_window = self[selection]
        if not check_completeness:
            return trajectory_window
        if not trajectory_window.is_complete():
            trajectory_window = trajectory_window.interpolate()
        if len(trajectory_window) == stop - start + 1:
            return trajectory_window
        return trajectory_window.slice_window(start, stop)


if is_ipython:

    class Trajectory(Trajectory):

        def _repr_png_(self):
            buffer = io.BytesIO()
            fig = plt.figure(figsize=(2, 2), dpi=300)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.axis('off')
            ax.set_aspect('equal')
            (x_min, x_max), (y_min, y_max) = get_trajectory_range(self)
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            plot_trajectory(self, ax)
            plt.savefig(buffer, format='png', dpi='figure')
            ax.clear()
            fig.clear()
            plt.close()
            return buffer.getvalue(), {'width': 100, 'height': 100}

        def _ipython_key_completions_(self):
            return self.keys()
