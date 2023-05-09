import numpy as np

import pyTrajectory.config

is_ipython = False
try:
    import IPython
    import io
    import matplotlib.pyplot as plt
    from .visualization import *
    is_ipython = True
except ModuleNotFoundError:
    pass


def format_value(value):
    if value is None:
        return None
    if type(value) in [int, float]:
        return value
    value = np.asarray(value)
    if len(value.shape) == 0:
        return value.item()
    if len(value.shape) > 1 and value.shape[0] == 1:
        return value[0]
    return value


class Instance(object):

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if key not in pyTrajectory.config.cfg.trajectory_keys:
                raise KeyError
            setattr(self, key, format_value(value))
        for key in set(pyTrajectory.config.cfg.trajectory_keys) - set(kwargs.keys()):
            setattr(self, key, None)

    def __getitem__(self, key):
        if key not in pyTrajectory.config.cfg.trajectory_keys:
            raise KeyError
        return getattr(self, key)

    def __setitem__(self, key, value):
        if key not in pyTrajectory.config.cfg.trajectory_keys:
            raise KeyError
        setattr(self, key, format_value(value))
        

if is_ipython:

    class Instance(Instance):

        def _repr_png_(self):
            cfg = pyTrajectory.config.cfg
            _cfg = cfg.apply(self)
            buffer = io.BytesIO()
            fig, ax = plt.subplots(1, 1, figsize=_cfg.figure.figsize, dpi=_cfg.figure.dpi)
            xlim, ylim = get_instance_range(self)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_aspect('equal')
            ax.axis('off')
            # draw bounding box
            try:
                ax.add_patch(prepare_box(self[_cfg.key_box], **_cfg.instance.box()))
            except KeyError:
                pass
            # draw keypoints and posture
            try:
                ax.add_collection(prepare_line_segments(self[_cfg.key_keypoints_line][:, :2],
                                                        **_cfg.instance.keypoints_line()))
                ax.scatter(*self[_cfg.key_keypoints][:, :2].T, **_cfg.instance.keypoints())
            except KeyError:
                pass
            # draw category and score
            try:
                ax.text(0, 1.05,
                        f'{self[_cfg.key_category]}: {self[_cfg.key_score]:.2f}',
                        transform=ax.transAxes,
                        **_cfg.instance.label())
            except KeyError:
                pass
            fig.tight_layout()
            plt.savefig(buffer, format='png', dpi='figure')
            ax.clear()
            fig.clear()
            plt.close()
            return buffer.getvalue(), {'width': _cfg.figure.width, 'height': _cfg.figure.height}

        def _ipython_key_completions_(self):
            global cfg
            return cfg.trajectory_keys
