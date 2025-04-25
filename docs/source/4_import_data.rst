Import your own data
====================

The *automated-scoring* package is implemented in a object-oriented style, which allows for easy integration with your own data.
To run the entire pipeline, you need two primary data sources:

1. Tracking data of individual animals
2. Example behavioral data of interacting animals

1. Loading trajectories
-----------------------

The tracking data can originate from various sources, for example from software designated for animal tracking such as *DeepLabCut*, *sleap.ai*, *deepposekit* (list not exhaustive) or your own custom tracking solution.
In any case, you will have collected posture data for each animal and video frame the animal is visible in. With the *automated-scoring* package, you can easily import and process this data by arranging it into a numpy array and passing it to the :code:`Trajectory` class.

.. jupyter-execute::

    from automated_scoring.config import cfg
    from automated_scoring.data_structures import Trajectory

    # configure trajectory objects, providing your names for the collected data
    cfg.trajectory_keys = ("time", "posture")

    # assign them to the following two, minimally required configuration keys
    cfg.key_timestamp = "time"
    cfg.key_keypoints = "posture"

Assuming you have tracked an animal using four keypoints (e.g., nose, left shoulder, right shoulder, tail tip) in video coordinates (i.e., pixels: x, y), you need to prepare a numpy array with the shape :code:`(num_frames, num_keypoints, 2)`, where :code:`num_frames` is the number of frames the animal was tracked in, :code:`num_keypoints` is the number of keypoints tracked, and :code:`2` represents the x and y coordinates of each keypoint.

.. jupyter-execute::

    import numpy as np

    # random example data
    num_frames = 100
    num_keypoints = 4
    shape = (num_frames, num_keypoints, 2)

    # create a random number generator
    rng = np.random.default_rng()

    time = np.arange(num_frames)
    posture = rng.random(shape)

    # create a Trajectory object
    trajectory = Trajectory(
        data={
            "time": time,
            "posture": posture
        }
    )

    # alternatively, with the configuration
    trajectory = Trajectory(
        data={
            cfg.key_timestamp: time,
            cfg.key_keypoints: posture
        }
    )

    print(trajectory.is_sorted, trajectory.is_complete)

Your data does not need to be sorted or complete, the :code:`Trajectory` class provides methods to sort and interpolate missing data.

.. jupyter-execute::

    # create a trajectory with incomplete and unordered timestamps
    time = rng.choice(np.arange(200), num_frames, replace=False)

    trajectory = Trajectory(
        data={
            cfg.key_timestamp: time,
            cfg.key_keypoints: posture
        }
    )

    print(trajectory.is_sorted, trajectory.is_complete)

Now, you can use the :code:`sort` method to sort the trajectory by timestamps and :code:`interpolate` to fill missing data. The :code:`timestep` property is inferred from the input data as the greatest common divisor of the time differences between consecutive timestamps.

.. jupyter-execute::

    trajectory = trajectory.sort()
    print(trajectory.is_sorted, trajectory.is_complete)

    trajectory = trajectory.interpolate()
    print(trajectory.is_sorted, trajectory.is_complete, trajectory.timestep)

.. hint::
    These methods have a :code:`copy=False` parameter to control whether a new trajectory is created or the original one is modified in place.
    Only sorted trajectories can be interpolated.

Interpolation can also be used for temporal resampling. Without providing a :code:`timestep` argument, the trajectory is resampled to its inferred :code:`timestep`. Alternatily, you can pass a :code:`timestep` parameter when initializing the :code:`Trajectory` object:

.. jupyter-execute::

    trajectory_2 = Trajectory(
        data={
            cfg.key_timestamp: time,
            cfg.key_keypoints: posture
        },
        timestep=0.5
    )

    print(trajectory_2.sort().interpolate() == trajectory.sort().interpolate(0.5))

.. hint::
    You can also set the :code:`timestep` parameter of the configuration object globally. If no configuration is passed when initializing trajectories, the global configuration from :code:`automated_scoring.config.cfg` is used.

2. Creating groups
------------------

The :code:`Trajectory` class is the fundamental data structure to hold individual trajectory data. The *automated-scoring* package provides additional classes to represent groups of multiple animals.
Depending on whether you want to score individual or social behavior (specified via the :code:`target` parameter), a :code:`Group` consists of either :code:`Individual` or :code:`Dyad` objects. Both are initialized with :code:`Trajectory` objects:

.. jupyter-execute::

    from automated_scoring.dataset import Group

    def create_random_trajectory():
        global cfg, rng, num_frames, shape
        time = rng.choice(np.arange(200), num_frames, replace=False)
        posture = rng.random(shape)
        return Trajectory(
            data={
                cfg.key_timestamp: time,
                cfg.key_keypoints: posture,
            }
        )

    animals = ["animal_1", "animal_2", "animal_3"]

    trajectories = {
        animal: create_random_trajectory().sort().interpolate()
        for animal in animals
    }

    group_1 = Group(trajectories, target="individual")
    group_2 = Group(trajectories, target="dyad")

    # note that groups can be iterated over, yielding tuples of
    # (identifier, sampleable); where sampleable is an object that implements
    # the sampling interface (methods 'sample' and 'subsample')
    print("Group 1 consists of individuals:")
    for identifier, sampleable in group_1:
        print(f"{identifier}: {type(sampleable)}")
    print("\nGroup 2 consists of dyads:")
    for identifier, sampleable in group_2:
        print(f"{identifier}: {type(sampleable)}")

.. hint::
    Groups can be initialized with a dictionary of :code:`Trajectory` objects, where the keys can be either :code:`str` or :code:`int`. Alternatively, you can pass a list of :code:`Trajectory` objects, in which case the indices are used as identifiers.
    When initializing a group, data validation is performed to ensure that all trajectories are sorted and complete, otherwise an error will be raised.

3. Adding behavioral annotations
--------------------------------

The package also implements the :code:`Dataset` class, which provides a further level of nesting to comprise multiple groups. All dataset types (:code:`Individual`, :code:`Dyad`, :code:`Group`, :code:`Dataset`) can be annotated with behavioral intervals.
These annotations can be added as :code:`pandas.DataFrame`, with different column requirements depending on the dataset type.

.. jupyter-execute::

    from automated_scoring.dataset import Individual, Dyad, Group, Dataset

    for dataset_type in [Individual, Dyad]:
        print(dataset_type)
        print(f"Adding annotations requires following columns:")
        print(dataset_type.REQUIRED_COLUMNS(), "\n")

    for target in ["individual", "dyad"]:
        for dataset_type in [Group, Dataset]:
            print(dataset_type)
            print(f"Adding annotations (target: {target}) requires following columns:")
            print(dataset_type.REQUIRED_COLUMNS(target), "\n")

Let's create some example behavioral annotations for the two groups that were initialized above. Both have the same number of animals,
but :code:`group_1` targets individual (non-social) behavior, whereas :code:`group_2` targets social (dyadic) behavior. This is reflected in the required columns, individual annotations only needs an :code:`'actor'` column, but dyadic annotations require an :code:`'actor'` and :code:`'recipient'` column. Each annotation interval (row) also needs a value for the behavioral :code:`category`, and :code:`'start'` and :code:`'stop'` timestamps.

If you collected your behavioral data with scoring software such as BORIS, you can use pandas to read the data into a DataFrame, drop unnecessary columns, and rename columns to match the required columns.

.. attention::
    When creating annotated dataset objects, the behavioral annotation data is checked to meet a few requirements. All required columns must be present and intervals should be strictly non-overlapping per actor (also across different actor-recipient dyads). Intervals should also be sorted by :code:`'start'` timestamps.

.. jupyter-execute::

    import pandas as pd
    from interactive_table import Table

    # Create example annotations for group_1
    observations_group_1 = pd.DataFrame(
        {
            'actor': ['animal_1', 'animal_2', 'animal_3'],
            'category': ['foraging', 'grooming', 'foraging'],
            'start': [10, 20, 30],
            'stop': [15, 25, 35]
        }
    )

    # Create example annotations for group_2
    observations_group_2 = pd.DataFrame(
        {
            'actor': ['animal_1', 'animal_1', 'animal_3'],
            'recipient': ['animal_2', 'animal_3', 'animal_2'],
            'category': ['fighting', 'fighting', 'grooming'],
            'start': [10, 20, 30],
            'stop': [15, 25, 35]
        }
    )

    annotated_group_1 = group_1.annotate(
        observations_group_1,
        categories=('foraging', 'grooming'),
        background_category='none',
    )
    annotated_group_2 = group_2.annotate(
        observations_group_2,
        categories=('fighting', 'grooming'),
        background_category='none',
    )

    print("Observations for group 1:")
    display(Table(annotated_group_1.observations))

    print("Observations for group 2:")
    display(Table(annotated_group_2.observations))

.. note::
    Intervals that are not annotated are automatically assigned to the behavioral background category.

4. Creating Datasets
--------------------

Finally, since your dataset most likely contains multiple groups, you can create an annotated dataset as the entry point for the entire *automated-scoring* pipeline.

.. jupyter-execute::

    from automated_scoring.dataset import AnnotatedDataset

    observations = pd.DataFrame(
        {
            'group': ['a', 'a', 'a', 'b', 'b'],
            'actor': ['a_1', 'a_1', 'a_3', 'b_2', 'b_3'],
            'recipient': ['a_2', 'a_3', 'a_2', 'b_3', 'b_2'],
            'category': ['fighting', 'fighting', 'grooming', 'fighting', 'fighting'],
            'start': [10, 20, 30, 15, 16],
            'stop': [15, 25, 35, 42, 35],
        }
    )

    group_a = Group(
        trajectories = {
            animal: create_random_trajectory().sort().interpolate()
            for animal in ["a_1", "a_2", "a_3"]
        },
        target="dyad",
    )

    group_b = Group(
        trajectories = {
            animal: create_random_trajectory().sort().interpolate()
            for animal in ["b_1", "b_2", "b_3"]
        },
        target="dyad",
    )

    dataset = AnnotatedDataset(
        {
            "a": group_a,
            "b": group_b,
        },
        observations=observations,
        target="dyad",
        categories=('fighting', 'grooming'),
        background_category='none',
    )

    print(dataset.category_counts)

With this dataset at hand, you can proceed with the pipeline by defining a feature extractor, sampling the dataset, and then training a classifier.
