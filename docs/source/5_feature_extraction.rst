Feature extraction
==================

.. jupyter-execute::
    :hide-code:

    import numpy as np
    import pandas as pd

    from interactive_table import Table

    from automated_scoring.dataset import Group, AnnotatedDataset
    from automated_scoring.config import cfg
    from automated_scoring.data_structures import Trajectory

    # configure trajectory objects, providing your names for the collected data
    cfg.trajectory_keys = ("time", "posture")

    # assign them to the following two, minimally required configuration keys
    cfg.key_timestamp = "time"
    cfg.key_keypoints = "posture"


    num_frames = 100
    num_keypoints = 4
    shape = (num_frames, num_keypoints, 2)

    # create a random number generator
    rng = np.random.default_rng(1)


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

On the previous page, we created a dataset with two groups of animals, each with three trajectories. We then created a dataframe of observations, which we used to create an annotated dataset. Let's recap and display the observations in a table.

.. jupyter-execute ::

    display(Table(dataset.observations))

The dataset consists of groups and can be iterated over. Similarly, groups consist of dyads (note the target parameter) and can also be iterated over. Each of these objects is a :code:`Sampleable` that provides methods for feature extraction, compatible with our feature extraction workflow.

.. jupyter-execute ::

    for group_id, group in dataset:
        for dyad_id, dyad in group:
            print(f"Group {group_id}, Dyad {dyad_id} with {len(dyad)} samples")

Manual feature calculation
--------------------------

Each timestamp that both animals are present is considered a sample. For each sample, we can extract features such as distances, angles, or speed. We can do this manually by using the feature functions defined in the :code:`automated_scoring.features` module:

.. hint ::
    All feature functions return a numpy array containing the computed features. The shape depends on the number of postural elements and whether the feature should be computed :code:`element_wise` with regard to these elements.
    All feature functions have an additional :code:`flat` parameter, if specified, the features will be returned as an array with shape (n_samples, n_features).

.. jupyter-execute ::

    from automated_scoring.features import (
        keypoint_distances, posture_angles, speed
    )

    # calculate all pairwise keypoint distances (for four keypoints)
    # element_wise=True to only calculate four distances
    # (would otherwise calculate 16 distances)
    distances = keypoint_distances(
        dyad.trajectory,
        trajectory_other=dyad.trajectory_other,
        keypoints_1=(0, 1, 2, 3),
        keypoints_2=(0, 1, 2, 3),
        element_wise=True,
        flat=True,
    )
    # calculate one posture angle for the 'actor'
    # (the first individual in the dyad)
    angles_actor = posture_angles(
        dyad.trajectory,
        keypoint_pairs_1=((1, 0), ),
        keypoint_pairs_2=((3, 2), ),
        flat=True,
    )
    # calculate one angle between posture angles between
    # both individuals using the same function
    angles = posture_angles(
        dyad.trajectory,
        trajectory_other=dyad.trajectory_other,
        keypoint_pairs_1=((1, 0), ),
        keypoint_pairs_2=((1, 0), ),
        flat=True,
    )
    # calculate speed for both individuals with a step size of 15
    speed_actor = speed(
        dyad.trajectory,
        keypoints=(0, 2),
        step=15,
        flat=True,
    )
    speed_recipient = speed(
        dyad.trajectory_other,
        keypoints=(0, 2),
        step=15,
        flat=True,
    )

    features = np.concatenate((distances, angles_actor, angles, speed_actor, speed_recipient), axis=1)
    print(features.shape)

For a more intuative and transparent worfklow (especially in subsequent steps), feature functions can be wrapped using the :code:`as_dataframe` decorator:

.. jupyter-execute ::

    from automated_scoring.features import as_dataframe

    speed_actor_df = as_dataframe(speed)(
        dyad.trajectory,
        keypoints=(0, 2),
        step=15,
        flat=True,
    )
    speed_recipient_df = as_dataframe(speed)(
        dyad.trajectory_other,
        keypoints=(0, 2),
        step=15,
        flat=True,
    )

    print(pd.concat([speed_actor_df, speed_recipient_df], axis=1).head())

.. note ::

    In the example above, speed is a temporal feature and needs some padding (:code:`step // 2`) to align with other features, therefore the first values are repeated.
    Feature names can not distinguish between actor and recipient because they are calculated using the same function.

Using feature extractors
------------------------

A more reproducible workflow can be achieved by using the :code:`DataFrameFeatureExtractor` class.
This allows you to define a set of features and their parameters in a more structured way.

.. jupyter-execute ::

    from automated_scoring.features import DataFrameFeatureExtractor

    extractor = DataFrameFeatureExtractor(
        features=[
            (
                posture_angles,
                dict(
                    keypoint_pairs_1=((1, 0), ),
                    keypoint_pairs_2=((3, 2), ),
                ),
            ),
            (
                speed,
                dict(
                    keypoints=(0, 2),
                    step=15,
                ),
            ),
            (
                speed,
                dict(
                    keypoints=(0, 2),
                    step=15,
                    reversed_dyad=True,
                ),
            )
        ],
        dyadic_features=[
            (
                keypoint_distances,
                dict(
                    keypoints_1=(0, 1, 2, 3),
                    keypoints_2=(0, 1, 2, 3),
                    element_wise=True,
                ),
            ),
            (
                posture_angles,
                dict(
                    keypoint_pairs_1=((1, 0), ),
                    keypoint_pairs_2=((3, 2), ),
                ),
            ),
        ],
        cache_mode=False,
    )

    print(extractor.extract(dyad.trajectory, dyad.trajectory_other).head())

This computes the same 10 features that were obtained before by concatenating the results of separate feature functions, but has the advantage of being more reproducible and easier to maintain.
For example, we can easily save and load the configuration to and from a YAML file.

.. jupyter-execute ::

    extractor.save_yaml('config.yaml')
    extractor = DataFrameFeatureExtractor(cache_mode=False).read_yaml('config.yaml')

The saved configuration file is shown below. You can also start by creating such a YAML file and add individual and dyadic features with their respective arguments.

.. literalinclude:: ../config.yaml
   :language: yaml

Have a look at the API documentation (submodules :code:`features.features` and :code:`feautres.temporal_features`) for all implemented features and their respective arguments.

.. hint ::
    Feature extractors allow additional arguments to be passed to the feature functions to allow for more flexibility:

    - :code:`as_absolute` to compute absolute values (e.g., helpful for positive and negative angles).
    - :code:`reversed_dyad` to switch actor and recipient trajectories, for example to calculate specific individual featues for the recipient.
    - :code:`as_sign_change_latency` to compute the latency between sign changes of a feature, for example to measure the time between changes a clockwise or anticlockwise posture angle.

    The :code:`DataFrameFeatureExtractor` adds additional arguments to specify which columns to :code:`discard` from the resulting DataFrame. You can specify one or more strings (patterns) to drop, and use :code:`keep` to specify exceptions for columns that should be kept regardless of these patterns.

Feature extractors can not only be used with their :code:`extract` method, but also as an argument for all :code:`Sampleable` dataset objects. This includes individuals, dyads, groups, and datasets.
