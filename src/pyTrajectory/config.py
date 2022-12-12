KEYS = ('time_stamp',
        'position',
        'pose',
        'segmentation',
        'bbox',
        'score',
        'category')

VISUALIZATION_CONFIG = dict(
    plot_segmentation = False,
    plot_pose = True,
    plot_position = False,
    segmentation_edge_color = 'k',
    segmentation_edge_alpha = 1,
    segmentation_face_color = 'k',
    segmentation_face_alpha = 0,
    segmentation_edge_width = 0.5,
    pose_line_color = 'k',
    pose_line_alpha = 1,
    pose_line_width = 1,
    position_marker = 'o',
    position_face_color = 'k',
    position_face_alpha = 1,
    position_edge_color = 'k',
    position_edge_alpha = 0,
    position_line_width = 1,
    position_size = 5
)
