from solution.enums import Axis


class Hyperparameters:
    def __init__(self):
        self.keep_edges_in_image_threshold = 127
        self.struct_sizes_err = 0.8
        self.face_image_extractor_cube_area_threshold = .75
        self.iou_threshold = 0.5

        self.padding_amount = dict()

        self.padding_amount[Axis.X] = 20
        self.padding_amount[Axis.Y] = 20
        self.padding_amount[Axis.Z] = 20

    def __str__(self):
        return f"Hyperparameters[keep_edges_in_image_threshold={self.keep_edges_in_image_threshold}]"