import json

from solution.globals import constants


class ThickLinesCouldNotHaveBeneIdentified(Exception):
    def __init__(self, type: str, identified: int):
        self.message = f"The thick lines of the cube couldn't have been properly detected ({identified} / {constants.cube_size} {type} segments were identified)."

        super().__init__(self.message)

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, 
            sort_keys=True, indent=4)

class ThePathsCouldNotHaveBeenProperlyIdentified(Exception):
    def __init__(self):
        self.message = f"The paths couldn't have been properly identified."

        super().__init__(self.message)

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, 
            sort_keys=True, indent=4)

class TheNumberOfIdentifiedFacesIsWrong(Exception):
    def __init__(self, identified_faces_count: int):
        self.message = f"{identified_faces_count} out of {constants.cubed_cube_size} faces were identified."

        super().__init__(self.message)

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, 
            sort_keys=True, indent=4)

class SomethingWrongHappened(Exception):
    def __init__(self):
        self.message = f"Something wrong happened. Please try again."

        super().__init__(self.message)

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, 
            sort_keys=True, indent=4)


def check_thick_lines_points_have_been_identified_correctly(
    vertical_edges_left_points,
    vertical_edges_right_points,
    backslash_edges_up_points,
    backslash_edges_down_points,
    slash_edges_down_points,
    slash_edges_up_points
):
    types = ["Vertical [left]", "Vertical [right]", "Backslash [up]",
             "Backslash [down]", "Slash [up]", "Slash [down]"]

    for type, points in zip(types, [vertical_edges_left_points, vertical_edges_right_points, backslash_edges_up_points, backslash_edges_down_points, slash_edges_up_points, slash_edges_down_points]):
        identified = len(points)

        if identified != constants.cube_size:
            raise ThickLinesCouldNotHaveBeneIdentified(type, identified)
