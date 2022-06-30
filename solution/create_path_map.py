from collections import defaultdict
from typing import List

from solution.types import PathMap, SudokuPath

def create_path_map(paths: List[SudokuPath]):
    paths_by_coordinates: PathMap = defaultdict(lambda: [])

    for path in paths:
        for face in path.faces:
            paths_by_coordinates[face.coordinates].append(path)

    return paths_by_coordinates
