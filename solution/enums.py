from enum import Enum


class SegmentType(Enum):
    VERTICAL = 0
    SLASH = 1
    BACKSLASH = 2


class PathDirection(Enum):
    VERTICAL = 0
    HORIZONTAL = 1


class Axis(Enum):
    X = 0
    Y = 1
    Z = 2
