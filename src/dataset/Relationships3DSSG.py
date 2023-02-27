from collections.abc import Sequence
from enum import IntEnum, unique
from typing import Tuple, List, Set

@unique
class Relationships3DSSG(IntEnum):
    """ Class containing all information for semantic relationship taxonomy in 3DSSG dataset"""
    def __str__(self):
        return super().__str__().replace("Relationships","")

    @staticmethod
    def key_to_value(key: str) -> "Relationships3DSSG":
        def mapping():
            return {str(x).split(".")[-1]: x for x in Relationships3DSSG}
        return mapping()[key]

    @staticmethod
    def binary_encode(rels: Sequence) -> List[bool]:
        ret = [False] * len(Relationships3DSSG)
        rel_idx = [Relationships3DSSG.key_to_value(r) for r in rels]
        for r in rel_idx:
            ret[r] = True
        return ret

    @staticmethod
    def to_enum(raw: str) -> "Relationships3DSSG":
        e = raw.replace(":", "_").replace(" ", "_").replace("/", "_or_").replace("-", "_").lower()
        try:
            return Relationships3DSSG.key_to_value(e)
        except KeyError:
            raise ValueError(f"{raw} cannot be converted to known attributes")


    none = 0
    supported_by = 1
    left = 2
    right = 3
    front = 4
    behind = 5
    close_by = 6
    inside = 7
    bigger_than = 8
    smaller_than = 9
    higher_than = 10
    lower_than = 11
    same_symmetry_as = 12
    same_as = 13
    attached_to = 14
    standing_on = 15
    lying_on = 16
    hanging_on = 17
    connected_to = 18
    leaning_against = 19
    part_of = 20
    belonging_to = 21
    build_in = 22
    standing_in = 23
    cover = 24
    lying_in = 25
    hanging_in = 26
    same_color = 27
    same_material = 28
    same_texture = 29
    same_shape = 30
    same_state = 31
    same_object_type = 32
    messier_than = 33
    cleaner_than = 34
    fuller_than = 35
    more_closed = 36
    more_open = 37
    brighter_than = 38
    darker_than = 39
    more_comfortable_than = 40
