from collections.abc import Sequence
from typing import Tuple, List, Set
from enum import IntEnum, unique

@unique
class Attributes3DSSG(IntEnum):
    """ Class containing all information for semantic attribute taxonomy in 3DSSG dataset"""
    def __str__(self) -> str:
        return super().__str__().replace("Attributes3DSSG.", "")

    @staticmethod
    def binary_encode(rels: Sequence) -> List[bool]:
        ret = [False] * len(Attributes3DSSG)
        rel_idx = [Attributes3DSSG.key_to_value(r) for r in rels]
        for r in rel_idx:
            ret[r] = True
        return ret

    @staticmethod
    def key_to_value(key: str) -> "Attributes3DSSG":
        def mapping():
            return {str(x): x for x in Attributes3DSSG}
        return mapping()[key]


    @staticmethod
    def categories():
        return [
        "color",
        "shape",
        "style",
        "state",
        "size",
        "material",
        "texture",
        "other",
        "symmetry",
    ]

    @staticmethod
    def to_enum(raw: str) -> "Attributes3DSSG":
        e = raw.replace(":", "_").replace(" ", "_").replace("/", "_or_").replace("-", "_").lower()
        try:
            return Attributes3DSSG.key_to_value(e)
        except KeyError:
            raise ValueError(f"{raw} cannot be converted to known attributes")
    color_white = 0
    color_black = 1
    color_green = 2
    color_blue = 3
    color_red = 4
    color_brown = 5
    color_yellow = 6
    color_gray = 7
    color_orange = 8
    color_purple = 9
    color_pink = 10
    color_beige = 11
    color_bright = 12
    color_dark = 13
    color_light = 14
    color_silver = 15
    color_gold = 16
    shape_round = 17
    shape_flat = 18
    shape_l_shaped = 19
    shape_semicircular = 20
    shape_circular = 21
    shape_square = 22
    shape_rectangular = 23
    shape_sloping = 24
    shape_cylindrical = 25
    shape_oval = 26
    shape_bunk = 27
    shape_heart_shaped = 28
    shape_u_shaped = 29
    shape_octagon = 30
    style_classy = 31
    style_classical = 32
    style_minimalistic = 33
    state_new = 34
    state_old = 35
    state_dirty = 36
    state_clean = 37
    state_open = 38
    state_empty = 39
    state_full = 40
    state_hanging = 41
    state_half_full_or_empty = 42
    state_closed = 43
    state_half_open_or_closed = 44
    state_messy = 45
    state_tidy = 46
    state_on = 47
    state_off = 48
    state_folded_together = 49
    state_seat_up = 50
    state_seat_down = 51
    state_up = 52
    state_down = 53
    state_half_up_or_down = 54
    state_bare = 55
    state_written_on = 56
    size_small = 57
    size_big = 58
    size_tall = 59
    size_low = 60
    size_narrow = 61
    size_wide = 62
    material_wooden = 63
    material_plastic = 64
    material_metal = 65
    material_glass = 66
    material_stone = 67
    material_leather = 68
    material_concrete = 69
    material_ceramic = 70
    material_brick = 71
    material_padded = 72
    material_cardboard = 73
    material_marbled = 74
    material_carpet = 75
    material_cork = 76
    material_velvet = 77
    texture_striped = 78
    texture_patterned = 79
    texture_dotted = 80
    texture_colorful = 81
    texture_checker = 82
    texture_painted = 83
    texture_shiny = 84
    texture_tiled = 85
    other_mobile = 86
    other_rigid = 87
    other_nonrigid = 88
    symmetry_no_symmetry = 89
    symmetry_1_plane = 90
    symmetry_2_planes = 91
    symmetry_infinite_planes = 92
