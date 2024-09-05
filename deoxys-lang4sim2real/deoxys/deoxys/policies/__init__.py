from deoxys.policies.pick_place import (PickPlace, PickPlaceN)
from deoxys.policies.wrap_wire import WrapWire

name_to_policy_class_map = {
    "pick_place": PickPlace,  # makes wedge motion /\ for pick, lift, drop
    "pick_place_n": PickPlaceN,  # makes n shape motion |-| for pick, lift, intermediate, and drop
    "wrap_wire": WrapWire,
}


def get_policy_class(policy_name):
    return name_to_policy_class_map[policy_name]
