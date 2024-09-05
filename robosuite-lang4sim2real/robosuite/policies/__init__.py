from .base_policy import Policy
from .grasp import GraspPolicy
from .pick_place import PickPlacePolicy
from .push import PushPolicy
from .wrap_policy import WrapPolicy


NAME_TO_POLICY_CLASS_MAP = {
    "grasp": GraspPolicy,
    "pick_place": PickPlacePolicy,
    "stack": PickPlacePolicy,
    "push": PushPolicy,
    "wrap": WrapPolicy,
    "wrap-completion": WrapPolicy,
    "wrap-relative-location": WrapPolicy,
}
