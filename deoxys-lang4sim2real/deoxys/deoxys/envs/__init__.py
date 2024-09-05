from deoxys.envs.franka_env import FrankaEnv, FrankaEnvPP, FrankaEnvObjToBowlToPlate
from deoxys.envs.franka_wirewrap_env import FrankaWireWrap

name_to_env_map = {
    "frka_base": FrankaEnv,
    "frka_pp": FrankaEnvPP,
    "frka_obj_bowl_plate": FrankaEnvObjToBowlToPlate,
    "frka_wirewrap": FrankaWireWrap,
}

registered_env_names = name_to_env_map.keys()


def init_env(env_name, **kwargs):
    return name_to_env_map[env_name](**kwargs)
