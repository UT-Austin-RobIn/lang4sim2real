import numpy as np

OBJ_NAME_TO_XYZ_OFFSET = {
    None: np.zeros(3),
    "milk": np.array([0,0,0.045]),
    "milk small": np.array([0,0,0.015]),
    "bread": np.zeros(3),
    "can": np.array([0,0,0.02]),
    "cereal": np.array([0,0,0.05]),
    "cereal small": np.array([0,0,0.035]),
    "platform": np.array([0,0,0]),
    "zone": np.array([0,0,0]),
    "pot": np.array([0, 0.06, 0.02]),
    "stove": np.array([0, 0.00, 0.01]),
    "spool": np.array([0, 0, 0.02]),
    "wire": np.array([0, 0, 0.02]),
    "spoolandwire": np.array([0, 0, .005]),
}
