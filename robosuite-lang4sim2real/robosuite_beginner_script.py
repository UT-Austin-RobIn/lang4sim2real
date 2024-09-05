import robosuite
import numpy as np
import imageio
from matplotlib import pyplot as plt

options = {}
options["env_name"] = "Lift"
options["controller_configs"] = robosuite.load_controller_config(default_controller="OSC_POSE")
options["robots"] = "Panda"

env = robosuite.make(
    **options,
    has_renderer=True,
    has_offscreen_renderer=False,
    ignore_done=True,
    use_camera_obs=False,
    horizon=1000,
    control_freq=20,
    camera_names=["agentview", "robot0_eye_in_hand", "frontview"]
)

obs = env.reset()

# # We need to invert the numpy array because of matplotlib visualization convention
# plt.imshow(np.concatenate((obs["agentview_image"][::-1], obs["robot0_eye_in_hand_image"][::-1]), axis=1))
# plt.axis("off")
# plt.show()

print(obs.keys())

images = []
for _ in range(100):
    obs, reward, done, _ = env.step(np.random.randn(7))
    # images.append(obs["frontview_image"][::-1])
    env.render()

writer = imageio.get_writer('output.mp4', fps=20)
for image in images:
    writer.append_data(image)
writer.close()
