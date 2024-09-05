"""
https://arxiv.org/pdf/1910.07113.pdf
Follows Algorithm 1 on page 12.
"""
import numpy as np

from robosuite.wrappers.domain_randomization_wrapper import (
    DomainRandomizationWrapper)
from robosuite.utils.mjmod import TextureModder


DEFAULT_DYNAMICS_ARGS = {
    # Opt parameters
    "randomize_density": False,  # True,
    "randomize_viscosity": False,  # True,
    "density_perturbation_ratio": 0.1,
    "viscosity_perturbation_ratio": 0.1,
    # Body parameters
    "body_names": None,  # all bodies randomized
    "randomize_position": False,  # True,
    "randomize_quaternion": False,  # True,
    "randomize_inertia": False,  # True,
    "randomize_mass": False,  # True,
    "position_perturbation_size": 0.0015,
    "quaternion_perturbation_size": 0.003,
    "inertia_perturbation_ratio": 0.02,
    "mass_perturbation_ratio": 0.02,
    # Geom parameters
    "geom_names": None,  # all geoms randomized
    "randomize_friction": False,  # True,
    "randomize_solref": False,  # True,
    "randomize_solimp": False,  # True,
    "friction_perturbation_ratio": 0.1,
    "solref_perturbation_ratio": 0.1,
    "solimp_perturbation_ratio": 0.1,
    # Joint parameters
    "joint_names": None,  # all joints randomized
    "randomize_stiffness": True,
    "randomize_frictionloss": True,
    "randomize_damping": False,  # True,
    "randomize_armature": False,  # True,
    "stiffness_perturbation_ratio": 0.1,
    "frictionloss_perturbation_size": 0.05,
    "damping_perturbation_size": 0.01,
    "armature_perturbation_size": 0.01,
}


class AlphaDummyModder:
    """
    Create a dummy modder for the alpha parameter (controlling RNA weight)
    """
    def __init__(self, action_size):
        self.action_size = action_size
        self.param_idx_to_name = [f"alpha{i}" for i in range(self.action_size)]
        self.num_params = len(self.param_idx_to_name)
        self.param_idx_to_hilo_range = np.array(
            [[0.0, 0.0] for param_idx in self.param_idx_to_name]
        )
        self.randomize()

    def get_param_limits(self, param_idx):
        """
        For automatic domain randomization
        """
        return self.param_idx_to_hilo_range[param_idx]

    def update_param_range(self, param_idx, new_range):
        # assert isinstance(new_range, list)
        assert len(new_range) == 2
        if new_range[0] is not None and new_range[1] is not None:
            assert new_range[0] <= new_range[1]

        # Clip alpha bounds to between 0, 1
        self.param_idx_to_hilo_range[param_idx] = np.clip(new_range, 0, 1)

    def get_num_adr_params(self):
        return len(self.param_idx_to_name)

    def randomize(self):
        self.alpha_vec = np.random.uniform(
            self.param_idx_to_hilo_range[:, 0],
            self.param_idx_to_hilo_range[:, 1])

    def set_param(self, param_idx, val):
        lo, hi = self.param_idx_to_hilo_range[param_idx]
        assert lo <= val <= hi
        self.alpha_vec[param_idx] = val

    def get_alpha_vec(self):
        return self.alpha_vec

    def save_defaults(self):
        pass

    def update_sim(self, sim):
        pass

    def restore_defaults(self):
        self.alpha_vec = np.zeros(self.action_size)


class AutomaticDomainRandomizationWrapper(DomainRandomizationWrapper):
    """
    ADR + RNA
    """
    def __init__(
            self,
            param_step_size,
            num_trajs_per_bound_update,
            *args,
            **kwargs):
        super().__init__(
            randomize_every_n_steps=0,
            randomize_color=False,
            randomize_camera=True,
            randomize_lighting=True,
            randomize_dynamics=True,
            dynamics_randomization_args=DEFAULT_DYNAMICS_ARGS,
            *args, **kwargs)

        import rlkit.util.pytorch_util as ptu

        # RNA params
        self.state_size = None
        self.action_size = self.action_space.low.shape[0]
        self.action_space_by_dim = np.concatenate(
            [self.action_space.low[None],
             self.action_space.high[None]]).T
        alpha_modder = AlphaDummyModder(self.action_size)
        self.modders.append(alpha_modder)
        self.alpha_idx = (len(self.modders) - 1, 0)

        self.num_modders = len(self.modders)
        self.modder_param_idx_to_hilo_range = self.get_param_ranges()
        self.modder_param_idx_to_perf_history = self.init_performance_buffers()
        self.param_step_size = param_step_size
        self.num_trajs_per_bound_update = num_trajs_per_bound_update
        self.perf_lo, self.perf_hi = (0.25, 0.75)

    def get_param_ranges(self):
        param_ranges = {}
        for i in range(self.num_modders):
            if isinstance(self.modders[i], TextureModder):
                continue
            for j in range(self.modders[i].num_params):
                param_ranges[(i, j)] = self.modders[i].get_param_limits(j)
        return param_ranges

    def init_performance_buffers(self):
        perf_buffers = {}
        for i in range(self.num_modders):
            if isinstance(self.modders[i], TextureModder):
                continue
            for j in range(self.modders[i].num_params):
                perf_buffers[(i, j)] = []
        return perf_buffers

    def init_rand_net_adv(self):
        """
        Net params from https://arxiv.org/pdf/1910.07113.pdf page 41
        """
        from rlkit.torch.networks.rand_net_adv import RandNetAdv
        if self.state_size is None:
            obs = self.env._get_observations()
            state = obs['state']
            self.state_size = state.shape[0]
        rna = RandNetAdv(
            input_size=self.state_size,
            action_size=self.action_size,
            action_space_by_dim=self.action_space_by_dim,
        )
        return rna

    def update_param_range(self, expand_range):
        modder_param_idx = self.get_boundary_modder_param_idx()
        modder_idx, param_idx = modder_param_idx
        hi_or_lo = self.boundary_param_dict["hi_or_lo"]
        new_range = self.modder_param_idx_to_hilo_range[modder_param_idx]
        if hi_or_lo == "hi":
            if expand_range:
                new_range[1] += self.param_step_size
            elif ((new_range[0] is None
                    and (new_range[1] - self.param_step_size) > 0) or
                    (new_range[0] is not None
                        and (new_range[0] + self.param_step_size < new_range[1]))):
                new_range[1] -= self.param_step_size
        elif hi_or_lo == "lo":
            if expand_range:
                new_range[0] -= self.param_step_size
            elif (new_range[0] + self.param_step_size) < new_range[1]:
                new_range[0] += self.param_step_size
        self.modders[modder_idx].update_param_range(param_idx, new_range)
        return

    def select_boundary_param(self):
        modder_idx = np.random.choice(range(self.num_modders))

        # No ADR support for texture modder (color ranges span entire
        # RGB spectrum and textures to sample from are discrete)
        while isinstance(self.modders[modder_idx], TextureModder):
            modder_idx = np.random.choice(range(self.num_modders))

        # For debug purposes:
        # modder_idx = 0
        # print(
        #     f"forcing modder_idx to be {modder_idx}, "
        #     f"{self.modders[modder_idx]}")

        param_idx = np.random.choice(
            self.modders[modder_idx].get_num_adr_params())
        param_name = self.modders[modder_idx].param_idx_to_name[param_idx]

        # For debug purposes:
        # print(f"forcing param_idx to be {param_idx}, {param_name}")

        hi_or_lo = np.random.choice(["hi", "lo"])
        if param_name in ["dir", "quat"]:
            hi_or_lo = "hi"
        boundary_param_dict = dict(
            modder_idx=modder_idx,
            param_idx=param_idx,
            param_name=param_name,
            hi_or_lo=hi_or_lo)
        return boundary_param_dict

    def get_alpha_vec(self):
        return self.modders[self.alpha_idx[0]].get_alpha_vec()

    def reset(self):
        self.rna = self.init_rand_net_adv()
        print(
            "self.modder_param_idx_to_hilo_range",
            self.modder_param_idx_to_hilo_range)
        self.boundary_param_dict = self.select_boundary_param()
        ret = super().reset()

        # Get alpha for the trajectory
        self.alpha_vec = self.get_alpha_vec()
        print("alpha_vec (reset)", self.alpha_vec)
        return ret

    def step(self, action):
        """
        Extends vanilla step() function call to accommodate domain
        randomization and adding RNA action.

        Returns:
            4-tuple:

                - (OrderedDict) observations from the environment
                - (float) reward from the environment
                - (bool) whether the current episode is completed or not
                - (dict) misc information
        """
        # Step the internal randomization state
        self.step_randomization()
        obs = self.env._get_observations()
        state = ptu.from_numpy(obs['state'][None])
        # ^ reshape to (1, state_size)
        rna_action = self.rna(state)[0]
        # ^ output is (1, 7), we want it to be (7,)
        action = (1 - self.alpha_vec) * action + self.alpha_vec * rna_action
        return super().step(action)

    def _post_traj(self, traj_perf):
        """
        After trajectory has been finished, update performance history
        """
        # TODO: call this in data collection
        param_perf_history = self.update_perf_history(traj_perf)
        if len(param_perf_history) >= self.num_trajs_per_bound_update:
            avg_perf = np.mean(param_perf_history)
            self.update_perf_history("clear_history")
            if avg_perf >= self.perf_hi:
                self.update_param_range(expand_range=True)
            elif avg_perf <= self.perf_lo:
                self.update_param_range(expand_range=False)

    def randomize_domain(self):
        """
        Runs domain randomization over the environment.
        """
        for modder in self.modders:
            modder.randomize()

        # Set the boundary parameter to a specific value
        modder_idx, modder_param_idx = self.get_boundary_modder_param_idx()
        hi_or_lo = self.boundary_param_dict["hi_or_lo"]
        self.modder_param_idx_to_hilo_range = self.get_param_ranges()
        param_hi_lo = self.modder_param_idx_to_hilo_range[
            (modder_idx, modder_param_idx)]
        val = param_hi_lo[0] if hi_or_lo == "lo" else param_hi_lo[1]
        # print(
        #     f"about to set {self.boundary_param_dict['param_name']} "
        #     f"to {hi_or_lo} {val}")
        self.modders[modder_idx].set_param(modder_param_idx, val)

    def get_boundary_modder_param_idx(self):
        modder_idx = self.boundary_param_dict["modder_idx"]
        param_idx = self.boundary_param_dict["param_idx"]
        modder_param_idx = modder_idx, param_idx
        return modder_param_idx

    def update_perf_history(self, traj_perf):
        modder_param_idx = self.get_boundary_modder_param_idx()

        if traj_perf == "clear_history":
            self.modder_param_idx_to_perf_history[modder_param_idx] = []
        elif isinstance(traj_perf, float) or isinstance(traj_perf, int):
            self.modder_param_idx_to_perf_history[modder_param_idx].append(
                traj_perf)
        else:
            raise NotImplementedError
        return self.modder_param_idx_to_perf_history[modder_param_idx]
