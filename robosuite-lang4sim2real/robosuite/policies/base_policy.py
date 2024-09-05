import abc  # for abstract base class definitions


class Policy(metaclass=abc.ABCMeta):
    """Base Class for Policy"""

    @abc.abstractmethod
    def start_control(self):
        raise NotImplementedError

    @abc.abstractmethod
    def get_action(self, env_obs):
        pass
