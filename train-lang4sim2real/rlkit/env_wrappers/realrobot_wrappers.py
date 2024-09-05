class FrkaEnvTrainWrapper:
    def __init__(self, env, num_tasks, target_obj):
        self.__dict__ = env.__dict__
        self.env = env
        self.num_tasks = num_tasks
        self.task_str_format = None
        self.task_idx = None
        self.target_obj = target_obj

    def step(self, *args, **kwargs):
        pass

    def reset(self):
        pass

    def import_realrobot_pkgs(self):
        pass

    def get_task_lang_dict(self):
        return self.env.get_task_lang_dict()
