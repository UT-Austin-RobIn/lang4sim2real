from functools import partial
import numpy as np
import rlkit.util.pytorch_util as ptu

create_rollout_function = partial


def rollout(
        env,
        agent,
        max_path_length=np.inf,
        render=False,
        render_kwargs=None,
        preprocess_obs_for_policy_fn=None,
        get_action_kwargs=None,
        return_dict_obs=False,
        full_o_postprocess_func=None,
        full_next_o_postprocess_func=None,
        reset_callback=None,
        relabel_rewards=False,
        task_emb_input_mode="concat_to_img_embs",
        aux_tasks=[],
        obs_processor_kwargs={},

        # kwargs for dist on final img obs emb between demos and rollouts
        output_cnn_embs_in_aux=False,
):
    if render_kwargs is None:
        render_kwargs = {}
    if get_action_kwargs is None:
        get_action_kwargs = {}
    get_action_kwargs['output_cnn_embs_in_aux'] = output_cnn_embs_in_aux
    if preprocess_obs_for_policy_fn is None:
        preprocess_obs_for_policy_fn = lambda x: x
    raw_obs = []
    raw_next_obs = []
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    next_observations = []
    final_obs_cnn_emb = None  # CNN emb of the last observation in the rollout.
    path_length = 0
    agent.reset()
    o = env.reset()
    if reset_callback:
        reset_callback(env, agent, o)
    if render:
        env.render(**render_kwargs)
    lang_embedding = -1
    if 'lang_embedding' in o.keys() and isinstance(
            o['lang_embedding'][0], np.ndarray):
        lang_embedding = [np.random.choice(len(o['lang_embedding']))]
    while path_length < max_path_length:
        raw_obs.append(o)
        if task_emb_input_mode in [
                "film", "film_video_concat_lang", "film_lang_concat_video"]:
            if lang_embedding != -1:
                o['lang_embedding'] = o['lang_embedding'][lang_embedding]
            o_dict = preprocess_obs_for_policy_fn(o, **obs_processor_kwargs)
            o_for_agent = o_dict['non_emb_obs']
            get_action_kwargs.update(film_inputs=o_dict['emb'])            
        else:
            o_for_agent = preprocess_obs_for_policy_fn(o)
        a, stats_dict, aux_outputs = agent.get_action(
            o_for_agent, **get_action_kwargs)
        agent_info = {}

        next_o, r, d, env_info = env.step(a.copy())

        env_info['reward'] = r

        # Add val aux losses and extra stats to env_info
        if (isinstance(aux_outputs, dict) and
                "losses" in aux_outputs and
                isinstance(aux_outputs['losses'], dict)):
            for aux_task in aux_tasks:
                env_info[f"{aux_task} loss"] = float(
                    ptu.get_numpy(aux_outputs['losses'][aux_task]))
        if isinstance(stats_dict, dict):
            env_info.update(stats_dict)

        if full_o_postprocess_func:
            full_o_postprocess_func(env, agent, o)
        if full_next_o_postprocess_func:
            full_next_o_postprocess_func(env, agent, next_o)

        if render:
            env.render(**render_kwargs)
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        next_observations.append(next_o)
        raw_next_obs.append(next_o)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        if ((d or (path_length == max_path_length - 1)) and
                (isinstance(aux_outputs, dict) and
                    "cnn_emb_for_dist_learning" in aux_outputs)):
            final_obs_cnn_emb = aux_outputs["cnn_emb_for_dist_learning"]

        path_length += 1
        if d:
            break
        o = next_o
    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    next_observations = np.array(next_observations)
    if return_dict_obs:
        observations = raw_obs
        next_observations = raw_next_obs
    rewards = np.array(rewards)
    if len(rewards.shape) == 1:
        rewards = rewards.reshape(-1, 1)
    if final_obs_cnn_emb is not None:
        final_obs_cnn_emb = np.array(final_obs_cnn_emb.detach().cpu())
    rollout_data = dict(
        observations=observations,
        actions=actions,
        rewards=rewards,
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
        full_observations=raw_obs,
        full_next_observations=raw_obs,
        final_obs_cnn_emb=final_obs_cnn_emb,
    )
    if relabel_rewards:
        rollout_data = env.relabel_rewards(rollout_data)
    return rollout_data
