import gym

import metaworld
import random
import mtenv
from mtenv.envs.metaworld.wrappers.normalized_env import (  # type: ignore[attr-defined]
    NormalizedEnvWrapper,
)


def game(args):
    # print(metaworld.MT1.ENV_NAMES)  # Check out the available environments

    mt1 = metaworld.MT1(args['task'])  # Construct the benchmark, sampling tasks

    env = mt1.train_classes[args['task']]()  # Create an environment with task `pick_place`
    task = random.choice(mt1.train_tasks)
    env.set_task(task)  # Set task
    env.seed(args['seed'])
    # env._freeze_rand_vec = True
    # env = NormalizedEnvWrapper(env,normalize_reward=True)
    return env
