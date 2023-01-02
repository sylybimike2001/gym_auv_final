from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from matplotlib import pyplot as plt
import gym,gym_auv
import os
import argparse
import numpy as np

def create_env(env_id, envconfig, test_mode=False, render_mode='2d', pilot=None, verbose=False):
    if pilot:
        env = gym.make(env_id, env_config=envconfig, test_mode=test_mode, render_mode=render_mode, pilot=pilot, verbose=verbose)
        env.seed(0)  # Thomas 04.08.21
    else:
        env = gym.make(env_id, env_config=envconfig, test_mode=test_mode, render_mode=render_mode, verbose=verbose)
        env.seed(0)  # Thomas 04.08.21
    return env

def save_agent(env_id,agent):
    PPO_path = os.path.join('Training', 'SavedModels', env_id)
    agent.save(PPO_path)
    del agent

def generate_path(env_id,env,agent_path):
    if env_id == 'gym_auv:' + 'Env0-v0':
        agent = PPO.load(agent_path, env)
        obs = env.reset()
        dones = False
        count = 0
        start_point = env.path.start
        end_point = env.path.end
        path_points = [start_point, end_point]
        while not dones:
            action, _states = agent.predict(obs)
            obs, rewards, dones, info = env.step(action)
            path_points.append(info['location'])
            count += 1
        # print(info)
        env.close()
        print("Generating path with env:" + env_id +" and agent:" +agent_path +" ,Total Steps:" + str(count))
        np.savetxt("./pathdata/" + env_id  +"_path_points", path_points)
    if env_id == 'gym_auv:' + 'Env1-v0':
        agent = PPO.load(agent_path, env)
        np.savetxt("./pathdata/" + env_id + "_org_path_points", env.path._points)
        obs = env.reset()
        dones = False
        count = 0
        path_points = []
        while not dones:
            action, _states = agent.predict(obs)
            obs, rewards, dones, info = env.step(action)
            path_points.append(info['location'])
            count += 1
        # print(info)
        env.close()
        print("Generating path with env:" + env_id +" and agent:" +agent_path +" ,Total Steps:" + str(count))
        np.savetxt("./pathdata/" + env_id + "_path_points", path_points)
    if env_id == 'gym_auv:' + 'Env2-v0':
        agent = PPO.load(agent_path, env)
        np.savetxt("./pathdata/" + env_id + "_org_path_points", env.path._points)
        obs = env.reset()
        dones = False
        count = 0
        path_points = []
        while not dones:
            action, _states = agent.predict(obs)
            obs, rewards, dones, info = env.step(action)
            path_points.append(info['location'])
            count += 1
        # print(info)
        env.close()
        print("Generating path with env:" + env_id +" and agent:" +agent_path +" ,Total Steps:" + str(count))
        np.savetxt("./pathdata/" + env_id + "_path_points", path_points)
        positions = []
        radius = []
        for static_obst in env.obstacles:
            positions.append(static_obst.position)
            radius.append(static_obst.radius)
        np.savetxt("./pathdata/" + env_id + "_positions", positions)
        np.savetxt("./pathdata/" + env_id + "_radius", radius)

def plot_path(env_id):
    if env_id == 'gym_auv:' + 'Env0-v0':
        _pathdata = np.loadtxt("./pathdata/" + env_id + "_path_points")
        start_point = _pathdata[0]
        end_point = _pathdata[1]
        xpoints = _pathdata[2:, 0]
        ypoints = _pathdata[2:, 1]

        target_xpoints = [start_point[0], end_point[0]]
        target_ypoints = [start_point[1], end_point[1]]
        plt.plot(target_xpoints, target_ypoints, 'b')
        plt.plot(xpoints, ypoints, 'r')
        plt.show()
    if env_id == 'gym_auv:' + 'Env1-v0':
        _pathdata = np.loadtxt("./pathdata/" + env_id + "_path_points")
        org_pathdata = np.loadtxt("./pathdata/" + env_id + "_org_path_points")
        org_xpoints = org_pathdata[:, 0]
        org_ypoints = org_pathdata[:, 1]
        xpoints = _pathdata[:, 0]
        ypoints = _pathdata[:, 1]
        plt.plot(xpoints, ypoints)
        plt.plot(org_xpoints, org_ypoints)
        plt.show()
    if env_id == 'gym_auv:' + 'Env2-v0':
        _pathdata = np.loadtxt("./pathdata/" + env_id + "_path_points")
        positions = np.loadtxt("./pathdata/" + env_id + "_positions")
        radius = np.loadtxt("./pathdata/" + env_id + "_radius")
        org_pathdata = np.loadtxt("./pathdata/" + env_id + "_org_path_points")
        org_xpoints = org_pathdata[:, 0]
        org_ypoints = org_pathdata[:, 1]
        xpoints = _pathdata[:, 0]
        ypoints = _pathdata[:, 1]

        figure, axes = plt.subplots()
        axes.set_aspect(1)
        for i in range(len(positions)):
            circle = plt.Circle(positions[i],radius[i],color = 'r')
            axes.add_artist(circle)
        plt.plot(org_xpoints, org_ypoints,'b')
        plt.plot(xpoints, ypoints,'r')
        plt.title('Circle')
        plt.show()

def main(args):
    env_id = 'gym_auv:' + args.scenario
    env_name = env_id.split(':')[-1] if ':' in env_id else env_id
    envconfig = gym_auv.SCENARIOS[env_name]['config'] if env_name in gym_auv.SCENARIOS else {}
    envconfig['show_indicators'] = True
    if args.mode == 'train':
        num_cpu = 1  # 4
        hyperparams = {
            # 'n_steps': 1024,
            # 'nminibatches': 32,
            # 'lam': 0.95,
            # 'gamma': 0.99,
            # 'noptepochs': 10,
            # 'ent_coef': 0.0,
            # 'learning_rate': 0.0003,
            # 'cliprange': 0.2,
            'n_steps': 1024,  # Default 128
            'batch_size': 128, #32 # Default 4
            'gae_lambda': 0.98,  # Default 0.95
            'gamma': 0.999,  # Default 0.99
            'n_epochs': 4,  # Default 4
            'ent_coef': 0.01,  # Default 0.01
            'learning_rate': 2e-3 #2e-4,  # Default 2.5e-4
        }
        # policy_kwargs = dict(act_fun=tf.nn.tanh, net_arch=[64, 64, 64])
        # policy_kwargs = dict(net_arch=[64, 64, 64])
        # layers = [256, 128, 64]
        layers = [64, 64]
        policy_kwargs = dict(net_arch=[dict(vf=layers, pi=layers)])
        # PPO("MlpPolicy",vec_env, **hyperparams,verbose=1)
        if args.verbose:
            vec_env = create_env(env_id,envconfig)
            agent = PPO(MlpPolicy,
                        vec_env, verbose=True, **hyperparams, policy_kwargs=policy_kwargs,tensorboard_log="./logs/"+env_id
                        )
        else:
            vec_env = DummyVecEnv([lambda: create_env(env_id, envconfig)])
            agent = PPO(MlpPolicy,
                        vec_env, verbose=True, **hyperparams, policy_kwargs=policy_kwargs,tensorboard_log="./logs/"+env_id
                        )

        mean_reward, std_reward = evaluate_policy(agent, vec_env, n_eval_episodes=10, render=False)
        print("before train:",mean_reward, std_reward)
        total_timesteps = args.timesteps
        print(total_timesteps)
        agent.learn(
            total_timesteps=int(total_timesteps)
        )
        mean_reward, std_reward = evaluate_policy(agent, vec_env, n_eval_episodes=10, render=False)
        print("after train",mean_reward, std_reward)

        save_agent(env_id,agent)

    if args.mode == 'path':
        print("Environment:"+env_id)
        env = create_env(env_id, envconfig, verbose=True)
        agent_path = os.path.join('Training', 'SavedModels', env_id)
        generate_path(env_id,env,agent_path)


    if args.mode == 'plot':
        plot_path(env_id)

    if args.mode == 'test':
        env_id = 'gym_auv:' + args.scenario
        agent_id = 'gym_auv:' + args.agent
        agent_path = os.path.join('Training', 'SavedModels', agent_id)

        env = create_env(env_id, envconfig, verbose=True)

        generate_path(env_id,env,agent_path)
        env.close()
        # plot_path(env_id)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'mode',
        help='Which program mode to run.',
        choices=['train','path','plot','test'],
    )
    parser.add_argument(
        'scenario',
        help='Which scenario to run.',
        choices=['Env0-v0','Env1-v0','Env2-v0','EmptyScenario-v0'],
    )
    parser.add_argument(
        '--timesteps',
        help='Total timesteps.',
        type=int,
        default=10000
    )
    parser.add_argument(
        '--verbose',
        help='Show debug info',
        type=bool,
        default=True
    )
    parser.add_argument(
        '--agent',
        help='Show debug info',
        choices=['Env0-v0','Env1-v0','EmptyScenario-v0'],
        default= 'Env0-v0',
    )
    parser.add_argument(
        '--savename',
        help='Show debug info',
        default='Test-v0',
    )
    args = parser.parse_args()
    main(args)