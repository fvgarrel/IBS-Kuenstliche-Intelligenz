import gym
from gym.envs.registration import register
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.ppo.policies import MlpPolicy


def get_lr(p: float) -> float:
    lr = 1.0
    p = 1-p
    if p <= 1.0:
        lr = 3e-4
    if p <= 0.9:
        lr = 25e-5
    if p <= 0.75:
        lr = 2e-4
    if p <= 0.5:
        lr = 15e-5
    if p <= 0.4:
        lr = 1e-4
    if p <= 0.2:
        lr = 5e-5
    return lr


# Funktion zum Trainieren der KI
def train_agent(env, num_steps):
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./DQN_tensorboard/",
        #gae_lambda=0.99,
        policy_kwargs=dict(net_arch=[256, 256]),
        learning_rate=lambda x: get_lr(x)
    )
    checkpointCallback = CheckpointCallback(save_freq=25000, save_path='./ppo_checkpoint/', name_prefix='model',
                                            verbose=1)
    shit = CallbackList([checkpointCallback])
    model.learn(total_timesteps=num_steps, callback=shit)
    # model = A2C('MlpPolicy', env, verbose=1)
    # model.learn(total_timesteps=num_steps)
    return model


# Funktion zum Testen der KI
def test_agent(model, env, num_episodes):
    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
            if done:
                if (rewards > 0) \
                        :
                    print(f"Spiel in Episode {ep} gewonnen.")
                    #env.render()


# Hauptprogramm
if __name__ == "__main__":
    register(
        id='MinesweeperDiscreet-v0',
        entry_point='minesweeper.minesweeper:MinesweeperDiscreetEnv',
    )
    env = gym.make("MinesweeperDiscreet-v0")
    # das was er trainiert
    num_steps = 1000000
    # das was er am Ende spielt
    num_episodes = 10000

    print("Training agent...")
    trained_model = train_agent(env, num_steps)
    # trained_model = PPO.load('ppo_1e8.zip')

    print("Testing agent...")
    test_agent(trained_model, env, num_episodes)
