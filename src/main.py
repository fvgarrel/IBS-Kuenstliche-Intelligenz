import gym
from gym.envs.registration import register
from stable_baselines3 import DQN
from stable_baselines3.ppo.policies import MlpPolicy

# Funktion zum Trainieren der KI
def train_agent(env, num_steps):
    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./DQN_tensorboard/",
        policy_kwargs=dict(net_arch=[256, 256]),
        learning_starts=10,
        batch_size=512,
        learning_rate=4e-3,
    )
    model.learn(total_timesteps=num_steps)
    #model = A2C('MlpPolicy', env, verbose=1)
    #model.learn(total_timesteps=num_steps)
    return model

# Funktion zum Testen der KI
def test_agent(model, env, num__test_episodes):
    for ep in range(num_test_episodes):
        obs = env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
            if (rewards == 1000)\
                    :
                print(f"Spiel in Episode {ep} gewonnen.")
                env.render()

# Hauptprogramm
if __name__ == "__main__":
    register(
        id='MinesweeperDiscreet-v0',
        entry_point='minesweeper.minesweeper:MinesweeperDiscreetEnv',
    )
    env = gym.make("MinesweeperDiscreet-v0")
    # das was er trainiert
    num_steps = 100000
    # das was er am Ende spielt
    num_test_episodes = 10000

    print("Training agent...")
    trained_model = train_agent(env, num_steps)

    print("Testing agent...")
    test_agent(trained_model, env, num_test_episodes)