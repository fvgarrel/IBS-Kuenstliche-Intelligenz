import gym
from gym.envs.registration import register
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy

# Funktion zum Trainieren der KI
def train_agent(env, num_steps):
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./DQN_tensorboard/",
        gamma = 0.99,
        gae_lambda = 0.95,
        batch_size = 64,
        learning_rate=1e-4
    )
    model.learn(total_timesteps=num_steps)
    #model = A2C('MlpPolicy', env, verbose=1)
    #model.learn(total_timesteps=num_steps)
    model.save("ppo_1e8")
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
                if (rewards >0)\
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
    num_steps = 1000000
    # das was er am Ende spielt
    num_episodes = 1000

    print("Training agent...")
    trained_model = train_agent(env, num_steps)

    print("Testing agent...")
    test_agent(trained_model, env, num_episodes)