import gym
from gym.envs.registration import register
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy

# Funktion zum Trainieren der KI
def train_agent(env, num_steps):
    model = PPO(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=num_steps)
    return model

# Funktion zum Testen der KI
def test_agent(model, env, num_episodes):
    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        total_reward += reward

    print(f"Total reward: {total_reward}")

# Hauptprogramm
if __name__ == "__main__":
    register(
        id='MinesweeperDiscreet-v0',
        entry_point='minesweeper.minesweeper:MinesweeperDiscreetEnv',
    )
    env = gym.make("MinesweeperDiscreet-v0")
    num_steps = 1e9
    num_episodes = 50

    print("Training agent...")
    trained_model = train_agent(env, num_steps)

    print("Testing agent...")
    test_agent(trained_model, env, num_episodes)