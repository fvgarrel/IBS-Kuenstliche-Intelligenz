import gym
from gym.envs.registration import register
from stable_baselines3 import PPO

register(
    id='MinesweeperDiscreet-v0',
    entry_point='minesweeper.minesweeper:MinesweeperDiscreetEnv',
)

# Umgebung erstellen
env = gym.make('MinesweeperDiscreet-v0')

# Modell initialisieren
model = PPO("MlpPolicy", env, verbose=1)

# Modell trainieren
model.learn(total_timesteps=10000)

# Modell speichern
model.save("lunarlander_model")

# Modell laden
loaded_model = PPO.load("lunarlander_model")

# Testen des Modells
obs = env.reset()
for _ in range(1000):
    action, _ = loaded_model.predict(obs, deterministic=True)
    obs, reward, done, _ = env.step(action)
    env.render()
    if done:
        obs = env.reset()

# Umgebung schließen
env.close()