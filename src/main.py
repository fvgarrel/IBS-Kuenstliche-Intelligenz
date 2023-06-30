import gym
from gym.envs.registration import register
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.ppo.policies import MlpPolicy


def get_lr(p: float) -> float:
    lr = 1.0
    if p <= 1.0:
        lr = 2e-3
    if p <= 0.75:
        lr = 3e-4
    if p <= 0.6:
        lr = 2e-4
    if p <= 0.5:
        lr = 1e-4
    if p <= 0.45:
        lr = 1e-3
    if p <= 0.4:
        lr = 1e-4
    return lr


class CustomCallback(BaseCallback):

    def __init__(self, verbose=0):
        super(CustomCallback, self).__init__(verbose)

    def _on_training_start(self) -> None:
        # Startwert der Learning Rate

        self.learning_rate = 2e-3
        self.model.learning_rate = self._get_lr

    def _get_lr(self):
        return self.learning_rate

    def _on_step(self) -> bool:
        # Hier kannst du deine eigene Logik zur Aktualisierung der Learning Rate implementieren

        # Zum Beispiel eine Abhängigkeit von der aktuellen Episode oder der erreichten Belohnung

        # self.learning_rate *= 0.99  # Beispiel: Verringere die Learning Rate um 1% pro Schritt
        if self.num_timesteps == 100000:
            self.learning_rate = 1e-3
        if self.num_timesteps == 200000:
            self.learning_rate = 3e-4
        if self.num_timesteps == 250000:
            self.learning_rate = 5e-5
        if self.num_timesteps == 300000:
            self.learning_rate = 1e-3
        if self.num_timesteps == 350000:
            self.learning_rate = 5e-5
        return True


# Funktion zum Trainieren der KI
def train_agent(env, num_steps):
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./DQN_tensorboard/",
        gamma=0.99,
        gae_lambda=0.95,
        batch_size=64,
        learning_rate=lambda x: get_lr(x)
    )
    callback = CustomCallback()
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
                    env.render()


# Hauptprogramm
if __name__ == "__main__":
    register(
        id='MinesweeperDiscreet-v0',
        entry_point='minesweeper.minesweeper:MinesweeperDiscreetEnv',
    )
    env = gym.make("MinesweeperDiscreet-v0")
    # das was er trainiert
    num_steps = 400000
    # das was er am Ende spielt
    num_episodes = 10000

    print("Training agent...")
    trained_model = train_agent(env, num_steps)
    # trained_model = PPO.load('ppo_1e8.zip')

    print("Testing agent...")
    test_agent(trained_model, env, num_episodes)
