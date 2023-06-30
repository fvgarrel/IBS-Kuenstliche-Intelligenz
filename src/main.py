import gym
from gym.envs.registration import register
from stable_baselines3 import DQN
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat

class EnvEvalCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._previous_env = None
        self.tb_formatter = None

    def _on_training_start(self) -> None:
        output_formats = self.logger.output_formats
        self.tb_formatter = next(
            formatter for formatter in output_formats if isinstance(formatter, TensorBoardOutputFormat))

    def _on_rollout_start(self) -> None:
        self._previous_env = None

    def _on_rollout_end(self) -> None:
        # plots the wins or game overs of the AI
        self.tb_formatter.writer.add_scalar(
            "game/state",
            2*int(self.locals["infos"][-1]["is_win"]) - int(self.locals["infos"][-1]["game_over"]),
            self.num_timesteps
        )
        self.logger.record("game/is_win", self.locals["infos"][-1]["is_win"])
        self.logger.record("game/game_over", self.locals["infos"][-1]["game_over"])


    def _on_step(self) -> bool:
        #plots what field the AI clicked on each action
        self.tb_formatter.writer.add_scalar(
            "games/action",
            self.locals["actions"][-1],
            self.num_timesteps
        )

        #plots the total amount of revealed fields for any given action
        board = self.training_env.envs[0].my_board
        self.tb_formatter.writer.add_scalar(
            "reveals/total",
            len(board[(board >= 0)]),
            self.num_timesteps
        )

        #plots the amount of reveals a single action did compared to the prvious board state
        if self._previous_env is not None:
            self.tb_formatter.writer.add_scalar(
                "reveals/delta",
                len(self._previous_env[self._previous_env >= 0]) - len(board[(board >= 0)]),
                self.num_timesteps
            )
        else:
            self.tb_formatter.writer.add_scalar(
                "reveals/delta",
                len(board[(board >= 0)]),
                self.num_timesteps
            )
        self._previous_env = board
        return True


# Funktion zum Trainieren der KI
def train_agent(env, num_steps):
    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./DQN_tensorboard/",
        gamma=0.8,
        batch_size=256,
        learning_rate=1e-3,
        learning_starts=128,
        target_update_interval=100,
    )
    model.learn(total_timesteps=num_steps, callback=EnvEvalCallback())
    # model = A2C('MlpPolicy', env, verbose=1)
    # model.learn(total_timesteps=num_steps)
    model.save("dqn_1e8")
    return model


# Funktion zum Testen der KI
def test_agent(model, env, num_episodes):
    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, rewards, done, info = env.step(action)
            if done and info["is_win"]:
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
    num_steps = 250000
    # das was er am Ende spielt
    num_episodes = 1000

    print("Training agent...")
    trained_model = train_agent(env, num_steps)

    print("Testing agent...")
    test_agent(trained_model, env, num_episodes)
