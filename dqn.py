import time
import gymnasium as gym
from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

def create_training_env():
    env = gym.make("PongNoFrameskip-v4")
    # AtariPreprocessing does grayscale, resize, and frame skip but NOT frame stacking
    env = AtariPreprocessing(env, frame_skip=4, grayscale_obs=True, scale_obs=True)
    return env

def create_render_env():
    env = gym.make("PongNoFrameskip-v4", render_mode='human')
    env = AtariPreprocessing(env, frame_skip=4, grayscale_obs=True, scale_obs=True)
    return envcs

if __name__ == "__main__":
    # Create a vectorized training environment
    train_env = DummyVecEnv([create_training_env])
    # Now wrap it with VecFrameStack to stack 4 frames
    train_env = VecFrameStack(train_env, n_stack=4)

    # Create the DQN model
    model = DQN(
        "CnnPolicy",
        train_env,
        verbose=1,
        learning_rate=1e-4,
        buffer_size=100000,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        target_update_interval=1000,
        train_freq=4,
        gamma=0.99,
        batch_size=32
    )

    # Train the model
    model.learn(total_timesteps=int(1e5))
    model.save("dqn_pong_model")

    model = DQN.load("dqn_pong_model")

    # For rendering: create a dummy vec env, stack frames, and watch the agent
    # Note: With rendering on, speed may be slower
    render_env = DummyVecEnv([create_render_env])
    render_env = VecFrameStack(render_env, n_stack=4)

    # Run a few episodes and watch the agent
    for _ in range(5):
        obs, info = render_env.reset()
        terminated, truncated = [False], [False]
        while not (terminated[0] or truncated[0]):
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, terminated, truncated, info = render_env.step(action)
            time.sleep(0.01)  # slow down gameplay for viewing

    render_env.close()
