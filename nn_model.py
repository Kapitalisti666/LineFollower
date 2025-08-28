import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from linefollower_env import LineFollowerEnv
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
import time
from datetime import datetime



def build_model(actions):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(3,activation='relu', input_shape = (1,3)),
        tf.keras.layers.Dense(6,activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(actions, activation='linear')
    ])
    return model

def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy, 
                  nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2)
    return dqn

if __name__ == "__main__":
    now = datetime.now()
    current_time = now.strftime("%d_%m_%Y_%H_%M_%S")
    env = LineFollowerEnv()
    actions = env.action_space.n
    model = build_model(actions)
    dqn = build_agent(model, actions)
    dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])
    #dqn.load_weights("C:/Users/teemu/Future_of_iot/line_follower_rl/ai_gym_train/gym_linefollower/weights/weights_21_10_2022_13_00_01.h5")
    dqn.fit(env, nb_steps=50000, visualize=True, verbose=1)
    dqn.save_weights(f"C:/Users/teemu/Future_of_iot/line_follower_rl/ai_gym_train/gym_linefollower/weights/weights_{current_time}.h5", overwrite=False)
    print("Training finished")
    time.sleep(10)
    breakpoint()
    print("Starting to test")
    time.sleep(5)
    dqn.test(env, nb_episodes=100, visualize=True)