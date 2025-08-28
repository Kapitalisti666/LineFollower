from linefollower_env import LineFollowerEnv
from stable_baselines3 import PPO

if __name__ == "__main__":
    episodes = 10
    env = LineFollowerEnv()
    model = PPO('MlpPolicy', env, verbose = 1,tensorboard_log = "./test/")
    #model.load(path="C:/Users/teemu/Future_of_iot/line_follower_rl/ai_gym_train/gym_linefollower/ppo_models/ppo_model_25_10_2022_12_06_28.zip", env = env)
    model.set_parameters("ppo_model_22_02_2023_21_42_38.zip")
    for episode in range(1, episodes+1):
        observation = env.reset()
        done = False
        score = 0 

        while not done:
            action = model.predict(observation)
            observation, reward, done, info = env.step(int(action[0]))
            env.render()
            score+=reward
            print('Episode:{} Score:{}'.format(episode, score))