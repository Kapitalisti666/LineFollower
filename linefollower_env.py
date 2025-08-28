import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy
import time
import os

import cv2

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from pybullet_client import PybulletClient
from track_load import TrackLoad
from linefollower_bot import LineFollowerBot
from stable_baselines3.common.env_checker import check_env



class LineFollowerEnv(gym.Env):

    def __init__(self, frame_stacking = 4, state_type = "raw", line_mode = "basic"):
        gym.Env.__init__(self)

        self.dt = 1.0/200.0
        self.pi = 3.141592654

        self.state_type = state_type
        self.line_mode  = line_mode
       

        self.models_path = os.path.dirname(__file__)
        if len(self.models_path) <= 0:
            self.models_path = "."
      
        if self.state_type == "raw": 
            self.observation_space = spaces.MultiBinary(3)

        self.action_space = spaces.Discrete(3)

        self.actions = []  
        
        self.actions.append([0.0, 0.3])
        self.actions.append([0.3, 0.3]) 
        self.actions.append([0.3, 0.0]) 

        self.observation_memory = numpy.zeros(3)

        self.pb_client = PybulletClient()
        self.reset()


    def reset(self):
        self.pb_client.resetSimulation()

        self.pb_client.setGravity(0, 0, -9.81)
        self.pb_client.setTimeStep(self.dt)


        if self.line_mode == "advanced":
            track_idx = numpy.random.randint(32)
            self.line = TrackLoad(self.pb_client, self.models_path + "/models_tracks/" + str(track_idx))
        else:
            self.line = TrackLoad(self.pb_client, self.models_path + "/models/track_plane_template")
        

        starting_position = self.line.get_start_random()

        self.bot = LineFollowerBot(self.pb_client, self.models_path + "/models/robot_simple.urdf", starting_position = starting_position)


        for i in range(100):
            self.pb_client.stepSimulation()

        self.steps = 0

        self.observation = None
        self.reward      = 0.0
        self.done        = False
        self.info        = {}

        self.line_polygon = Polygon(self.line.points)

        self.observation = self._update_observation()
        return self.observation

    def step(self, action):
        left_power_target, right_power_target = self.actions[action]
        return self.step_continuous(left_power_target, right_power_target)

    def step_continuous(self, left_power_target, right_power_target):
        self.steps+= 1

        robot_x, robot_y, robot_z, pitch, roll, yaw = self.bot.get_position()

        self.bot.set_throttle(left_power_target, right_power_target)
   
        self.pb_client.stepSimulation()

        closest_idx, closest_distance, closest_coord = self.line.get_closest(robot_x, robot_y)

        self.done   = False
        self.reward = 0.0

        #too many time steps
        if self.steps > 16384:
            self.done = True

        if closest_distance > 0.1:
            self.done  = True

        self.observation = self._update_observation()

        if str(self.observation) == "[0. 1. 0.]":
            self.reward = 1
        elif str(self.observation) == "[1. 1. 1.]":
            self.reward = -0.5 
        elif str(self.observation) == "[1. 1. 0.]":
            self.reward = 0.8
        elif str(self.observation) == "[0. 1. 1.]":
            self.reward = 0.8
        elif str(self.observation) == "[1. 0. 0.]":  
            self.reward = 0.4
        elif str(self.observation) == "[0. 0. 1.]":  
            self.reward = 0.4
        else:
            self.reward = -1
            
        return self.observation, self.reward, self.done, self.info
        
    def render(self, mode = None):
        if self.steps%4 == 0:
            robot_x, robot_y, robot_z, pitch, roll, yaw = self.bot.get_position()

            width  = 256
            height = 256
            
            #top view
            top_view = self.bot.get_image(yaw*180.0/self.pi - 90, -90.0, 0.0, 0.25, robot_x, robot_y, robot_z, width = width, height = height)

            #third person view
            dist = 0.02
            tp_view = self.bot.get_image(yaw*180.0/self.pi - 90, -40.0, 0.0, 0.1, robot_x+dist*numpy.cos(yaw), robot_y+dist*numpy.sin(yaw), robot_z, width = width, height = height, fov=100)

            #camera view
            cam_view = self._get_camera_view()

            dist = 0.02
            side_view = self.bot.get_image(yaw*180.0/self.pi - 0, -40.0, 0.0, 0.1, robot_x+dist*numpy.cos(yaw), robot_y+dist*numpy.sin(yaw), robot_z, width = width, height = height, fov=100)

            separator_width = 2
            vertical_separator   = numpy.ones((height, separator_width, 3))*0.5
            horizontal_separator = numpy.ones((separator_width, width*2 + separator_width, 3))*0.5

            image_a = numpy.hstack([ numpy.hstack([top_view, vertical_separator]), tp_view])
            image_b = numpy.hstack([ numpy.hstack([cam_view, vertical_separator]), side_view])

            image = numpy.vstack([numpy.vstack([image_a, horizontal_separator]), image_b] )
            
            image = numpy.clip(255*image, 0.0, 255.0)

            image = numpy.array(image, dtype=numpy.uint8)
            self._draw_fig(image)
        
    def _draw_fig(self, image):
        rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

        cv2.imshow("line follower", rgb)  
        cv2.waitKey(1)

    def _get_camera_view(self, width = 256, height = 256):
        robot_x, robot_y, robot_z, pitch, roll, yaw = self.bot.get_position()
        r  = 0.15
        cam_dx = r*numpy.cos(yaw)
        cam_dy = r*numpy.sin(yaw)
        return self.bot.get_image(yaw*180.0/self.pi - 90, -60.0, 0.0, 0.025, robot_x + cam_dx, robot_y + cam_dy, robot_z + 0.04, width = width, height = height, fov=120)
    
    def _update_observation(self):
        if self.state_type == "raw":

            left_sensor_pos, middle_sensor_pos, right_sensor_pos = self.bot.get_sensor_positions()
            line_position, linepoints = self._get_line_position(0.0)

            observation = numpy.zeros(3)

            middle_points_x = [round(middle_sensor_pos[0], 2), round(middle_sensor_pos[0], 2) + 0.01, round(middle_sensor_pos[0], 2) - 0.01]
            middle_points_y = [round(middle_sensor_pos[1], 2), round(middle_sensor_pos[1], 2) + 0.01, round(middle_sensor_pos[1], 2) - 0.01]

            left_points_x = [round(left_sensor_pos[0], 2), round(left_sensor_pos[0], 2) + 0.01, round(left_sensor_pos[0], 2) - 0.01]
            left_points_y = [round(left_sensor_pos[1], 2), round(left_sensor_pos[1], 2) + 0.01, round(left_sensor_pos[1], 2) - 0.01]

            right_points_x = [round(right_sensor_pos[0], 2), round(right_sensor_pos[0], 2) + 0.01, round(right_sensor_pos[0], 2) - 0.01]
            right_points_y = [round(right_sensor_pos[1], 2), round(right_sensor_pos[1], 2) + 0.01, round(right_sensor_pos[1], 2) - 0.01]
            
            for point in linepoints:
                if round(point[0], 2) in left_points_x and round(point[1], 2) in left_points_y:
                    print("LEFT")
                    observation[0] = 1 
                if round(point[0], 2) in middle_points_x and round(point[1], 2) in middle_points_y:
                    print("MIDDLE")
                    observation[1] = 1
                if round(point[0], 2) in right_points_x and round(point[1], 2) in right_points_y:
                    print("RIGHT")
                    observation[2] = 1
            print(observation)
        return observation

    def _get_line_position(self, sensor_distance = 0.04):

        x, y, _, _, _, yaw = self.bot.get_position()
        x_ = x + sensor_distance*numpy.cos(yaw)
        y_ = y + sensor_distance*numpy.sin(yaw) 

        _, distance, line_points  = self.line.get_closest(x_, y_)

        if self.line_polygon.contains(Point(x_, y_)):
            line_position = 1.0*distance
        else:
            line_position = -1.0*distance
        return line_position, line_points

if __name__ == "__main__":

    env = LineFollowerEnv()
    #check_env(env)
    env.reset()
    env.render()
    
    while True:
        action = env.action_space.sample()
        state, reward, done, _ = env.step(action)
        env.render()
		
        if done:
            env.reset()
        print("reward = ", reward)
        print("action = ", action)
        print("state = ", state)
        #breakpoint()
    