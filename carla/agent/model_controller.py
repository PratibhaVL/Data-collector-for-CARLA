from __future__ import print_function

import os

import scipy

import tensorflow as tf
import numpy as np




slim = tf.contrib.slim
import warnings
warnings.filterwarnings("ignore")
from carla.agent import Agent
from carla.carla_server_pb2 import Control , SpeedLimitSign 
from carla.carla_server_pb2 import Agent as helper
#from agents.imitation.imitation_learning_network import load_imitation_learning_network
from carla.agent.command_follower import CommandFollower

try:
    import pygame
    from pygame.locals import K_DOWN
    from pygame.locals import K_LEFT
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SPACE
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_d
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_w
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')
numtoCommands = {
                        0.0:  "Follow lane",
                        1.0: "Follow lane",
                        2.0: "Follow lane", 
                        3.0 : "Left", 
                        4.0 : "Right",
                        5.0 : "Straight" }
WINDOW_WIDTH = 100
WINDOW_HEIGHT = 100
MAX_SPEED = 25.0
class ImitationAgent(Agent):

    def __init__(self, city_name = 'Town01', avoid_stopping =True, memory_fraction=0.6, image_cut=[115, 510]):

        #Agent.__init__(self)
        dir_path = os.path.dirname(__file__)
        #self.dropout_vec = [1.0] * 8 + [0.7] * 2 + [0.5] * 2 + [0.5] * 1 + [0.5, 1.] *6
        self.dropout_vec = [1.0] * 8 + [0.7] * 2 + [0.5] * 2 + [0.5] * 1 + [0.5, 1.]*6
        self._image_size = (88, 200, 3)
        self._avoid_stopping = True
        self._image_cut = image_cut
        tf.reset_default_graph() 
        config_gpu = tf.ConfigProto(allow_soft_placement = True)
        # GPU to be selected, just take zero , select GPU  with CUDA_VISIBLE_DEVICES
        config_gpu.gpu_options.visible_device_list = '0'
        config_gpu.gpu_options.per_process_gpu_memory_fraction = memory_fraction
        self._sess = tf.Session(config=config_gpu)
        self._models_path ="/home/pankaj/Trainer_module/trainer11/CARLAILtrainer/models/"#"D:/outbox/changed_old_trainer/trainer5/models/"#dir_path + '/model/'
        self._sess.run(tf.global_variables_initializer())
        #self.load_model()
        with tf.device('/gpu:0'):
            saver = tf.train.import_meta_graph(self._models_path+'model.ckpt.meta')
        self._graph = tf.get_default_graph()
        self._input_images = self._graph.get_tensor_by_name('input_image:0')
        self._input_speed =  self._graph.get_tensor_by_name('input_speed:0')
        self._dout = self._graph.get_tensor_by_name('dropout:0')
        self._follow_lane = self._graph.get_tensor_by_name('Network/Branch_0/fc_8:0') 
        self._left = self._graph.get_tensor_by_name('Network/Branch_1/fc_11:0') 
        self._right = self._graph.get_tensor_by_name('Network/Branch_2/fc_14:0') 
        self._straight = self._graph.get_tensor_by_name('Network/Branch_3/fc_17:0') 
        self._speed = self._graph.get_tensor_by_name('Network/Branch_4/fc_20:0') 
        self._intent = self._graph.get_tensor_by_name('Network/Branch_5/fc_23:0') 
        with tf.device('/gpu:0'):
            saver.restore(self._sess , self._models_path+'model.ckpt')
        self._curr_dir=0
        self.count=0
        self.command_follower = CommandFollower("Town01")

        

    def run_step(self, measurements, sensor_data, directions, target):

        self.state = self.command_follower.get_wp_vectors(measurements, sensor_data, directions, target)
        model_control = self._compute_action(sensor_data['CameraRGB'].data,
                                   measurements.player_measurements.forward_speed, directions) 
        return model_control,self.state

    def _compute_action(self, rgb_image, speed, direction):

        rgb_image = rgb_image[self._image_cut[0]:self._image_cut[1], :]

        image_input = scipy.misc.imresize(rgb_image, [self._image_size[0],
                                                      self._image_size[1]])
        
        image_input = np.multiply(image_input, 1.0 / 255.0)
        
        steer, acc, brake , pred_intents , pred_speed = self._control_function(image_input, speed, direction)


        if acc > brake:
            brake = 0.0
        control= self.command_follower.controller.get_model_control(self.state['wp_angle'], self.state['wp_angle_speed'], min(pred_intents),
                                          speed * 3.6 , steer ,acc , brake )
        
        control.hand_brake = 0
        control.reverse = 0
        #print(f"Controls: \n Steer: {control.steer} \tThrottle: {control.throttle} \tBrake: {control.brake} \nCommand: {numtoCommands[direction]} ")
        #print(f" True ped_intent: {self.state['stop_pedestrian']}\t True veh_intent: {self.state['stop_vehicle']} \t True tra_intent: {self.state['stop_traffic_lights']}")
        ped_intent , tra_intent , veh_intent = pred_intents
        #print(f"Pred_ped_intent: {ped_intent} \tPred_veh_intent: {veh_intent} \tPred_tra_intent: {tra_intent} ")
        self.state.update({"stop_pedestrian_pred": ped_intent,
                    "stop_vehicle_pred":veh_intent,
                    "stop_traffic_lights_pred":tra_intent})
        return control

    def _control_function(self, image_input, speed, control_input):

        
        image_input = image_input.reshape(
            (1, self._image_size[0], self._image_size[1], self._image_size[2]))
        # Normalize with the maximum speed from the training set ( 90 km/h)
        curr_speed = speed

        speed = np.array(float(speed) / MAX_SPEED)
        speed = speed.reshape((1, 1))
        #print(f"Current Speed: ", speed *3.6 *MAX_SPEED)
        if control_input == 2 or control_input == 0.0:
            branch = self._follow_lane
        elif control_input == 3:
            branch = self._left
        elif control_input == 4:
            branch = self._right
        else:
            branch = self._straight
        

        feedDict = {self._input_images: image_input , self._input_speed:speed , self._dout: [1]*len(self.dropout_vec)}
        
        output , predicted_speed ,pred_intent = self._sess.run([branch,self._speed , self._intent] ,feed_dict=feedDict)
        os.system('clear')
        predicted_steers ,predicted_acc, predicted_brake = output[0]
        predicted_speed = predicted_speed[0]
        for i in range(len(pred_intent[0])):
            if pred_intent[0][i] >0.9:
               pred_intent[0][i] = 1   
            elif pred_intent[0][i] <= 0.05:
               pred_intent[0][i] = 0
        if predicted_speed[0] > 0.3 and curr_speed < 0.1: #False braking
            #print("False braking !!!!")
            predicted_acc = 0.5
            predicted_brake = 0
        #if (tra_intent - self.state['stop_traffic_lights'])  > 0.7:
        #    print(self.state['stop_traffic_lights'],  tra_intent)
        #    self.traffic_light_infraction = True
        return predicted_steers, predicted_acc, predicted_brake , pred_intent[0] , predicted_speed[0]*MAX_SPEED *3.6
