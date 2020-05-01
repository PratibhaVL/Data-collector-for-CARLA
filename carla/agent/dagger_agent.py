from __future__ import print_function

import os
import scipy
import math

import tensorflow as tf
import numpy as np

slim = tf.contrib.slim

from carla.agent import Agent
from carla.carla_server_pb2 import Control , SpeedLimitSign 
#from carla.carla_server_pb2 import Agent as helper

numtoCommands = {
                        0.0:  "Follow lane",
                        1.0: "Follow lane",
                        2.0: "Follow lane", 
                        3.0 : "Left", 
                        4.0 : "Right",
                        5.0 : "Straight" }
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
class DaggerAgent(Agent):

    def __init__(self, city_name, avoid_stopping = False, memory_fraction=0.5, image_cut=[115, 510]):

        Agent.__init__(self)
        dir_path = os.path.dirname(__file__)
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
        self._models_path = dir_path + "/models/"#"D:/outbox/changed_old_trainer/trainer5/models/"#dir_path + '/model/'
        print(self._models_path )
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
        #self._speed = self._graph.get_tensor_by_name('Network/Branch_4/fc_20:0') 
        #self._intent = self._graph.get_tensor_by_name('Network/Branch_5/fc_23:0') 
        with tf.device('/gpu:0'):
            saver.restore(self._sess , self._models_path+'model.ckpt')
        self._curr_dir=0
        
    

    def run_step(self, measurements, sensor_data, directions, target = None):
               
        
        
        model_control = self._compute_action(sensor_data['CameraRGB'].data,
                                       measurements.player_measurements.forward_speed, directions ) 
        
        
        return model_control
        

    def _compute_action(self, rgb_image, speed, direction):

        rgb_image = rgb_image[self._image_cut[0]:self._image_cut[1], :]

        image_input = scipy.misc.imresize(rgb_image, [self._image_size[0],
                                                      self._image_size[1]])
        
        image_input = np.multiply(image_input, 1.0 / 255.0)
        
        
        steer, acc, brake= self._control_function(image_input, speed, direction)
        print(direction)
        if self._curr_dir != direction:
            print(f"direction: {numtoCommands[direction]} ")
            self._curr_dir = direction
        if acc > brake:
            brake = 0.0
        control = Control()
        if steer > 0:
            steer = min(steer, 1)
        else:
            steer = max(steer, -1)


        control.steer = steer
        control.throttle = acc
        control.brake = brake

        control.hand_brake = 0
        control.reverse = 0

        return control

    def _control_function(self, image_input, speed, control_input):

        
        image_input = image_input.reshape(
            (1, self._image_size[0], self._image_size[1], self._image_size[2]))
        # Normalize with the maximum speed from the training set ( 90 km/h)
        curr_speed = speed
        speed = np.array(speed / 10)
        speed = speed.reshape((1, 1))
        #print("control_input: ",control_input)
        if control_input == 2 or control_input == 0.0:
            branch = self._follow_lane
        elif control_input == 3:
            branch = self._left
        elif control_input == 4:
            branch = self._right
        else:
            branch = self._straight
        
        feedDict = {self._input_images: image_input , self._input_speed:speed , self._dout: [1]*len(self.dropout_vec)}
        output  = self._sess.run(branch ,feed_dict=feedDict)
        predicted_steers, predicted_acc, predicted_brake = output[0]
        return predicted_steers, predicted_acc, predicted_brake 
