import os
import json
import shutil
import h5py
import numpy as np
from google.protobuf.json_format import MessageToJson, MessageToDict
import scipy

FILE_SIZE = 200
IMAGE_SIZE = [88,200,3]
MEASUREMENTS_SIZE = 8
RGB = np.zeros((FILE_SIZE , IMAGE_SIZE[0],IMAGE_SIZE[1],IMAGE_SIZE[2]) , dtype = np.uint8)
MEASUREMENTS = np.zeros((FILE_SIZE , MEASUREMENTS_SIZE ) , dtype = np.float32) 
fileCounter = 0
image_cut = [115, 510]
def update_measurements( data_point_id, measurements, control, control_noise,
                            state):
    
    global MEASUREMENTS
    MEASUREMENTS[data_point_id][0] = control.steer
    MEASUREMENTS[data_point_id][1] = control.throttle
    MEASUREMENTS[data_point_id][2] = control.brake
    MEASUREMENTS[data_point_id][3] = measurements.player_measurements.forward_speed
    MEASUREMENTS[data_point_id][4] = state['directions']
    MEASUREMENTS[data_point_id][5] = state['stop_pedestrian']
    MEASUREMENTS[data_point_id][6] = state['stop_vehicle']
    MEASUREMENTS[data_point_id][7] = state['stop_traffic_lights']
    
    
    
    
    
    '''with open(os.path.join(episode_path, 'measurements_' + data_point_id.zfill(5) + '.json'), 'w') as fo:
            
                    jsonObj = MessageToDict(measurements)
                    jsonObj.update(state)
                    jsonObj.update({'steer': control.steer})
                    jsonObj.update({'throttle': control.throttle})
                    jsonObj.update({'brake': control.brake})
                    jsonObj.update({'hand_brake': control.hand_brake})
                    jsonObj.update({'reverse': control.reverse})
                    jsonObj.update({'steer_noise': control_noise.steer})
                    jsonObj.update({'throttle_noise': control_noise.throttle})
                    jsonObj.update({'brake_noise': control_noise.brake})
            
                    fo.write(json.dumps(jsonObj, sort_keys=True, indent=4))'''


def update_sensor_data( data_point_id, sensor_data, sensors_frequency):
    
    global RGB
    rgb_image = sensor_data['CameraRGB'].data
    rgb_image = rgb_image[image_cut[0]:image_cut[1], :]
    rgb_image= scipy.misc.imresize(rgb_image, [IMAGE_SIZE[0],IMAGE_SIZE[1]])
    RGB [data_point_id]= rgb_image


def make_dataset_path(dataset_path):
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)


def add_metadata(dataset_path, settings_module):
    with open(os.path.join(dataset_path, 'metadata.json'), 'w') as fo:
        jsonObj = {}
        jsonObj.update(settings_module.sensors_yaw)
        jsonObj.update({'fov': settings_module.FOV})
        jsonObj.update({'width': settings_module.WINDOW_WIDTH})
        jsonObj.update({'height': settings_module.WINDOW_HEIGHT})
        jsonObj.update({'lateral_noise_percentage': settings_module.lat_noise_percent})
        jsonObj.update({'longitudinal_noise_percentage': settings_module.long_noise_percent})
        jsonObj.update({'car range': settings_module.NumberOfVehicles})
        jsonObj.update({'pedestrian range': settings_module.NumberOfPedestrians})
        jsonObj.update({'set_of_weathers': settings_module.set_of_weathers})
        fo.write(json.dumps(jsonObj, sort_keys=True, indent=4))

def add_episode_metadata(dataset_path, episode_number, episode_aspects):

    if not os.path.exists(os.path.join(dataset_path, 'episode_' + episode_number)):
        os.mkdir(os.path.join(dataset_path, 'episode_' + episode_number))

    with open(os.path.join(dataset_path, 'episode_' + episode_number, 'metadata.json'), 'w') as fo:

        jsonObj = {}
        jsonObj.update({'number_of_pedestrian': episode_aspects['number_of_pedestrians']})
        jsonObj.update({'number_of_vehicles': episode_aspects['number_of_vehicles']})
        jsonObj.update({'seeds_pedestrians': episode_aspects['seeds_pedestrians']})
        jsonObj.update({'seeds_vehicles': episode_aspects['seeds_vehicles']})
        jsonObj.update({'weather': episode_aspects['weather'], 
                        'goal': episode_aspects['episode_points'] , 
                        'time_taken': episode_aspects['time_taken']})

        fo.write(json.dumps(jsonObj, sort_keys=True, indent=4))



def add_data_point(measurements, control, control_noise, sensor_data, state,
                   dataset_path, episode_number, data_point_id, sensors_frequency):
    
    if fileCounter:
        data_point_id = data_point_id % (FILE_SIZE * fileCounter)
    
    update_sensor_data( data_point_id, sensor_data, sensors_frequency)
    update_measurements( data_point_id, measurements, control, control_noise,
                            state)
    if data_point_id == FILE_SIZE -1:
        writeh5(dataset_path , episode_number ,data_point_id)

def writeh5(dataset_path , episode_number ,data_point_id):
    global fileCounter
    global RGB 
    global MEASUREMENTS 
    if fileCounter: # For end of episode case
        data_point_id = data_point_id % (FILE_SIZE * fileCounter)

    episode_path = os.path.join(dataset_path, 'episode_' + episode_number)
    if not os.path.exists(os.path.join(dataset_path, 'episode_' + episode_number)):
        os.mkdir(os.path.join(dataset_path, 'episode_' + episode_number))

    if data_point_id < FILE_SIZE -1: # if episode ended 
        RGB = np.resize(RGB, (data_point_id , IMAGE_SIZE[0],IMAGE_SIZE[1],IMAGE_SIZE[2]))
        MEASUREMENTS = np.resize(MEASUREMENTS, (data_point_id , MEASUREMENTS_SIZE))

    with h5py.File(os.path.join(episode_path ,"episode_" +str(episode_number) + "_" +str(fileCounter)+".h5"), "w") as f:
        f.create_dataset('rgb', data=RGB)
        f.create_dataset('targets', data= MEASUREMENTS)
    fileCounter+=1
    RGB = np.zeros((FILE_SIZE , IMAGE_SIZE[0],IMAGE_SIZE[1],IMAGE_SIZE[2]) , dtype = np.uint8)
    MEASUREMENTS = np.zeros((FILE_SIZE , MEASUREMENTS_SIZE ) , dtype = np.float32) 

def reset_file_counter():
    global fileCounter
    fileCounter=0
    
# Delete an episode in the case
def delete_episode(dataset_path, episode_number):

    shutil.rmtree(os.path.join(dataset_path, 'episode_' + episode_number))