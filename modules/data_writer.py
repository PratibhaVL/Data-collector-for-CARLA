import os
import json
import shutil
import h5py
import numpy as np
from google.protobuf.json_format import MessageToJson, MessageToDict
import scipy
import cv2

FILE_SIZE = 200
IMAGE_SIZE = [88,200,3]
TARGETS_SIZE = 28
RGB = np.zeros((FILE_SIZE , IMAGE_SIZE[0],IMAGE_SIZE[1],IMAGE_SIZE[2]) , dtype = np.uint8)
TARGETS = np.zeros((FILE_SIZE , TARGETS_SIZE ) , dtype = np.float32) 
fileCounter = 0
image_cut = [115, 510]
WRITE_H5 = False
def _append_extension(filename, ext):
    return filename if filename.lower().endswith(ext.lower()) else filename + ext
def update_measurements( data_point_id, measurements, control, control_noise,
                            state):
    
    global TARGETS
    TARGETS[data_point_id][0] = control.steer
    TARGETS[data_point_id][1] = control.throttle
    TARGETS[data_point_id][2] = control.brake
    TARGETS[data_point_id][3] = measurements.game_timestamp /1000
    TARGETS[data_point_id][10] = measurements.player_measurements.forward_speed
    TARGETS[data_point_id][24] = state['directions']
    TARGETS[data_point_id][25] = state['stop_pedestrian']
    TARGETS[data_point_id][26] = state['stop_vehicle']
    TARGETS[data_point_id][27] = state['stop_traffic_lights']
 

def update_sensor_data( data_point_id, sensor_data, sensors_frequency):
    
    global RGB
    rgb_image = sensor_data['CameraRGB'].data
    rgb_image = rgb_image[image_cut[0]:image_cut[1], :]
    rgb_image= scipy.misc.imresize(rgb_image, [IMAGE_SIZE[0],IMAGE_SIZE[1]])
    RGB [data_point_id]= rgb_image

def write_json_measurements(episode_path, data_point_id, measurements, control, control_noise,
                            state):

    with open(os.path.join(episode_path, 'measurements_' + data_point_id.zfill(5) + '.json'), 'w') as fo:
        data = {'steer':control.steer,
                'throttle': control.throttle,
                 'brake': control.brake,
                 'direction': state['directions'], 
                 'stop_pedestrian': state['stop_pedestrian'],
                 'stop_traffic_lights':state['stop_traffic_lights'],
                 'stop_vehicle': state['stop_vehicle'] ,
                 'speed' : measurements.player_measurements.forward_speed}
                     
        fo.write(json.dumps(data, sort_keys=True, indent=4))


def write_sensor_data(episode_path, data_point_id, sensor_data, sensors_frequency):
    try:
        from PIL import Image as PImage
    except ImportError:
        raise RuntimeError(
            'cannot import PIL, make sure pillow package is installed')

    name = 'CameraRGB'
    filename = _append_extension( os.path.join(episode_path, name + '_' + data_point_id.zfill(5)), '.png')
    img = sensor_data['CameraRGB'].data   
    img = img[115:510,:]
    img = scipy.misc.imresize(img , [88 , 200])
    folder = os.path.dirname (filename)
    if not os.path.isdir(folder):
        os.makedirs(folder)
    cv2.imwrite(filename ,cv2.cvtColor(img, cv2.COLOR_RGB2BGR) )

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
                        'expert_points': episode_aspects['expert_points'] ,
                        'time_taken': episode_aspects['time_taken'],
                        'episode_lateral_noise': episode_aspects ['episode_lateral_noise'],
                        'episode_longitudinal_noise':episode_aspects ['episode_longitudinal_noise']
                        })

        fo.write(json.dumps(jsonObj, sort_keys=True, indent=4))



def add_data_point(measurements, control, control_noise, sensor_data, state,
                   dataset_path, episode_number, data_point_id, sensors_frequency):
    #if WRITE_H5:
    episode_path = os.path.join(dataset_path, 'episode_' + episode_number)
    if not os.path.exists(os.path.join(dataset_path, 'episode_' + episode_number)):
        os.mkdir(os.path.join(dataset_path, 'episode_' + episode_number))
    write_sensor_data(episode_path, str(data_point_id), sensor_data, sensors_frequency)
    write_json_measurements(episode_path, str(data_point_id), measurements, control, control_noise,
                            state)
    '''if fileCounter:
                    data_point_id = data_point_id % (FILE_SIZE * fileCounter)
                
                update_sensor_data( data_point_id, sensor_data, sensors_frequency)
                update_measurements( data_point_id, measurements, control, control_noise,
                                        state)
                if data_point_id == FILE_SIZE -1:
                    writeh5(dataset_path , episode_number ,data_point_id)'''
    #else:
    

def writeh5(dataset_path , episode_number ,data_point_id):
    global fileCounter
    global RGB 
    global TARGETS 
    #if fileCounter: # For end of episode case
    #    data_point_id = data_point_id % (FILE_SIZE * fileCounter)

    episode_path = os.path.join(dataset_path, 'episode_' + episode_number)
    if not os.path.exists(os.path.join(dataset_path, 'episode_' + episode_number)):
        os.mkdir(os.path.join(dataset_path, 'episode_' + episode_number))

    if data_point_id < FILE_SIZE -1: # if episode ended 
        RGB = np.resize(RGB, (data_point_id , IMAGE_SIZE[0],IMAGE_SIZE[1],IMAGE_SIZE[2]))
        TARGETS = np.resize(TARGETS, (data_point_id , TARGETS_SIZE))

    with h5py.File(os.path.join(episode_path ,"episode_" +str(episode_number) + "_" +str(fileCounter)+".h5"), "w") as f:
        f.create_dataset('rgb', data=RGB)
        f.create_dataset('targets', data= TARGETS)
    fileCounter+=1
    RGB = np.zeros((FILE_SIZE , IMAGE_SIZE[0],IMAGE_SIZE[1],IMAGE_SIZE[2]) , dtype = np.uint8)
    TARGETS = np.zeros((FILE_SIZE , TARGETS_SIZE ) , dtype = np.float32) 

def reset_file_counter():
    global fileCounter
    fileCounter=0
def get_file_counter():
    global fileCounter
    return fileCounter

# Delete an episode in the case
def delete_episode(dataset_path, episode_number):
    episode_path = os.path.join(dataset_path, 'episode_' + episode_number)
    if os.path.exists(episode_path):
        shutil.rmtree(episode_path)