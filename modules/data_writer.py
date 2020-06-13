import os
import json
import shutil
##Added by pankaj
import cv2
import scipy 
##
from google.protobuf.json_format import MessageToJson, MessageToDict
import numpy as np
def _append_extension(filename, ext):
    return filename if filename.lower().endswith(ext.lower()) else filename + ext

def write_json_measurements(episode_path, data_point_id, measurements, control,
                            state):

    with open(os.path.join(episode_path, 'measurements_' + data_point_id.zfill(5) + '.json'), 'w') as fo:
        if state['modelControl'] == True: # Add model intent signals if model is controlling vehicle 
            data = {'steer':control.steer,
                    'throttle': control.throttle,
                     'brake': control.brake,
                     'direction': state['directions'], 
                     'stop_pedestrian': state['stop_pedestrian_pred'],
                     'stop_traffic_lights':state['stop_traffic_lights_pred'],
                     'stop_vehicle': state['stop_vehicle_pred'] ,
                     'speed' : measurements.player_measurements.forward_speed},
                     'model_control': state['modelControl'],
                     'oracle_control': state['oracleControl']        
        else:
            data = {'steer':control.steer,
                    'throttle': control.throttle,
                     'brake': control.brake,
                     'direction': state['directions'], 
                     'stop_pedestrian': state['stop_pedestrian'],
                     'stop_traffic_lights':state['stop_traffic_lights'],
                     'stop_vehicle': state['stop_vehicle'] ,
                     'speed' : measurements.player_measurements.forward_speed,
                     'model_control': state['modelControl'],
                     'oracle_control': state['oracleControl']
                     }         
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
        jsonObj.update({'lateral_noise': settings_module.lat_noise_after})
        jsonObj.update({'longitudinal_noise': settings_module.long_noise_after})
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



def add_data_point(measurements, control, sensor_data, state,
                   dataset_path, episode_number, data_point_id, sensors_frequency):

    episode_path = os.path.join(dataset_path, 'episode_' + episode_number)
    if not os.path.exists(os.path.join(dataset_path, 'episode_' + episode_number)):
        os.mkdir(os.path.join(dataset_path, 'episode_' + episode_number))
    write_sensor_data(episode_path, data_point_id, sensor_data, sensors_frequency)
    write_json_measurements(episode_path, data_point_id, measurements, control,
                            state)

# Delete an episode in the case
def delete_episode(dataset_path, episode_number):
    if os.path.exists(os.path.join(dataset_path, 'episode_' + episode_number)) :
        shutil.rmtree(os.path.join(dataset_path, 'episode_' + episode_number))
    
