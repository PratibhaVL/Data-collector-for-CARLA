from carla import sensor
from carla.settings import CarlaSettings
"""
Example of dataset configuration. See a more complex example at coil_training_dataset.py
"""
# The image size definition for the cameras
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

#Positions a vector of tuples containing the [START, END] positions
#of the episodes that the (expert driver)[docs/agent_module.md] is going to follow. The
#possible positions for agent placement can be viewed by running
POSITIONS = [[95, 70],
 [95, 104],
 [30, 41],
 [102, 95],
 [132, 27],
 [78, 44],
 [21, 16],
 [73, 31],
 [27, 130],
 [41, 102],
 [47, 79],
 [18, 107],
 [79, 45],
 [37, 81],
 [68, 50],
 [69, 45],
 [140, 134],
 [0, 17],
 [65, 18],
 [68, 44],
 [110, 15],
 [34, 67],
 [4, 129],
 [84, 34],
 [148, 129],
 [111, 64],
 [18, 145],
 [78, 70],
 [53, 67],
 [84, 69],
 [130, 17],
 [95, 102],
 [105, 29],
 [102, 87],
 [68, 85],
 [68, 129]]

# The FOV for all the cameras
FOV = 100
# The frequency where all the the camera will be colleted. 1 means every frame, 0.5 every two frames
sensors_frequency = {'CameraRGB': 1}
# The yaw of every sensor.
sensors_yaw = {'CameraRGB': 0}
# The percentage of episodes to have lateral noise. More information about the noise can
# be found at docs/agent_module.md
lat_noise_after = 0
# The percentage of episodes with longitudinal noise.
long_noise_after = 0
# The interval of vehicles/pedestrians that every episode can have
NumberOfVehicles = [0, 20]  # The range for the random numbers that are going to be generated
NumberOfPedestrians = [40, 60]

set_of_weathers = [1]

"""
Returns the entire CarlaSettings to be used on all the episodes.
Here we are defining the cameras used. The number of vehicles and pedestrians will 
be sampled from the interval defined on the NumberOfVehicle and NumberOfPedestrians variables.
This function must be redefined on each of the daset cofiguration files.
"""

def make_carla_settings(seeds_vehicles = None , seeds_pedestrians = None):
    """Make a CarlaSettings object with the settings we need."""

    settings = CarlaSettings()
    settings.set(
        SendNonPlayerAgentsInfo=True,
        SynchronousMode=True)
    settings.set(DisableTwoWheeledVehicles=True)

    settings.randomize_seeds(seeds_vehicles ,seeds_pedestrians)
    # Add a carla camera.
    camera0 = sensor.Camera('CameraRGB')
    camera0.set_image_size(WINDOW_WIDTH, WINDOW_HEIGHT)
    camera0.set(FOV=FOV)
    camera0.set_position(2.0, 0.0, 1.4)
    camera0.set_rotation(-15.0, 0, 0)
    settings.add_sensor(camera0)

    return settings

