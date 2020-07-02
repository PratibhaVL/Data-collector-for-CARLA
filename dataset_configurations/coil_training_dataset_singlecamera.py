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
POSITIONS = [[61, 49],
 [70, 73],
 [127, 71],
 [64, 111],
 [110, 62],
 [91, 85],
 [77, 68],
 [43, 60],
 [133, 50],
 [116, 42],
 [114, 93],
 [5, 57],
 [66, 3],
 [52, 101],
 [145, 16],
 [79, 14],
 [47, 79],
 [129, 148],
 [85, 91],
 [40, 33],
 [74, 71],
 [15, 97],
 [96, 49],
 [26, 69],
 [97, 147],
 [71, 127],
 [116, 38],
 [143, 132],
 [51, 42],
 [20, 79],
 [130, 27],
 [21, 12],
 [77, 0],
 [114, 138],
 [90, 85],
 [39, 53],
 [85, 90],
 [41, 97],
 [116, 23],
 [51, 49],
 [105, 137],
 [44, 24],
 [111, 61],
 [97, 41],
 [27, 40],
 [134, 140],
 [64, 3],
 [79, 19],
 [46, 67],
 [107, 16],
 [81, 37],
 [67, 34],
 [79, 23],
 [26, 96],
 [11, 98],
 [116, 95],
 [49, 96],
 [71, 50],
 [32, 12],
 [97, 15],
 [108, 5],
 [69, 84],
 [101, 52],
 [7, 30],
 [23, 79],
 [110, 35],
 [30, 7],
 [76, 131]]

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

