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
LRs = [[85,95] ,[100,96] ,[107,120],[88,151],[54,132],[109,10],
[88,151],[44,41],[17,48],[50,16],[119,96],[140,89],[136,52],[148,67],[70,68],[16,91],[138,136],[26,14],[13,27],[90,20],[94,84],[107,99],[79,101],[90,15],
[42,45],[49,18],[17,51],[53,135],[64,147],[69,71] ,[93,134]]
straights = [[49 , 53] , [53, 56] , [64 , 67] , [93,139] , [100, 120] ,[106, 104] , [140,28] , [54 , 149] , [26,24] , [19, 15]]
POSITIONS = straights + LRs
# The FOV for all the cameras
FOV = 100
# The frequency where all the the camera will be colleted. 1 means every frame, 0.5 every two frames
sensors_frequency = {'CameraRGB': 1}
# The yaw of every sensor.
sensors_yaw = {'CameraRGB': 0}
# The percentage of episodes to have lateral noise. More information about the noise can
# be found at docs/agent_module.md
lat_noise_after = 2
# The percentage of episodes with longitudinal noise.
long_noise_after = 0
# The interval of vehicles/pedestrians that every episode can have
NumberOfVehicles = [0, 20]  # The range for the random numbers that are going to be generated
NumberOfPedestrians = [0, 20]

set_of_weathers = [1, 3, 6, 8]

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

