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
POSITIONS = [ [29, 105], [130, 27], [87, 102], [27, 132], [44, 24],
              [26, 96], [67, 34], [1, 28], [134, 140], [9, 105],
              [129, 148], [16, 65], [16, 21], [97, 147], [51, 42],
              [41, 30], [107, 16], [47, 69], [95, 102], [145, 16],
              [64, 111], [47, 79], [69, 84], [31, 73], [81, 37],
              [57, 35], [116, 42], [47, 75], [143, 132], [8, 145],
              [107, 43], [111, 61], [105, 137], [72, 24], [77, 0],
              [80, 17], [32, 12], [64, 3], [32, 146], [40, 33],
              [127, 71], [116, 21], [49, 51], [110, 35], [85, 91],
              [114, 93], [30, 7], [110, 133], [60, 43], [11, 98], [96, 49], [90, 85],
              [27, 40], [37, 74], [97, 41], [110, 62], [19, 2], [138, 114], [131, 76],
              [116, 95], [50, 71], [15, 97], [50, 133],
              [23, 116], [38, 116], [101, 52], [5, 108], [23, 79], [13, 68]
             ] +\
             [[19, 66], [79, 14], [19, 57], [39, 53], [60, 26],
             [53, 76], [42, 13], [31, 71], [59, 35], [47, 16],
             [10, 61], [66, 3], [20, 79], [14, 56], [26, 69],
             [79, 19], [2, 29], [16, 14], [5, 57], [77, 68],
             [70, 73], [46, 67], [57, 50], [61, 49], [21, 12]
             ]+\
             [[71, 127], [21, 116],  [51, 49], [35, 110], [91, 85],
             [93, 114], [7, 30], [133, 110], [43, 60], [98, 11], [49, 96], [85, 90],
             [40, 27], [74, 37], [41, 97], [62, 110], [2, 19], [114, 138], [76, 131],
             [95, 116], [71, 50], [97, 15], [74, 71], [133, 50],
             [116, 23], [116, 38], [52, 101], [108, 5], [79, 23], [68, 13]]

# The FOV for all the cameras
FOV = 100
# The frequency where all the the camera will be colleted. 1 means every frame, 0.5 every two frames
sensors_frequency = {'CameraRGB': 1}
# The yaw of every sensor.
sensors_yaw = {'CameraRGB': 0}
# The percentage of episodes to have lateral noise. More information about the noise can
# be found at docs/agent_module.md
lat_noise_after = 1
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

