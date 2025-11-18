from env import *

a = ShoverWorldEnv(40, 10, 5, 3, 3, 2, 2, 2)
# map, num_of_boxes, num_of_barriers, num_of_lavas = ShoverWorldEnv._random_map_generation(3, 3, 2, 2, 2)
print(a._get_map_repr())