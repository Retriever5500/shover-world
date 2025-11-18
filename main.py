from env import *
from gui import ShoverHUD

env = ShoverWorldEnv(40, 10, 5, 6, 6, 5, 2, 2)
env.reset()

ShoverHUD(env)
