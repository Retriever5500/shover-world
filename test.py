import numpy as np
from env import ShoverWorldEnv
env = ShoverWorldEnv(render_mode=None, n_rows=6, n_cols=9, number_of_boxes=15,
    number_of_barriers=5, number_of_lavas=4, initial_force=4.0, unit_force=1.0)

obs, info = env.reset()
done, trun = False, False
total_r = 0.0

while not done and not trun:
    a = env.action_space.sample()
    obs, r, done, trun, info = env.step(a)
    total_r += r

print("Episode return:", total_r)
env.close()