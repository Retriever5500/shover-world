import os
from gymnasium import *
import numpy as np
import copy
from gui import ShoverGUI

class Square:
    """
    The class for representing any of the squares (`Lavas`, `Boxes`, `Empty Spaces`, and `Barriers`) in the map of `ShoverWorldEnv`.

    Args:
        val (int): 
            Integer representation of the square. -100 for `Lava`, 0 for `Empty`, 1-10 for `Box`, and 100 for `Barrier`.
        btype (str):
            Name of the square. one of the values `Lava`, `Empty`, `Box`, or `Barrier`.
    """ 
    def __init__(self, val, btype):
        self.val = val
        self.btype = btype
        if 1 <= self.val <= 10 and self.btype == 'Box':
            self.non_stationary_d = 5

    def __str__(self):
        return str(self.val)
    
    def get_square_type(self):
        """
        Returns the square type of the instance, which can be any of `Box`, `Barrier`, `Lava`, or `Empty`.
        """
        return self.btype
    
    def make_stationary(self):
        """
        Makes the `Box` instance stationary in all directions. It should only be used for an instance which has been initialized with btype = `Box`. 
        """
        assert self.btype == 'Box', "this function should only be used for Box btype."
        self.non_stationary_d = 5
    
    def make_non_stationary_in_d(self, d):
        """
        Makes the `Box` instance non-stationary in the given direction d. It should only be used for an instance which has been initialized with btype = `Box`. 
        """
        assert self.btype == 'Box', "this function should only be used for Box btype."
        assert 1 <= d <= 4, "invalid direction for making the Box non-stationary."
        self.non_stationary_d = d

    def is_non_stationary_in_d(self, d):
        """
        Determines whether the `Box` is non-stationary in the given direction d or not. It should only be used for an instance which has been initialized with btype = `Box`. 
        """
        return self.non_stationary_d == d


class ShoverWorldEnv(Env):
    """
    Environment Class for Shover World.

    A configurable grid-world environment where an agent pushes boxes, avoids lava,
    and manages stamina. Maps can be generated randomly or loaded from a file.

    Args:
        render_mode (str | None, default='human'):
            ``'human'`` enables interactive Pygame rendering for visualization.
            ``None`` runs the environment in headless mode (no rendering).
        n_rows (int): Height of the grid world.
        n_cols (int): Width of the grid world. Any ``a * b`` map size is supported.
        max_timestep (int, default=400):
            Maximum number of steps per episode. The episode truncates after this limit.
        number_of_boxes (int, default=0):
            Number of movable box obstacles placed during random map generation.
        number_of_barriers (int, default=0):
            Number of impassable barrier cells placed during random map generation.
        number_of_lavas (int, default=0):
            Number of hazardous lava cells placed during random map generation.
        initial_stamina (int, default=1000):
            Initial stamina value assigned to the agent at the start of each episode.
        initial_force (float, default=1.0):
            Base positive scalar used in the stamina cost calculation
            (see environment mechanics documentation).
        unit_force (float, default=1.0):
            Per-cell positive scalar used in the stamina cost calculation
            (see environment mechanics documentation).
        perf_sq_initial_age (int, default=10):
            Initial age of perfect-square tiles. These tiles automatically dissolve
            after reaching a certain number of steps.
        map_path (str | None, default=None):
            Path to a predefined map file (e.g., ``.txt``). If provided,
            random map generation is skipped and this map is loaded instead.
        action_map (dict[int, str], default={1:'up', \
                                            2:'right', \
                                            3:'down', \
                                            4:'left', \
                                            5:'barrier_marker', \
                                            6:'hellify'}):
            A dictionary, mapping action values to verbose descriptions; 
            we will work with verbose descriptions rather than action values.

    """
    def __init__(self, initial_force, 
                unit_force, 
                perf_sq_initial_age, 
                n_rows=None, 
                n_cols=None, 
                number_of_boxes=None, 
                number_of_barriers=None, 
                number_of_lavas=None, 
                r_lava=None,
                r_barrier_marker=None,
                initial_stamina=1000, 
                max_timestep=400, 
                map_path=None, 
                render_mode='human'):
        
        self.metadata = {'render_modes':['human', None], 'render_fps':30}
        assert (map_path == None and all([n_rows, n_cols, number_of_boxes, number_of_barriers, number_of_lavas])), \
            'specify map_path or map parameters for random map generation.'
        
        assert (map_path != None or all([n_rows, n_cols, number_of_boxes, number_of_barriers, number_of_lavas])), \
            'exactly one of map_path or random map generation parameters, should be specified.'
        
        # if map_path is given, try to load and populate attributes related to map with that.
        _, \
            n_rows, \
            n_cols, \
            number_of_boxes, \
            number_of_barriers, \
            number_of_lavas = ShoverWorldEnv._load_map(map_path) if (map_path != None) else (None, n_rows, n_cols, number_of_boxes, number_of_barriers, number_of_lavas)
        
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.number_of_boxes = number_of_boxes
        self.number_of_barriers = number_of_barriers
        self.number_of_lavas = number_of_lavas
        self.initial_force = initial_force
        self.unit_force = unit_force
        self.r_lava = r_lava if r_lava != None else initial_force
        self.r_barrier_marker = (lambda n: (r_barrier_marker if r_barrier_marker != None else 10 * n * n))
        self.perf_sq_initial_age = perf_sq_initial_age
        self.initial_stamina = initial_stamina
        self.max_timestep = max_timestep
        self.map_path = map_path
        self.render_mode = render_mode
        
        # to be initialized in env.reset()
        self.terminated = None
        self.truncated = None
        self.map = None
        self.time_step = None
        self.stamina = None
        self.curr_number_of_boxes = None
        self.curr_number_of_barriers = None
        self.curr_number_of_lavas = None
        self.destroyed_number_of_boxes = None
        self.perfect_squares_available_dict = None # key=(top_left_x, top_left_y, n), value=perfect_square_age

        self.action_space = spaces.Tuple(spaces=[spaces.Box(low=0, high=max(self.n_rows, self.n_cols) - 1, shape=(2,), dtype=int), spaces.Discrete(n=6, start=1, dtype=int)])
        self.observation_space = spaces.Dict(spaces={'map':spaces.Box(low=-100, high=100, shape=(self.n_rows, self.n_cols,), dtype=int), 
                                                     'stamina':spaces.Box(low=0, high=(2**63)-2, shape=(1,), dtype=float),
                                                     'prev_selected_pos':spaces.Box(low=0, high=max(self.n_rows, self.n_cols) - 1, shape=(2,), dtype=int),
                                                     'prev_selected_action':spaces.Discrete(n=6, start=1, dtype=int)})
                                                     
        if(self.render_mode == "human"):
            self.game = ShoverGUI(self.n_rows, self.n_cols)
        else:
            self.game = None
        
    def reset(self, *, seed = None, options = None):
        super().reset(seed=seed, options=options)
        
        self.terminated = False
        self.truncated = False

        if self.map_path:
            self.map, \
                self.n_rows, \
                self.n_cols, \
                self.curr_number_of_boxes, \
                self.curr_number_of_barriers, \
                self.curr_number_of_lavas = ShoverWorldEnv._load_map(self.map_path)
        else:
            self.map, \
                self.curr_number_of_boxes, \
                self.curr_number_of_barriers, \
                self.curr_number_of_lavas = ShoverWorldEnv._random_map_generation(self.n_rows, 
                                                                                    self.n_cols,
                                                                                    self.number_of_boxes, 
                                                                                    self.number_of_barriers, 
                                                                                    self.number_of_lavas)
        self.time_step = 0
        self.stamina = self.initial_stamina
        self.destroyed_number_of_boxes = 0
        self.perfect_squares_available_dict = dict()
        
        perfect_squares_available_list = self._find_perfect_squares()
        for perfect_square in perfect_squares_available_list:
            self.perfect_squares_available_dict[perfect_square] = 0

        self.last_action_valid = False
        self.chain_length_k = 0

        if(self.render_mode == "human"):
            self.render()

        return {# observation
                'map':self._get_map_repr(), 
                'stamina':self.stamina,
                'prev_selected_pos':(0, 0),
                'prev_selected_action':(6)
            }, \
            {# info
                'timestep':self.time_step,
                'current_number_of_boxes':self.curr_number_of_boxes, 
                'destroyed_number_of_boxes':self.destroyed_number_of_boxes, 
                'last_action_valid':False, 
                'chain_length_k':0, 
                'initial_force_applied':False, 
                'lava_destroyed_this_step':False, 
                'perfect_squares_available':self.perfect_squares_available_dict
            }
    
    def step(self, action):
        def _update_stationary_state_for_all_boxes(make_non_stationary_dict, map, n_rows, n_cols):
            """
            Updates stationary or non-stationary state, for all Boxes on the map.

            Make the Boxes, which have been pushed (and moved) in some direction in the last step, non-stationary in that direction. 
            And those make other boxes stationary.

            Args:
                make_non_stationary_dict (dict[tuple(int, int), int]):
                    Dictionary mapping position of the Boxes which should be made non-stationary to the direction which they should be made
                    non-stationary upon. 
            """
            non_stationary_poses = make_non_stationary_dict.keys()

            for i in range(n_rows):
                for j in range(n_cols): 
                    if map[i][j].get_square_type() == 'Box':
                        if (i, j) not in non_stationary_poses:
                            map[i][j].make_stationary()
                        else:
                            d = make_non_stationary_dict[(i, j)]
                            map[i][j].make_non_stationary_in_d(d)

        def _automatic_dissolution(perfect_squares_available_dict, map, perf_sq_initial_age):
            """ 
            Applies automatic dissolution to the perfect squares which are as aged as perf_sq_initial_age.

            Args:
                perfect_squares_available_dict (dict[tuple(int, int, int), int]):
                    The dictionary mapping (top_left_x, top_left_y, n) to age of the perfect square.
                perf_sq_initial_age (int):
                    The thresholding number, which causes the perfect squares with age >= threshold to dissolve (disappear).

            Returns:
                destroyed_num_of_boxes (int):
                    Total number of boxes, destroyed during automatic dissolution.
            """
            destroyed_num_of_boxes = 0
            for perf_sqr, age in perfect_squares_available_dict.items():
                
                if age >= perf_sq_initial_age:
                    perf_sqr_top_left_x, perf_sqr_top_left_y, n = perf_sqr
                    destroyed_num_of_boxes += n * n

                    for i in range(perf_sqr_top_left_x, perf_sqr_top_left_x + n):
                        for j in range(perf_sqr_top_left_y, perf_sqr_top_left_y + n):
                            map[i][j] = Square(val=0, btype='Empty')
            return destroyed_num_of_boxes

        assert self.action_space.contains(action), 'action should be contained in the environment\'s action space.'
        
        is_action_valid = False
        chain_length_k = 0
        initial_force_applied = False
        lava_destroyed_this_step = False
        reward = 0

        make_non_stationary_dict = dict() # a dictionary which contains positions to make non-stationary as keys and the direction to make non-stationary in as values

        selected_pos_x, selected_pos_y, selected_action = action[0][0], action[0][1], action[1]
        
        # when env.truncated == True, we allow further calls to env.step() (meaning that environment dynamics will work), 
        # but we don't allow it when env.terminated == True (meaning that environment will be static for further steps).
        if not self.terminated:
            # move-actions
            if 1 <= selected_action <= 4:
                selected_obj = self.map[selected_pos_x][selected_pos_y]

                # we can proceed to a move-action only if the selected object is a Box square
                if selected_obj.get_square_type() == 'Box':
                    target_x, target_y = ShoverWorldEnv._get_target_pos_after_move_action(selected_pos_x, selected_pos_y, selected_action)
                    
                    # the target square is in-bounds
                    if (0 <= target_x < self.n_rows) and (0 <= target_y < self.n_cols):
                        target_obj = self.map[target_x][target_y]
                        target_obj_square_type = target_obj.get_square_type()

                        # moving a single Box into a Empty square
                        if target_obj_square_type == 'Empty':
                            push_cost = (self.initial_force * int(not selected_obj.is_non_stationary_in_d(selected_action))) + self.unit_force * 1
                            if push_cost <= self.stamina:
                                is_action_valid = True
                                chain_length_k = 1
                                initial_force_applied = not selected_obj.is_non_stationary_in_d(selected_action)
                                make_non_stationary_dict[(target_x, target_y)] = selected_action

                                self.map[target_x][target_y] = Square(val=10, btype='Box')
                                self.map[selected_pos_x][selected_pos_y] = Square(val=0, btype='Empty')

                                # maintenance of stamina
                                self.stamina -= push_cost
                            
                        # moving a single Box into a Lava square 
                        elif target_obj_square_type == 'Lava':
                            push_cost = (self.initial_force * int(not selected_obj.is_non_stationary_in_d(selected_action))) + self.unit_force * 1
                            if push_cost <= self.stamina:
                                is_action_valid = True
                                chain_length_k = 1
                                initial_force_applied = not selected_obj.is_non_stationary_in_d(selected_action)
                                lava_destroyed_this_step = True
                                self.curr_number_of_boxes -= 1
                                self.destroyed_number_of_boxes += 1

                                self.map[selected_pos_x][selected_pos_y] = Square(val=0, btype='Empty')

                                # maintenance of stamina
                                self.stamina -= push_cost

                                reward += self.r_lava
                            
                        # pusing a single Box into a Barrier square 
                        elif target_obj_square_type == 'Barrier':
                            push_cost = (self.initial_force * int(not selected_obj.is_non_stationary_in_d(selected_action))) + self.unit_force * 1
                            if push_cost <= self.stamina:
                                is_action_valid = False
                                chain_length_k = 1
                                initial_force_applied = not selected_obj.is_non_stationary_in_d(selected_action)

                                # maintenance of stamina
                                self.stamina -= push_cost

                        # pushing a chain of Boxes, aligned along the pusing direction
                        else:
                            # finding the target square
                            chain_length_k = 1
                            while (0 <= target_x < self.n_rows and 0 <= target_y < self.n_cols) and self.map[target_x][target_y].get_square_type() == 'Box':
                                chain_length_k += 1
                                target_x, target_y = ShoverWorldEnv._get_target_pos_after_move_action(target_x, target_y, selected_action)

                            # the target square is in-bounds
                            if (0 <= target_x < self.n_rows) and (0 <= target_y < self.n_cols):
                                target_obj = self.map[target_x][target_y]

                                # moving a chain of Boxes into a Empty sqaure
                                if target_obj.get_square_type() == 'Empty':
                                    push_cost = (self.initial_force * int(not selected_obj.is_non_stationary_in_d(selected_action))) + self.unit_force * chain_length_k
                                    if push_cost <= self.stamina:
                                        is_action_valid = True
                                        initial_force_applied = not selected_obj.is_non_stationary_in_d(selected_action)
                                        
                                        # if the target square lies above the chain
                                        if target_x < selected_pos_x:
                                            j = selected_pos_y
                                            for i in range(target_x, selected_pos_x):
                                                make_non_stationary_dict[(i, j)] = selected_action

                                        # if the target square lies below the chain
                                        elif target_x > selected_pos_x:
                                            j = selected_pos_y
                                            for i in range(selected_pos_x + 1, target_x + 1):
                                                make_non_stationary_dict[(i, j)] = selected_action

                                        # if the target square lies to the left of the chain
                                        elif target_y < selected_pos_y:
                                            i = selected_pos_x
                                            for j in range(target_y, selected_pos_y):
                                                make_non_stationary_dict[(i, j)] = selected_action
                                        
                                        # if the target square lies to the right of the chain
                                        else:
                                            i = selected_pos_x
                                            for j in range(selected_pos_y + 1, target_y + 1):
                                                make_non_stationary_dict[(i, j)] = selected_action

                                        self.map[target_x][target_y] = Square(val=10, btype='Box')
                                        self.map[selected_pos_x][selected_pos_y] = Square(val=0, btype='Empty')

                                        # maintenance of stamina
                                        self.stamina -= push_cost
                                    
                                # moving a chain of Boxes into a Lava sqaure
                                elif target_obj.get_square_type() == 'Lava':
                                    push_cost = (self.initial_force * int(not selected_obj.is_non_stationary_in_d(selected_action))) + self.unit_force * chain_length_k
                                    if push_cost <= self.stamina:
                                        is_action_valid = True
                                        lava_destroyed_this_step = True
                                        self.curr_number_of_boxes -= 1
                                        self.destroyed_number_of_boxes += 1
                                        initial_force_applied = not selected_obj.is_non_stationary_in_d(selected_action)
                                        
                                        # if the target square lies above the chain
                                        if target_x < selected_pos_x:
                                            j = selected_pos_y
                                            for i in range(target_x + 1, selected_pos_x):
                                                make_non_stationary_dict[(i, j)] = selected_action

                                        # if the target square lies below the chain
                                        elif target_x > selected_pos_x:
                                            j = selected_pos_y
                                            for i in range(selected_pos_x + 1, target_x):
                                                make_non_stationary_dict[(i, j)] = selected_action

                                        # if the target square lies to the left of the chain
                                        elif target_y < selected_pos_y:
                                            i = selected_pos_x
                                            for j in range(target_y + 1, selected_pos_y):
                                                make_non_stationary_dict[(i, j)] = selected_action
                                        
                                        # if the target square lies to the right of the chain
                                        else:
                                            i = selected_pos_x
                                            for j in range(selected_pos_y + 1, target_y):
                                                make_non_stationary_dict[(i, j)] = selected_action

                                        self.map[selected_pos_x][selected_pos_y] = Square(val=0, btype='Empty')

                                        # maintenance of stamina
                                        self.stamina -= push_cost

                                        reward += self.r_lava

                                # moving a chain of Boxes into a Barrier sqaure
                                elif target_obj.get_square_type() == 'Barrier':
                                    push_cost = (self.initial_force * int(not selected_obj.is_non_stationary_in_d(selected_action))) + self.unit_force * chain_length_k
                                    if push_cost <= self.stamina:
                                        is_action_valid = False
                                        initial_force_applied = not selected_obj.is_non_stationary_in_d(selected_action)

                                        # maintenance of stamina
                                        self.stamina -= push_cost 

                            # the target square is out-bounds
                            else:
                                push_cost = (self.initial_force * int(not selected_obj.is_non_stationary_in_d(selected_action))) + self.unit_force * chain_length_k
                                if push_cost <= self.stamina:
                                    is_action_valid = True
                                    initial_force_applied = not selected_obj.is_non_stationary_in_d(selected_action)
                                    
                                    # if the target square lies above the chain
                                    if target_x < selected_pos_x:
                                        j = selected_pos_y
                                        for i in range(target_x + 1, selected_pos_x):
                                            make_non_stationary_dict[(i, j)] = selected_action

                                    # if the target square lies below the chain
                                    elif target_x > selected_pos_x:
                                        j = selected_pos_y
                                        for i in range(selected_pos_x, target_x):
                                            make_non_stationary_dict[(i, j)] = selected_action

                                    # if the target square lies to the left of the chain
                                    elif target_y < selected_pos_y:
                                        i = selected_pos_x
                                        for j in range(target_y + 1, selected_pos_y + 1):
                                            make_non_stationary_dict[(i, j)] = selected_action
                                    
                                    # if the target square lies to the right of the chain
                                    else:
                                        i = selected_pos_x
                                        for j in range(selected_pos_y, target_y):
                                            make_non_stationary_dict[(i, j)] = selected_action

                                    self.map[selected_pos_x][selected_pos_y] = Square(val=0, btype='Empty')

                                    # maintenance of stamina
                                    self.stamina -= push_cost

                    # the target sqaure is out-bounds
                    else:
                        push_cost = (self.initial_force * int(not selected_obj.is_non_stationary_in_d(selected_action))) + self.unit_force * 1
                        if push_cost <= self.stamina:
                            is_action_valid = True
                            chain_length_k = 1
                            initial_force_applied = not selected_obj.is_non_stationary_in_d(selected_action)
                            self.curr_number_of_boxes -= 1
                            self.destroyed_number_of_boxes += 1

                            self.map[selected_pos_x][selected_pos_y] = Square(val=0, btype='Empty')

                            # maintenance of stamina
                            self.stamina -= push_cost
            
            # Hellify or Barrier Marker actions
            else:
                # Barrier Marker
                if selected_action == 5:
                    # checking whether if these is at least a perfect square such that n >= 2
                    perf_sqr_for_mark_exists = False

                    # find the oldest perfect square such that n >= 2
                    oldest_perf_sqr, oldest_age = None, None

                    for perf_sqr, age in self.perfect_squares_available_dict.items():
                        if perf_sqr[2] >= 2:
                            perf_sqr_for_mark_exists = True
                            is_action_valid = True

                            if oldest_perf_sqr == None:
                                    oldest_perf_sqr, oldest_age = perf_sqr, age
                            
                            elif oldest_age < age:
                                oldest_perf_sqr, oldest_age = perf_sqr, age
                    
                    # if we have at least a perfect square such that n >= 2:
                    if perf_sqr_for_mark_exists:

                        # convert all of the Boxes inside the perfect square into Barriers
                        top_left_x, top_left_y, n = oldest_perf_sqr
                        for i in range(top_left_x, top_left_x + n):
                            for j in range(top_left_y, top_left_y + n):
                                self.map[i][j] = Square(val=100, btype='Barrier')
                            
                        # increament stamina
                        self.stamina += n * n

                        # update current number of boxes, barriers, and destroyed number of boxes
                        self.curr_number_of_boxes -= n * n
                        self.curr_number_of_barriers += n * n
                        self.destroyed_number_of_boxes += n * n

                        reward += self.r_barrier_marker(n)
                
                # Hellify
                else:
                    # checking whether if these is at least a perfect square such that n > 2
                    perf_sqr_for_hellify_exists = False

                    # find the oldest perfect square such that n > 2
                    oldest_perf_sqr, oldest_age = None, None
                    

                    for perf_sqr, age in self.perfect_squares_available_dict.items():
                        if perf_sqr[2] > 2:
                            perf_sqr_for_hellify_exists = True
                            is_action_valid = True
                            
                            if perf_sqr[2] > 2: # n > 2
                                if oldest_perf_sqr == None:
                                    oldest_perf_sqr, oldest_age = perf_sqr, age
                            
                                elif oldest_age < age:
                                    oldest_perf_sqr, oldest_age = perf_sqr, age
                    
                    # if we have at least a perfect square such that n > 2:
                    if perf_sqr_for_hellify_exists:

                        # convert all Boxes on the outer ring to Empty squares, and others to Lava squares
                        top_left_x, top_left_y, n = oldest_perf_sqr
                        for i in range(top_left_x, top_left_x + n):
                            for j in range(top_left_y, top_left_y + n):
                                
                                # if element lies on outer ring, convert it to Empty square
                                if any([i == top_left_x, \
                                    i == (top_left_x + n - 1), \
                                    j == top_left_y, \
                                    j == (top_left_y + n - 1)]):
                                    
                                    self.map[i][j] = Square(val=0, btype='Empty')
                                    
                                # else, element lies in the inner (n-2) * (n-2) perfect square. So, convert it to Lava square
                                else:
                                    self.map[i][j] = Square(val=-100, btype='Lava')

                        # update current number of Boxes and Lavas
                        self.curr_number_of_boxes -= n * n
                        self.destroyed_number_of_boxes += n * n
                        self.curr_number_of_lavas += (n - 2) * (n - 2)

            # maintenance of time_step
            self.time_step += 1

            # update stationary or non-stationary state for all boxes
            _update_stationary_state_for_all_boxes(make_non_stationary_dict, self.map, self.n_rows, self.n_cols)


            # maintenance of perfect_squares_available_dict
            perfect_squares_available_list = self._find_perfect_squares()

            # remove those previous perfect squares which are not available now
            for perfect_square in set(self.perfect_squares_available_dict.keys()):
                if perfect_square not in perfect_squares_available_list:
                    self.perfect_squares_available_dict.pop(perfect_square)

            # increament the age of previous perfect squares which are available now 
            for perfect_square in self.perfect_squares_available_dict.keys():
                self.perfect_squares_available_dict[perfect_square] += 1

            # add new created perfect squares
            for perfect_square in perfect_squares_available_list:
                if perfect_square not in self.perfect_squares_available_dict.keys():
                    self.perfect_squares_available_dict[perfect_square] = 0

            # apply automatic dissolution for aged perfect squares
            destroyed_num_of_boxes_in_automatic_dissolution = _automatic_dissolution(self.perfect_squares_available_dict, self.map, self.perf_sq_initial_age)
            self.curr_number_of_boxes -= destroyed_num_of_boxes_in_automatic_dissolution
            self.destroyed_number_of_boxes += destroyed_num_of_boxes_in_automatic_dissolution

            # check for termination or truncation conditions
            if self.curr_number_of_boxes == 0 or self.stamina == 0:
                self.terminated = True
            if self.time_step == self.max_timestep:
                self.truncated = True

        self.last_action_valid = is_action_valid
        self.chain_length_k = chain_length_k

        if(self.render_mode == "human"):
            self.render()
        
        return {# observation
                'map':self._get_map_repr(), 
                'stamina':self.stamina,
                'prev_selected_pos':(selected_pos_x, selected_pos_y),
                'prev_selected_action':(selected_action)
            }, \
            reward, \
            self.terminated, \
            self.truncated, \
            {# info
                'timestep':self.time_step,
                'current_number_of_boxes':self.curr_number_of_boxes, 
                'destroyed_number_of_boxes':self.destroyed_number_of_boxes, 
                'last_action_valid':is_action_valid, 
                'chain_length_k':chain_length_k, 
                'initial_force_applied':initial_force_applied, 
                'lava_destroyed_this_step':lava_destroyed_this_step, 
                'perfect_squares_available':self.perfect_squares_available_dict
            }

    def render(self, update=True):
        if(self.game):
            self.game.draw(self.map, self.time_step, self.stamina, self.chain_length_k, self.last_action_valid, update=update)

    def close(self):
        if(self.game):
            self.game.close()

    def _get_target_pos_after_move_action(start_x, start_y, action):
        """Returns the x, y coordinates of the landing square after a move action.

        Args:
            start_x (int): 
                x-coordinate of the starting position.
            start_y (int): 
                y-coordinate of the starting position.

        Returns:
            tuple[int, int]: A tuple containing (target_x, target_y), the coordinates
                of the landing position after the move.
        """
        assert 1 <= action <= 4, "this function should only be used when the action is a move."
        
        target_x, target_y = start_x, start_y

        if action == 1:
            target_x -= 1
        elif action == 2:
            target_y += 1
        elif action == 3:
            target_x += 1
        else:
            target_y -= 1
        
        return target_x, target_y

    def _load_map(map_path):
        """
        Function for loading a map from a .txt file. Format A will be used.

        Args:
            map_path (str | None, default=None):
            Path to a predefined map file (e.g., ``.txt``). Note that the shover position should be specfied at the last line of the file.

        Returns:
            map (list[list]): 
                Map containing integers (representing barriers, lavas, or empty square) or instances of `ShoverBox` class.
            n_rows (int):
                number of rows in the map.
            n_cols (int):
                number of columns in the map.
            curr_number_of_boxes (int): 
                curr number of boxes in the map.
            curr_number_of_barriers (int): 
                curr number of barriers in the map.
            curr_number_of_lavas (int): 
                curr number of lavas in the map.
        """
        if not os.path.isfile(map_path):
            raise FileNotFoundError(f"{map_path} not found")
        
        file_map = []
        n_rows = 0
        n_cols = None
        curr_number_of_boxes = 0
        curr_number_of_barriers = 0
        curr_number_of_lavas = 0

        with open(map_path, 'r') as file:
            for line in file:
                row = []
                line = line.strip().split()

                for value in line:
                    value = int(value)
                    if(value == -100):
                        row.append(Square(-100, "Lava"))
                        curr_number_of_lavas += 1
                    elif(value == 0):
                        row.append(Square(0, "Empty"))
                    elif(value > 0 and value <= 10):
                        row.append(Square(value, "Box"))
                        curr_number_of_boxes += 1
                    elif(value == 100):
                        row.append(Square(value, "Barrier"))
                        curr_number_of_barriers += 1

                if(n_cols == None):
                    n_cols = len(row)
                elif(n_cols != len(row)):
                    raise ValueError(f"{map_path} columns don't match!")

                file_map.append(row)

        n_rows = len(file_map)
        
        return file_map, n_rows, n_cols, curr_number_of_boxes, curr_number_of_barriers, curr_number_of_lavas

    def _random_map_generation(n_rows, n_cols, number_of_boxes, number_of_barriers, number_of_lavas):
        """
        Function for random map generation from parameters

        Args:
            n_rows (int): Height of the grid world.
            n_cols (int): Width of the grid world. Any ``a * b`` map size is supported.
            number_of_boxes (int):
                Number of movable box obstacles placed during random map generation.
            number_of_barriers (int):
                Number of impassable barrier cells placed during random map generation.
            number_of_lavas (int):
                Number of hazardous lava cells placed during random map generation.

        Returns:
            map (list[list[Square]]): 
                Map containing instances of Square which represent the objects in the map.
            curr_number_of_boxes (int): 
                curr number of boxes in the map.
            curr_number_of_barriers (int): 
                curr number of barriers in the map.
            curr_number_of_lavas (int): 
                curr number of lavas in the map.
        """
        assert (n_rows * n_cols) >= (number_of_boxes + number_of_barriers + number_of_lavas), 'invalid config for random map genration.'
        
        map = [[Square(val=0, btype='Empty') for j in range(n_cols)] for i in range(n_rows)]

        all_x_idxs = np.arange(stop=n_rows)
        all_y_idxs = np.arange(stop=n_cols)
        X, Y = np.meshgrid(all_x_idxs, all_y_idxs, indexing='ij')
        all_idxs = np.stack([X.ravel(), Y.ravel()], axis=1)
        np.random.shuffle(all_idxs)

        barrier_rand_posses = all_idxs[:number_of_barriers]
        lava_rand_posses = all_idxs[number_of_barriers:number_of_barriers + number_of_lavas]
        box_rand_posses = all_idxs[number_of_barriers + number_of_lavas:number_of_barriers + number_of_lavas + number_of_boxes]

        for barrier_pos in barrier_rand_posses:
            barrier_x, barrier_y = barrier_pos[0], barrier_pos[1]
            map[barrier_x][barrier_y] = Square(val=100, btype='Barrier')

        for lava_pos in lava_rand_posses:
            lava_x, lava_y = lava_pos[0], lava_pos[1]
            map[lava_x][lava_y] = Square(val=-100, btype='Lava')

        for box_pos in box_rand_posses:
            box_x, box_y = box_pos[0], box_pos[1]
            map[box_x][box_y] = Square(val=10, btype='Box')
        
        return map, number_of_boxes, number_of_barriers, number_of_lavas

    def _get_map_repr(self):
        """
        Get the integer representation of the map.

        Returns:
            map (list[list[int]]): 
                Integer representation of the map.
        """
        return [[str(square) for square in row] for row in self.map]
    
    def _find_perfect_squares(self):
        """
        A function for finding the perfect squares.

        Returns:
            perfect_squares_available (list[tuple[int, int, int]]):
                A list of available perfect squares, identified by their n and their top-left x, y coordinate. E.g. [[1, 1, 1], [1, 4, 5], ...].
        """
        def _is_perfect_square(map, top_left_x, top_left_y, n):
            is_perfect_square = True

            # check if the n*n block, starting at top-left corner contains only of boxes
            for i in range(top_left_x, top_left_x + n):
                for j in range(top_left_y, top_left_y + n):
                    if map[i][j].get_square_type() != 'Box':
                        is_perfect_square = False
                        return is_perfect_square
                    
            # check the outer neighbours
            n_rows, n_cols = len(map), len(map[0])

            # top bar (including outer top-left corner)
            i = top_left_x - 1
            for j in range(top_left_y - 1, top_left_y + n):
                if 0 <= i <= (n_rows - 1) and 0 <= j <= (n_cols - 1):
                    if map[i][j].get_square_type() == 'Box':
                       is_perfect_square = False
                       return is_perfect_square 

            # left bar (including outer bottom-left corner)
            j = top_left_y - 1
            for i in range(top_left_x, top_left_x + n + 1):
                if 0 <= i <= (n_rows - 1) and 0 <= j <= (n_cols - 1):
                    if map[i][j].get_square_type() == 'Box':
                       is_perfect_square = False
                       return is_perfect_square 

            # bottom bar (including outer bottom-right corner)
            i = top_left_x + n
            for j in range(top_left_y, top_left_y + n + 1):
                if 0 <= i <= (n_rows - 1) and 0 <= j <= (n_cols - 1):
                    if map[i][j].get_square_type() == 'Box':
                       is_perfect_square = False
                       return is_perfect_square
                    
            # right bar (including outer top-right corner)
            j = top_left_y + n
            for i in range(top_left_x - 1, top_left_x + n - 1):
                if 0 <= i <= (n_rows - 1) and 0 <= j <= (n_cols - 1):
                    if map[i][j].get_square_type() == 'Box':
                       is_perfect_square = False
                       return is_perfect_square

            return is_perfect_square


        perfect_squares_available = list()
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                last_perfect_square = None
                for n in range(1, min(self.n_rows - i, self.n_cols - j) + 1):
                    if _is_perfect_square(self.map, i, j, n):
                        last_perfect_square = (i, j, n)
                
                if last_perfect_square != None:
                    perfect_squares_available.append(last_perfect_square)     

        return perfect_squares_available