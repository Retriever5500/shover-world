from gymnasium import *
import numpy as np

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
        return self.val
    
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
        self.non_stationary_d = 4

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
                initial_stamina=1000, 
                max_timestep=400, 
                map_path=None, 
                render_mode='human',
                action_map={1:'up',
                            2:'right',
                            3:'down',
                            4:'left',
                            5:'barrier_marker',
                            6:'hellify'}):
        
        self.metadata = {'render_modes':['human', None], 'render_fps':30}
        assert not (map_path != None and any(n_rows, n_cols, number_of_boxes, number_of_barriers, number_of_lavas)), \
            'if map_path is specified, none of the parameters, related to random map generation should be specified.'
        
        assert (map_path != None or all(n_rows, n_cols, number_of_boxes, number_of_barriers, number_of_lavas)), \
            'exactly one of map_path or random map generation parameters, should be specified.'
        
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.number_of_boxes = number_of_boxes
        self.number_of_barriers = number_of_barriers
        self.number_of_lavas = number_of_lavas
        self.initial_force = initial_force
        self.unit_force = unit_force
        self.perf_sq_initial_age = perf_sq_initial_age
        self.initial_stamina = initial_stamina
        self.max_timestep = max_timestep
        self.map_path = map_path
        self.render_mode = render_mode
        
        # to be initialized in env.reset()
        self.map = None
        self.time_step = None
        self.stamina = None
        self.curr_number_of_boxes = None
        self.curr_number_of_barriers = None
        self.curr_number_of_lavas = None
        self.destroyed_number_of_boxes = None

        self.action_space = spaces.Tuple(spaces=[spaces.Box(shape=(2,), low=0, high=max(self.n_rows, self.n_cols) - 1, dtype=int), spaces.Discrete(start=1, n=6, dtype=int)])
        self.observation_space = spaces.Dict(spaces={'map':spaces.Box(shape=(self.n_rows, self.n_cols,), low=-100, high=100, dtype=int), 
                                                     'stamina':spaces.Box(shape=(1,), low=0, high=(2**63)-2, dtype=float),
                                                     'prev_selected_pos':spaces.Box(shape=(2,), low=spaces.Box(shape=(2,), low=0, high=max(self.n_rows, self.n_cols) - 1, dtype=int)),
                                                     'prev_selected_action':spaces.Discrete(start=1, n=6, dtype=int)})
        
    def reset(self, *, seed = None, options = None):
        super().reset(seed=seed, options=options)
        if self.map_path:
            self.map, self.curr_number_of_boxes, \
                self.curr_number_of_barriers, self.curr_number_of_lavas = ShoverWorldEnv._load_map(self.map_path)
        else:
            self.map, self.curr_number_of_boxes, \
                self.curr_number_of_barriers, self.curr_number_of_lavas = ShoverWorldEnv._random_map_generation(self.np_random, 
                                                                                                                self.n_rows, 
                                                                                                                self.n_cols,
                                                                                                                self.number_of_boxes, 
                                                                                                                self.number_of_barriers, 
                                                                                                                self.number_of_lavas)
        self.time_step = 0
        self.stamina = self.initial_stamina
        self.destroyed_number_of_boxes = 0
        
        # TODO indentify perfect squares available
        perfect_squares_available = self._find_perfect_squares()

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
                'perfect_squares_available':perfect_squares_available
            }
    
    def step(self, action):
        assert self.action_space.contains(action), 'action should be contained in the environment\'s action space.'
        
        is_action_valid = None
        chain_length_k = 0
        initial_force_applied = False
        lava_destroyed_this_step = False
        perfect_squares_available = None
        selected_pos_x, selected_pos_y, selected_action = action[0][0], action[0][1], action[1]
        
        # move-actions
        if 1 <= selected_action <= 4:
            target_x, target_y = ShoverWorldEnv._get_target_pos_after_move_action(selected_pos_x, selected_pos_y, selected_action)
            target_obj = self.map[target_x][target_y]
            
            # out-of-bounds move
            if (target_x < 0 or target_x >= self.n_rows) or (target_y < 0 or target_y >= self.n_cols):
                is_action_valid = False
            
            # in-bounds move
            else:
                target_obj_square_type = target_obj.get_square_type()

                # moving to a Empty square
                if target_obj_square_type == 'Empty':
                    is_action_valid = True
                    self.shover_pos = (target_x, target_y)
                
                # moving to a Barrier or Lava square
                elif target_obj_square_type in ['Barrier', 'Lava']:
                    is_action_valid = False

                # moving to a Box square (pushing a Box)
                else:
                    # TODO: the logic for pusing a Box.
                    is_action_valid = True
        
        # Hellify or Barrier Marker actions
        else:
            # Barrier Marker
            if selected_action == 5:
                # TODO: the logic for Barrier Marker
                pass
            
            # Hellify
            else:
                # TOOD: the logic for Hellify
                pass

        self.time_step += 1
        perfect_squares_available = self._find_perfect_squares()

        return {# observation
                'map':self._get_map_repr(), 
                'stamina':self.stamina,
                'prev_selected_pos':(selected_pos_x, selected_pos_y),
                'prev_selected_action':(selected_action)
            }, \
            {# info
                'timestep':self.time_step,
                'current_number_of_boxes':self.curr_number_of_boxes, 
                'destroyed_number_of_boxes':self.destroyed_number_of_boxes, 
                'last_action_valid':is_action_valid, 
                'chain_length_k':chain_length_k, 
                'initial_force_applied':initial_force_applied, 
                'lava_destroyed_this_step':lava_destroyed_this_step, 
                'perfect_squares_available':perfect_squares_available
            }

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
            curr_number_of_boxes (int): 
                curr number of boxes in the map.
            curr_number_of_barriers (int): 
                curr number of barriers in the map.
            curr_number_of_lavas (int): 
                curr number of lavas in the map.
        """
        # TODO: load the map and shover pos
        map = None
        curr_number_of_boxes = None
        curr_number_of_barriers = None
        curr_number_of_lavas = None
        return map, curr_number_of_boxes, curr_number_of_barriers, curr_number_of_lavas

    def _random_map_generation(PRNG, n_rows, n_cols, number_of_boxes, number_of_barriers, number_of_lavas):
        """
        Function for random map generation from parameters

        Args:
            PRNG (np.random.Generator):
                PRNG intialized with a seed, which will be used for random map generation.
            n_rows (int): Height of the grid world.
            n_cols (int): Width of the grid world. Any ``a * b`` map size is supported.
            number_of_boxes (int, default=0):
                Number of movable box obstacles placed during random map generation.
            number_of_barriers (int, default=0):
                Number of impassable barrier cells placed during random map generation.
            number_of_lavas (int, default=0):
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
        # TODO: generate a random map and place the shover
        map = None
        curr_number_of_boxes = None
        curr_number_of_barriers = None
        curr_number_of_lavas = None
        return map, curr_number_of_boxes, curr_number_of_barriers, curr_number_of_lavas

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
        # TODO: find the perfect squares.
        perfect_squares_available = None
        return perfect_squares_available