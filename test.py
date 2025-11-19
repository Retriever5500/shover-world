import pytest
import os
import numpy as np
import tempfile
from env import ShoverWorldEnv, Square

# --- Fixtures ---

@pytest.fixture
def basic_env():
    """
    Creates a headless environment with known force constants for easy math.
    We initialize with 1 box/barrier/lava to pass the env's strict validation,
    then manually clear the map to ensure a clean slate for unit tests.
    """
    # Initialize with 1s to satisfy the 'all()' check in env.py
    env = ShoverWorldEnv(
        render_mode=None,
        n_rows=10, n_cols=10,
        number_of_boxes=1, number_of_barriers=1, number_of_lavas=1,
        initial_force=10, unit_force=1, initial_stamina=1000,
        perf_sq_initial_age=5
    )
    env.reset()
    
    # Manually clear the map and counters for test isolation
    env.map = [[Square(val=0, btype='Empty') for __ in range(env.n_cols)] for _ in range(env.n_rows)]
    env.curr_number_of_boxes = 0
    env.curr_number_of_barriers = 0
    env.curr_number_of_lavas = 0
    env.destroyed_number_of_boxes = 0
    env.perfect_squares_available_dict = {}
    
    return env

def create_temp_map(content):
    """Helper to create a temporary map file."""
    fd, path = tempfile.mkstemp(suffix='.txt', text=True)
    with os.fdopen(fd, 'w') as f:
        f.write(content)
    return path

def set_cell(env, r, c, val, btype):
    """Helper to manually set a cell in the map."""
    env.map[r][c] = Square(val, btype)
    # Update counts manually if necessary for specific tests, 
    # though step logic usually relies on map state.
    if btype == 'Box':
        env.curr_number_of_boxes += 1

# --- 1. Push chain formation (k) and blocking behavior ---

def test_push_single_box(basic_env):
    """Test pushing a single box into an empty space."""
    # Setup: [Box] [Empty]
    set_cell(basic_env, 5, 5, 10, 'Box')
    set_cell(basic_env, 5, 6, 0, 'Empty')
    
    # Action: Push box at (5,5) Right (2)
    obs, reward, term, trunc, info = basic_env.step(((5, 5), 2))
    
    assert info['last_action_valid'] is True
    assert info['chain_length_k'] == 1
    assert basic_env.map[5][6].btype == 'Box'
    assert basic_env.map[5][5].btype == 'Empty'

def test_push_chain_k_2(basic_env):
    """Test pushing a chain of 2 boxes."""
    # Setup: [Box] [Box] [Empty]
    set_cell(basic_env, 5, 5, 10, 'Box')
    set_cell(basic_env, 5, 6, 10, 'Box')
    set_cell(basic_env, 5, 7, 0, 'Empty')

    # Action: Push box at (5,5) Right (2)
    obs, _, _, _, info = basic_env.step(((5, 5), 2))

    assert info['last_action_valid'] is True
    assert info['chain_length_k'] == 2
    # Check positions shifted
    assert basic_env.map[5][5].btype == 'Empty'
    assert basic_env.map[5][6].btype == 'Box'
    assert basic_env.map[5][7].btype == 'Box'

def test_push_blocked_by_barrier(basic_env):
    """Test pushing a box into a barrier (should fail)."""
    # Setup: [Box] [Barrier]
    set_cell(basic_env, 5, 5, 10, 'Box')
    set_cell(basic_env, 5, 6, 100, 'Barrier')

    prev_stamina = basic_env.stamina
    obs, _, _, _, info = basic_env.step(((5, 5), 2))

    assert info['last_action_valid'] is False
    assert basic_env.map[5][5].btype == 'Box' # Should not move
    assert basic_env.stamina < prev_stamina # Should still cost stamina

# --- 2. Stamina updates ---

def test_stamina_baseline_cost(basic_env):
    """
    Formula: Initial_Force * Is_Stationary + Unit_Force * k
    Params: Init=10, Unit=1
    """
    # Case: Stationary Box (k=1)
    # Cost = 10 * 1 + 1 * 1 = 11
    set_cell(basic_env, 2, 2, 10, 'Box')
    
    start_stamina = basic_env.stamina
    basic_env.step(((2, 2), 2)) # Move Right
    
    expected_cost = 10 + 1
    assert basic_env.stamina == start_stamina - expected_cost

def test_stamina_chain_cost(basic_env):
    """
    Case: Stationary Chain (k=2)
    Cost = 10 * 1 + 1 * 2 = 12
    """
    set_cell(basic_env, 2, 2, 10, 'Box')
    set_cell(basic_env, 2, 3, 10, 'Box')
    
    start_stamina = basic_env.stamina
    basic_env.step(((2, 2), 2)) # Move Right
    
    expected_cost = 10 + 2
    assert basic_env.stamina == start_stamina - expected_cost

def test_lava_destruction_rewards(basic_env):
    """
    Test box destruction in lava. 
    Note: env.py implements this as a Reward increase, not a stamina refund 
    (based on code analysis of lines 266-276).
    """
    # Setup: [Box] [Lava]
    set_cell(basic_env, 5, 5, 10, 'Box')
    set_cell(basic_env, 5, 6, -100, 'Lava')
    
    obs, reward, _, _, info = basic_env.step(((5, 5), 2))
    
    assert info['lava_destroyed_this_step'] is True
    assert info['destroyed_number_of_boxes'] == 1
    # According to env.py, reward should increase by r_lava (default initial_force = 10)
    assert reward == 10 
    assert basic_env.map[5][5].btype == 'Empty'

def test_barrier_maker_stamina_gain_n2(basic_env):
    """
    Test stamina gain from Barrier Maker special action (n=2).
    Reward = r_barrier_maker(n) = n^2 (default), so 4.
    """
    env = basic_env
    
    # 1. Setup the perfect square on the map (2x2)
    for r in [0, 1]:
        for c in [0, 1]:
            set_cell(env, r, c, 10, 'Box')
    
    # 2. Manually set the environment state for the test
    start_stamina = 500 # Use an arbitrary, clean starting point
    env.stamina = start_stamina
    
    # 3. Ensure the special action is available by simulating detection
    # (0, 0, 2) means top-left at (0,0), size n=2
    env.perfect_squares_available_dict = {(0, 0, 2): 0} 
    
    # 4. Execute Action 5: Barrier Maker
    # The first tuple (0,0) is used as the target position
    env.step(((0, 0), 5)) 
    
    # 5. Assert the result (1 box is destroyed, 3 others converted to Barrier)
    expected_gain = 4 # n^2 = 2^2
    assert env.stamina == start_stamina + expected_gain
    assert env.map[0][0].btype == 'Barrier'
    assert env.map[1][1].btype == 'Barrier'
    assert env.curr_number_of_boxes == 0 # All 4 were consumed

# --- 3. Stationary/Non-Stationary Transitions ---

def test_stationary_transition(basic_env):
    """
    Test that a box becomes non-stationary in the direction it was pushed,
    resulting in cheaper costs for the next immediate push in the same direction.
    """
    # Setup
    set_cell(basic_env, 5, 5, 10, 'Box')
    set_cell(basic_env, 5, 6, 0, 'Empty')
    set_cell(basic_env, 5, 7, 0, 'Empty')

    # 1. First Push (Right): Stationary
    # Cost = 10 (init) + 1 (unit) = 11
    stamina_1 = basic_env.stamina
    basic_env.step(((5, 5), 2)) 
    stamina_2 = basic_env.stamina
    
    assert stamina_1 - stamina_2 == 11
    assert basic_env.map[5][6].is_non_stationary_in_d(2) is True

    # 2. Second Push (Right): Non-Stationary
    # Cost = 0 (init) + 1 (unit) = 1
    basic_env.step(((5, 6), 2))
    stamina_3 = basic_env.stamina
    
    assert stamina_2 - stamina_3 == 1

def test_stationary_reset_different_direction(basic_env):
    """Test that non-stationary status is direction specific."""
    # Setup
    set_cell(basic_env, 5, 5, 10, 'Box')
    
    # 1. Push Right
    basic_env.step(((5, 5), 2)) # Box now at (5,6)
    
    # 2. Push Down (Different direction)
    # Even though it just moved, it moved Right. It is stationary relative to Down.
    # Cost should include initial force (11).
    stamina_before = basic_env.stamina
    basic_env.step(((5, 6), 3)) # Down
    cost = stamina_before - basic_env.stamina
    
    assert cost == 11

# --- 4. Perfect Square Detection & Special Actions ---

def test_perfect_square_detection_2x2(basic_env):
    """Test detection of a standard 2x2 square."""
    # Create 2x2 at 0,0
    for r in [0, 1]:
        for c in [0, 1]:
            set_cell(basic_env, r, c, 10, 'Box')
            
    squares = basic_env._find_perfect_squares()
    assert (0, 0, 2) in squares

def test_perfect_square_adjacency_fail(basic_env):
    """Test that a 2x2 square with an attached neighbor is NOT a perfect square."""
    # Create 2x2 at 0,0
    for r in [0, 1]:
        for c in [0, 1]:
            set_cell(basic_env, r, c, 10, 'Box')
    
    # Add an extra box adjacent to the square
    set_cell(basic_env, 0, 2, 10, 'Box')
    
    squares = basic_env._find_perfect_squares()
    # Should be empty or not contain the 2x2 at 0,0
    assert (0, 0, 2) not in squares

def test_hellify_action(basic_env):
    """Test Hellify action (n > 2, so n=3)."""
    # Create 3x3 at 0,0
    for r in range(3):
        for c in range(3):
            set_cell(basic_env, r, c, 10, 'Box')
            
    # Manually update avail dict to simulate age/availability
    basic_env.perfect_squares_available_dict = {(0,0,3): 1}
    
    # Action 6: Hellify
    basic_env.step(((0,0), 6))
    
    # Outer ring becomes Empty(0), Inner (1,1) becomes Lava(-100)
    assert basic_env.map[0][0].val == 0
    assert basic_env.map[1][1].val == -100
    assert basic_env.map[1][1].btype == 'Lava'

# --- 5. Invalid Actions ---

def test_invalid_action_push_empty(basic_env):
    """Test selecting an empty square to push."""
    set_cell(basic_env, 0, 0, 0, 'Empty')
    obs, r, term, trunc, info = basic_env.step(((0, 0), 2))
    
    assert info['last_action_valid'] is False

def test_invalid_special_action_unavailable(basic_env):
    """Test trying to use Hellify when no perfect square exists."""
    # Ensure no squares
    basic_env.perfect_squares_available_dict = {}
    
    obs, r, term, trunc, info = basic_env.step(((0, 0), 6)) # 6 = Hellify
    
    # Action shouldn't change map, valid flag might be True in step logic flow 
    # but logic inside Hellify requires existence.
    # Actually, looking at env.py: if perf_sqr_for_hellify_exists is False, 
    # nothing happens. It doesn't explicitly set is_action_valid to False in the `else` block
    # strictly, but `is_action_valid` is init to False at start of step.
    
    assert info['last_action_valid'] is False

# --- 6. Map Loading ---

def test_load_valid_map_format_a():
    """Test loading a valid integer grid map."""
    content = (
        "0 0 0\n"
        "0 10 0\n"
        "0 0 0"
    )
    path = create_temp_map(content)
    
    try:
        env = ShoverWorldEnv(map_path=path, n_rows=3, n_cols=3)
        env.reset()
        assert env.map[1][1].val == 10
        assert env.map[1][1].btype == 'Box'
        assert env.n_rows == 3
        assert env.n_cols == 3
    finally:
        os.remove(path)

def test_load_map_malformed_rows():
    """Test loading a map with inconsistent row lengths."""
    content = (
        "0 0 0\n"
        "0 10\n"  # Missing column
        "0 0 0"
    )
    path = create_temp_map(content)
    
    with pytest.raises(ValueError):
        ShoverWorldEnv(map_path=path, n_rows=3, n_cols=3)
    
    os.remove(path)

if __name__ == "__main__":
    # Manual run if pytest not available
    import sys
    sys.exit(pytest.main(["-v", __file__]))