from pynput import keyboard, mouse
import pygame
import threading
from time import sleep
class ShoverGUI:
    TILE_SIZE = 50
    MARGIN = 10
    INIDE_MARGIN = 8
    BACKGROUND = (8, 0, 122)
    EMPTY = (239, 158, 95)
    LAVA = (255, 48, 7)
    BARRIER = (40, 9, 3)
    BOX = (241, 249, 7)
    FPS = 30
    def __init__(self, n_rows, n_cols, width=800, height=600):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.width = width
        self.height = height
        self.agent_pos = (0, 0)

        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        self.current_world = None
        self.last_update = 0

    def draw(self, world, current_timestep, stamina, chain_length, is_action_valid, frames=10, update=True):
        clock = pygame.time.Clock()
        if(update):
            self.last_update = frames
        for i in range(frames):
            
            self._handle_events(world)

            self.screen.fill(self.BACKGROUND)  # Background color
            ratio = self.last_update / frames
            self.last_update = max(0, self.last_update - 1)
            self._draw_grid(world, ratio)  # Draw the grid
            self._draw_bar(current_timestep, stamina, chain_length, is_action_valid)
            pygame.display.flip()  # Update the display

            clock.tick(self.FPS)

    def close(self):
        pygame.quit()

    def _draw_grid(self, grid, ratio):
        for row in range(len(grid)):
            for col in range(len(grid[0])):
                tile_value = grid[row][col].val
                left = self.width - (self.TILE_SIZE + self.MARGIN) * len(grid[0])
                top = self.height - (self.TILE_SIZE + self.MARGIN) * len(grid)
                x = col * (self.TILE_SIZE + self.MARGIN) + self.MARGIN + left // 2
                y = row * (self.TILE_SIZE + self.MARGIN) + self.MARGIN + top // 2
                
                pygame.draw.rect(self.screen, self.EMPTY, (x, y, self.TILE_SIZE, self.TILE_SIZE))

                if(tile_value == 100):
                    pygame.draw.rect(self.screen, self.BARRIER, (x, y, self.TILE_SIZE, self.TILE_SIZE))
                elif(0 < tile_value and tile_value <= 10):
                    direc = grid[row][col].non_stationary_d
                    top_left_x = x + self.INIDE_MARGIN
                    top_left_y = y + self.INIDE_MARGIN
                    x_size = self.TILE_SIZE - 2 * self.INIDE_MARGIN
                    y_size = self.TILE_SIZE - 2 * self.INIDE_MARGIN

                    if(direc == 5):
                        pygame.draw.rect(self.screen, self.BOX, (top_left_x, top_left_y, x_size, y_size))
                    else:
                        dx = (self.TILE_SIZE + self.MARGIN)
                        dy = (self.TILE_SIZE + self.MARGIN)
                        if(direc == 1):
                            dy = -dy
                            dx = 0
                        elif(direc == 2):
                            dy = 0
                        elif(direc == 3):
                            dx = 0
                        elif(direc == 4):
                            dx = -dx
                            dy = 0

                        dx *= ratio
                        dy *= ratio

                        pygame.draw.rect(self.screen, self.BOX, (top_left_x - dx, top_left_y - dy, x_size, y_size))

                elif(tile_value == -100):
                    pygame.draw.rect(self.screen, self.LAVA, (x, y, self.TILE_SIZE, self.TILE_SIZE))
                
                if(self.agent_pos):
                    # Create a font object
                    x = self.agent_pos[1] * (self.TILE_SIZE + self.MARGIN) + self.MARGIN + left // 2
                    y = self.agent_pos[0] * (self.TILE_SIZE + self.MARGIN) + self.MARGIN + top // 2
                    font = pygame.font.Font(None, 48)  # None uses the default font, size 36

                    # Render the text
                    text_surface = font.render("A", True, (255, 255, 255))  # White color
                    text_rect = text_surface.get_rect(center=(x + self.TILE_SIZE // 2, y + self.TILE_SIZE // 2))

                    # Draw the text on the screen
                    self.screen.blit(text_surface, text_rect)

    def _draw_bar(self, current_step, stamina, chain_length, is_action_valid):
        # Define text properties
        font_color = (255, 255, 255)  # White
        font_size = 24
        font = pygame.font.SysFont('Arial', font_size)
        text_to_display = f"Current Step: {current_step}   Stamina: {stamina}   Chain length: {chain_length}"
        if(is_action_valid):
            text_to_display += "   Valid Action"
        else:
            text_to_display += "   Invalid Action"

        # Render the text
        text_surface = font.render(text_to_display, True, font_color)
        text_rect = text_surface.get_rect()

        # Position the text at the center top
        text_rect.centerx = self.width // 2  # Center horizontally
        text_rect.top = 20  # Set a top margin

        self.screen.blit(text_surface, text_rect)

    def _handle_events(self, grid):
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONUP:
                (x, y) = event.pos
                left = self.width - (self.TILE_SIZE + self.MARGIN) * len(grid[0])
                top = self.height - (self.TILE_SIZE + self.MARGIN) * len(grid)
                col = (x - (self.MARGIN + left // 2)) / (self.TILE_SIZE + self.MARGIN)
                row = (y - (self.MARGIN + top // 2)) / (self.TILE_SIZE + self.MARGIN)
                col, row = int(col), int(row)
                if(row >= 0 and row < self.n_rows and col >= 0 and col < self.n_cols):
                    self.agent_pos = (row, col)

def ShoverHUD(env):
    current_input = None

    def get_input():
        nonlocal current_input
        running = True
        while running:
            with keyboard.Events() as events:
                for event in events:
                    if event is None:
                        continue
                    if not hasattr(event.key, 'char'):
                        continue
                    elif(event.key.char == "w"):
                        current_input = 1
                    elif(event.key.char == "d"):
                        current_input = 2
                    elif(event.key.char == "s"):
                        current_input = 3
                    elif(event.key.char == "a"):
                        current_input = 4
                    elif(event.key.char == "b"):
                        current_input = 5
                    elif(event.key.char == "h"):
                        current_input = 6
                    elif(event.key.char == "r"):
                        current_input = 7
                    elif(event.key.char == "q"):
                        current_input = 8
                        running = False
                        break      

    thread = threading.Thread(target=get_input)
    thread.start()

    running = True

    while running:
        if(not env.game.agent_pos or not current_input):
            env.render(update=False)
            continue

        if(0 < current_input and current_input <= 6):
            env.step((env.game.agent_pos, current_input))
            current_input = None
        elif(current_input == 7):
            env.reset()
            current_input = None
        elif(current_input == 8):
            running = False
            thread.join()

