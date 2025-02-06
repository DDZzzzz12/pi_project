# __init__.py

from .circle_detector import CircleDetector
from .game_state_detector import GameStateProcessing, detect_hsv_values
import numpy as np
import cv2
import time

game_state = np.zeros((8, 8), dtype=int)
detector = CircleDetector()
image_processor = GameStateProcessing(game_state=game_state)


def initialize(shared_data, data_lock):
    while shared_data["initialized"] is False:
        image = shared_data["image"]
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray = detector.preprocess_image(rgb_image)
        circles = detector.detect_circles(gray)
        circled_image = detector.draw_circles(rgb_image, circles)
        
        if circled_image is not None:
            if shared_data["circles"] is None:
                if len(circles) == 64:
                    print("Circles detected: ==64: {}", len(circles))
                else:
                    print("Circles detected: not ==64: {}", len(circles))
                    continue
        
        grid_array, hsv_grid = process_image(circles, circled_image)
        if grid_array.shape == (8,8,3) and hsv_grid.shape == (8,8,3):
            
            # set initial game state
            with data_lock:
                shared_data["circles"] = circles
                shared_data["grid_array"] = grid_array
                shared_data["hsv_grid"] = hsv_grid
                shared_data["initialized"] = True
            

def process_image(circles, image):
    
    grid_array = detector.group_circles_into_rows(circles)
    hsv_grid = detect_hsv_values(image, grid_array)
            
    return grid_array, hsv_grid
        
def process_game_state(shared_data, data_lock):
    image = shared_data["image"]
    grid_array = shared_data["grid_array"]
    changes = image_processor.update_game_state(image, grid_array)
    shared_data["game_state"] = image_processor.game_state
    print("Changes detected:", changes)
    print("Game state:{}" , image_processor.game_state)  
    
    if changes > 0:
        print("Game state updated.")
        time.sleep(10)
      
def get_circled_image(image, circles):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return detector.draw_circles(rgb_image, circles)