import cv2
import numpy as np

class GameStateProcessing:
    def __init__(self, game_state):
        """Initialize game state and color classification settings."""
        self.game_state = game_state
        self.color_settings = {
            "min_saturation": 130,
            "min_brightness": 110,
            "red_hue_low": 0,
            "red_hue_high": 12,
            "red_hue_alt_low": 150,
            "red_hue_alt_high": 180,
            "yellow_hue_low": 22,
            "yellow_hue_high": 40
        }

    def update_color_settings(self, new_settings):
        """Update color classification settings dynamically."""
        self.color_settings.update(new_settings)
        print(f"✅ Updated color classification settings: {self.color_settings}")

    def classify_color(self, hsv_color):
        """Classify the color based on dynamically set HSV values."""
        h, s, v = hsv_color
        
        min_saturation = max(self.color_settings["min_saturation"], min(255, int(v * 0.5)))  # Adjusted min saturation dynamically
        min_brightness = self.color_settings["min_brightness"]  # Lowered slightly for better shadow handling

        if s < min_saturation or v < min_brightness:
            return 0  # Empty

        if (self.color_settings["red_hue_low"] <= h <= self.color_settings["red_hue_high"]) or \
           (self.color_settings["red_hue_alt_low"] <= h <= self.color_settings["red_hue_alt_high"]):
            return 1  # Red

        elif self.color_settings["yellow_hue_low"] <= h <= self.color_settings["yellow_hue_high"]:
            return 2  # Yellow

        return 0  # Other colors

    def update_game_state(self, image, grid_array):
        """Update the game state using detected colors in each circle."""
        temp_state = self.game_state.copy()
        hsv_grid = detect_hsv_values(image, grid_array)
        changes = 0

        for row in range(grid_array.shape[0]):
            for col in range(grid_array.shape[1]):
                x, y, r = grid_array[row,col].astype(int)
                avg_hsv = hsv_grid[row, col]

                if np.any(avg_hsv):  # Ensure valid HSV values
                    new_state = self.classify_color(avg_hsv)
                    temp_state[row,col] = new_state  # Update state using flat index
                    print("Position ({}, {}), State: {}, HSV: {}".format(row, col, new_state, avg_hsv))
                else:
                    temp_state[row,col] = 0  # Default if no data

                # Detect changes
                if self.game_state[row,col] != temp_state[row,col]:
                    changes += 1

        self.game_state = temp_state
        return changes
    
    
def detect_hsv_values(image, grid_array):
    """Optimized HSV detection using filtering for red and yellow pixels separately."""
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Apply CLAHE for brightness normalization
    v_channel = hsv_image[:, :, 2]
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    hsv_image[:, :, 2] = clahe.apply(v_channel)

    img_height, img_width = hsv_image.shape[:2]
    hsv_grid = np.zeros((grid_array.shape[0], grid_array.shape[1], 3), dtype=np.float32)

    for row in range(grid_array.shape[0]):
        for col in range(grid_array.shape[1]):
            x, y, r = grid_array[row, col].astype(int)

            if not (0 <= x < img_width and 0 <= y < img_height):
                continue  # Skip out-of-bounds points

            # Extract circular region efficiently
            mask = np.zeros((img_height, img_width), dtype=np.uint8)
            cv2.circle(mask, (x, y), r - 3, 255, thickness=-1)
            masked_hsv = cv2.bitwise_and(hsv_image, hsv_image, mask=mask)

            # Convert to list of valid HSV pixels
            valid_pixels = masked_hsv[mask == 255]

            if valid_pixels.size > 0:
                h_values= valid_pixels[:, 0]

                # Identify Red Pixels (hue range 0-12 and 150-180)
                red_mask = ((0 <= h_values) & (h_values <= 12)) | ((150 <= h_values) & (h_values <= 180))
                red_hues = h_values[red_mask]

                # Identify Yellow Pixels (hue range 22-40)
                yellow_mask = (22 <= h_values) & (h_values <= 40)
                yellow_hues = h_values[yellow_mask]

                # Determine Dominant Color
                if red_hues.size > (h_values.size - red_hues.size) and red_hues.size > 0:
                    # Use the brightest red pixels for detection
                    dominant_hsv = np.percentile(valid_pixels[red_mask], 50, axis=0)
                elif yellow_hues.size > (h_values.size - red_hues.size) and yellow_hues.size > 0:
                    # Use the brightest yellow pixels
                    dominant_hsv = np.percentile(valid_pixels[yellow_mask], 50, axis=0)
                else:
                    # If no dominant hue, take the brightest percentile from all pixels
                    dominant_hsv = np.percentile(valid_pixels, 70, axis=0)

                # Store HSV values
                hsv_grid[row, col] = dominant_hsv

    return hsv_grid