import cv2
import numpy as np


class GameStateProcessing:
    def __init__(self, game_state):
        """Initialize the game state (8x8 grid of zeros)."""
        self.game_state = game_state

    def classify_color(self, hsv_color):
        """Classify the color based on HSV values."""
        h, s, v = hsv_color

        # Adjusted thresholds for better detection
        min_saturation = 130  # Lowered for better tolerance
        min_brightness = 110  # Lowered slightly

        # Ignore very low-saturation or dark areas
        if s < min_saturation or v < min_brightness:
            return 0  # Empty

        # Red color detection (accounting for red wrapping at 0/180)
        if (0 <= h <= 12) or (150 <= h <= 180):
            return 1  # Red

        # Yellow color detection (expanded range)
        elif 22 <= h <= 40:
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
    """Optimized HSV detection using contour extraction and adaptive equalization."""
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

            # Extract circular region more efficiently
            mask = np.zeros((img_height, img_width), dtype=np.uint8)
            cv2.circle(mask, (x, y), r - 3, 255, thickness=-1)
            masked_hsv = cv2.bitwise_and(hsv_image, hsv_image, mask=mask)

            # Convert to list of valid HSV pixels
            valid_pixels = masked_hsv[mask == 255]

            if valid_pixels.size > 0:
                avg_hsv = np.percentile(valid_pixels, 0, axis=0)  # Faster than median
                hsv_grid[row, col] = avg_hsv

    return hsv_grid


def detect_hsv_values(image, grid_array):
    """Detect HSV values in each circle using a weighted color method based on brightness."""
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_image[:, :, 2] = cv2.equalizeHist(hsv_image[:, :, 2])  # Normalize brightness

    img_height, img_width = hsv_image.shape[:2]
    hsv_grid = np.zeros((grid_array.shape[0], grid_array.shape[1], 3), dtype=np
                        .float32)

    for row in range(grid_array.shape[0]):
        for col in range(grid_array.shape[1]):
            x, y, r = grid_array[row, col].astype(int)  # Extract (x, y, r)
        
            # Ensure coordinates are within image bounds
            if not (0 <= x < img_width and 0 <= y < img_height):
                continue

            # Create mask for circular region
            mask = np.zeros((img_height, img_width), dtype=np.uint8)
            cv2.circle(mask, (x, y), r - 3, 255, thickness=-1)

            # Extract HSV pixels inside the circle
            hsv_values = hsv_image[mask == 255]

            if hsv_values.size > 0:
                # Extract H, S, V channels
                H, S, V = hsv_values[:, 0], hsv_values[:, 1], hsv_values[:, 2]

                # Normalize V values to use as weights (avoid division by zero)
                V_sum = np.sum(V)
                if V_sum > 0:
                    V_weights = V / V_sum  # Normalize brightness as weight
                else:
                    V_weights = np.ones_like(V) / len(V)  # Equal weight if all are zero

                # Compute the weighted average for each HSV channel
                weighted_H = np.sum(H * V_weights)
                weighted_S = np.sum(S * V_weights)
                weighted_V = np.sum(V * V_weights)

                # Store the weighted dominant HSV value
                hsv_grid[row, col] = [weighted_H, weighted_S, weighted_V]

    return hsv_grid

def detect_hsv_values(image, grid_array):
    
    """Detect HSV values in each circle using OpenCV's K-Means clustering."""
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_image[:, :, 2] = cv2.equalizeHist(hsv_image[:, :, 2])

    

    img_height, img_width = hsv_image.shape[:2]
    hsv_grid = np.zeros((grid_array.shape[0], grid_array.shape[1], 3), dtype=np.float32)

    for row in range(grid_array.shape[0]):
            for col in range(grid_array.shape[1]):
                x, y, r = grid_array[row, col].astype(int)  # Extract (x, y, r)
        
                # Ensure coordinates are within image bounds
                if not (0 <= x < img_width and 0 <= y < img_height):
                    continue

                # Create mask for circular region
                mask = np.zeros((img_height, img_width), dtype=np.uint8)
                cv2.circle(mask, (x, y), r - 3, 255, thickness=-1)

                # Extract HSV pixels inside the circle
                hsv_values = hsv_image[mask == 255]

                if hsv_values.size > 0:
                    # Convert pixels to float32 for OpenCV K-Means
                    hsv_values = np.float32(hsv_values)
                    k = 3
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                    _, labels, centers = cv2.kmeans(hsv_values, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

                    # Find the dominant color cluster
                    unique_labels, counts = np.unique(labels, return_counts=True)
                    dominant_hsv = centers[np.argmax(counts)]
                    hsv_grid[row,col] = dominant_hsv  # Store dominant HSV value

    return hsv_grid



import cv2
import numpy as np

class GameStateProcessing:
    def __init__(self, game_state):
        """Initialize the game state (8x8 grid of zeros)."""
        self.game_state = game_state

    def classify_color(self, hsv_color):
        """Classify the color based on HSV values with adaptive filtering."""
        h, s, v = hsv_color

        # Dynamic threshold adjustments based on brightness
        min_saturation = max(100, min(180, int(v * 0.5)))  # Adjusted min saturation dynamically
        min_brightness = 110  # Lowered slightly for better shadow handling

        # Ignore very low-saturation or dark areas
        if s < min_saturation or v < min_brightness:
            return 0  # Empty slot

        # **Red Color Detection**
        if (0 <= h <= 12) or (150 <= h <= 180):
            return 1  # Red

        # **Yellow Color Detection**
        elif 22 <= h <= 40:
            return 2  # Yellow

        return 0  # Default to empty

    def update_game_state(self, image, grid_array):
        """Update the game state using detected colors in each circle."""
        temp_state = self.game_state.copy()
        hsv_grid = detect_hsv_values(image, grid_array)
        changes = 0

        for row in range(grid_array.shape[0]):
            for col in range(grid_array.shape[1]):
                avg_hsv = hsv_grid[row, col]

                if np.any(avg_hsv):  # Ensure valid HSV values
                    new_state = self.classify_color(avg_hsv)
                    temp_state[row, col] = new_state  # Update state using flat index
                    print("Position ({}, {}), State: {}, HSV: {}".format(row, col, new_state, avg_hsv))
                else:
                    temp_state[row, col] = 0  # Default if no data

                # Detect changes
                if self.game_state[row, col] != temp_state[row, col]:
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
                h_values, s_values, v_values = valid_pixels[:, 0], valid_pixels[:, 1], valid_pixels[:, 2]

                # Identify Red Pixels (hue range 0-12 and 150-180)
                red_mask = ((0 <= h_values) & (h_values <= 12)) | ((150 <= h_values) & (h_values <= 180))
                red_hues = h_values[red_mask]
                red_brightness = v_values[red_mask]

                # Identify Yellow Pixels (hue range 22-40)
                yellow_mask = (22 <= h_values) & (h_values <= 40)
                yellow_hues = h_values[yellow_mask]
                yellow_brightness = v_values[yellow_mask]

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
