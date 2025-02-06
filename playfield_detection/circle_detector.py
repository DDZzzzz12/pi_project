import cv2
import numpy as np
import pandas as pd

class CircleDetector:
    def __init__(self, settings = None, max_attempts = 10):
        """Initialize default Hough Transform settings."""
        self.settings = settings if settings else{
            "param1": 90,
            "param2": 20,
            "min_radius": 25,
            "max_radius": 27,
            "dp": 1.3,
            "min_dist": 40,
            "clipLimit": 2.0,
            "tileGridSize": (8, 8)
        }
        self.max_attempts = max_attempts
        self.attempt_count = 0  # Track recursion depth
        self.best_circles = []  # Stores best detection results
        self.best_settings = self.settings.copy()  # Stores best settings
        

    def preprocess_image(self, img):
        """Enhance the image contrast and apply filtering."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        clahe = cv2.createCLAHE(clipLimit=self.settings["clipLimit"], tileGridSize=self.settings["tileGridSize"])
        gray = clahe.apply(gray)
        gray = cv2.medianBlur(gray, 5)
        return gray
    
    def detect_circles(self, image):
        """Detect circles using the Hough Circle Transform."""
        circles = cv2.HoughCircles(
            image,
            cv2.HOUGH_GRADIENT,
            dp=self.settings["dp"],
            minDist=self.settings["min_dist"],
            param1=self.settings["param1"],
            param2=self.settings["param2"],
            minRadius=self.settings["min_radius"],
            maxRadius=self.settings["max_radius"]
        )
        # If circles are found, convert to a list of tuples, otherwise return an empty list
        return [(int(x), int(y), int(r)) for x, y, r in np.round(circles[0, :])] if circles is not None else []

    def draw_circles(self, image, circles):
        """Draw circles on the image and collect center colors."""
        output_image = image.copy()
        if not circles:
            print("Error: 'circles' is empty or not a valid list")
            return output_image
        for (x, y, r) in circles:
            cv2.circle(output_image, (x, y), r, (0, 255, 0), 2)
        return output_image
    
    # Funktion zur Gruppierung von Kreisen in Reihen
    def group_circles_into_rows(self,temp_circles, num_rows=8, row_threshold=20):
        if not temp_circles:
            print("Keine Kreise zum Gruppieren gefunden!")
            return np.array([]), False


        # Falls temp_circles kein Listentyp ist, umwandeln
        if not isinstance(temp_circles, list):
            print("Warnung: temp_circles ist keine Liste! Versuche umzuwandeln...")
            temp_circles = list(temp_circles)
            # Nach y-Koordinate sortieren
        temp_circles.sort(key=lambda c:c[1])

        # Gruppieren der Kreise in Reihen
        rows = []
        current_row = [temp_circles[0]]  # Erste Reihe initialisieren

        for i in range(1, len(temp_circles)):
            # Prüfen, ob der aktuelle Kreis zur aktuellen Reihe gehört
            if abs(temp_circles[i][1] - current_row[-1][1]) <= row_threshold:
                current_row.append(temp_circles[i])
            else:
                rows.append(current_row)  # Aktuelle Reihe speichern
                current_row = [temp_circles[i]]  # Neue Reihe beginnen
        rows.append(current_row)  # Letzte Reihe hinzufügen

        # Prüfen, ob die erwartete Anzahl von Reihen erkannt wurde
        if len(rows) != num_rows:
            print(f"Warnung: Es wurden nicht genau {num_rows} Reihen erkannt! Gefundene Reihen: {len(rows)}")
            return np.array(rows, dtype=object), False

        # Sortiere Kreise innerhalb jeder Reihe nach x-Koordinate
        for row in rows:
            row.sort(key=lambda c: c[0])

        # In ein 2D-Array umwandeln
        circle_grid = np.array(rows)

        return circle_grid
    

    def display_circle_grid(self, circle_grid, image):
        if circle_grid.size == 0:
            print("Kein gültiges Gitter zum Anzeigen!")
            return

        # Convert each tuple (x, y, r) to a string and add pixel color data
        formatted_grid = [[f"({x},{y},{r}) Color: {image[y, x]}" for x, y, r in row] for row in circle_grid]

        # Generate labels
        row_labels = [f"Row {i}" for i in range(len(circle_grid))]
        col_labels = [f"Col {i}" for i in range(len(circle_grid[0]))]

        # Convert to DataFrame
        df = pd.DataFrame(formatted_grid, index=row_labels, columns=col_labels)

        print(df)



    def update_settings(self, new_settings):
        """Update the circle detection settings dynamically."""
        if isinstance(new_settings, dict):
            self.settings.update(new_settings)
            print(f"✅ Updated circle detection settings: {self.settings}")
        else:
            print("⚠️ Invalid settings format. Expected a dictionary.")
            

    def adjust_settings(self, circles):
        """Adjust detection settings dynamically based on previous results."""
        new_settings = self.settings.copy()
        
        if len(circles) > len(self.best_circles):
            print(f"✅ New best detection: {len(circles)} circles.")
            self.best_circles = circles
            self.best_settings = new_settings.copy()  # Save as best settings
        else:
            print("⚠️ Detection decreased! Reverting to last best settings.")
            self.settings = self.best_settings.copy()  # Revert to best settings
            

        # Try alternate parameter modifications
        new_settings["param1"] -= 5
        new_settings["param2"] += 2
        new_settings["min_radius"] -= 1
        new_settings["max_radius"] -= 1
        new_settings["dp"] = max(0.5, new_settings["dp"] - 0.1)  # Prevent too small dp
        new_settings["min_dist"] = max(10, new_settings["min_dist"] - 5)  # Prevent too small min_dist

        print(f"🔄 Adjusting parameters: {new_settings}")
        self.settings = new_settings
        