import threading
from server import run_server
from camera_setup import Camera
from playfield_detection import get_circled_image
from signal_processing import listen_for_signal
import numpy as np
#from signal_processing.signal import listen_for_signal

# Global dictionary to store shared data
shared_data = {
    "image": None, 
    "circled_image": None,
    "grid_array": None, 
    "hsv_grid": None, 
    "circles": None,
    "camera_paused": False,
    "initialized" : False,
    "game_state": np.zeros(8,8)
    }
data_lock = threading.Lock()  # Lock for thread safety#
pause_event = threading.Event()  # Event to pause the processing

camera = Camera()
runing = True




def main():
    # Start server in a separate thread
    server_thread = threading.Thread(target=run_server, args=(shared_data, data_lock, pause_event), daemon=True)
    server_thread.start()
    
    # Capture picture stream from camera
    camera_thread = threading.Thread(target=capture_image, args=(shared_data, data_lock, camera, pause_event), daemon=True)
    camera_thread.start()
        
    if shared_data["initialized"] is True:
        # Start listening for GPIO signals in a separate thread
        signal_thread = threading.Thread(target=listen_for_signal,args=(shared_data, data_lock, pause_event), daemon=True)
        signal_thread.start()
    

 
        



def capture_image(shared_data, data_lock, camera, pause_event):
    while runing: 
        with data_lock:
            if shared_data["camera_paused"]:
                pause_event.wait()  # Wait until the pause event is cleared 
                continue
        i = camera.capture_image()
        
        if len(shared_data["circles"]) != 64 or shared_data["circles"] is None:
            print("Circles detected: not ==64: {}", len(shared_data["circles"]))
            continue
        circled_image = get_circled_image(i, shared_data["circles"])
        
        with data_lock:
            shared_data["image"] = i
            shared_data["circled_image"] = circled_image
            
            

            
            
if __name__ == "__main__":
    main()


