import RPi.GPIO as GPIO
import time
from playfield_detection import GameStateProcessing

# Set up the GPIO pin
GPIO.setmode(GPIO.BCM)  # Use BCM pin numbering
GPIO_PIN_IN = 18  # Replace with your GPIO input pin number
GPIO_PIN_OUT = 23  # Replace with your GPIO output pin number
GPIO.setup(GPIO_PIN_IN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)  # Set up the pin as an input with a pull-down resistor
GPIO.setup(GPIO_PIN_OUT, GPIO.OUT)  # Set up the pin as an output

def signal_callback(channel, shared_data, data_lock, pause_event):
    """Callback function to handle GPIO signal."""
    print("Signal detected on GPIO pin", channel)
    # Disable event detection while processing
    GPIO.remove_event_detect(GPIO_PIN_IN)
    pause_event.clear()  # Pause the main loop
    
    # Trigger the process
    process_triggered(shared_data, data_lock)
    

    pause_event.set()  # Signal main loop to resume    # Re-enable event detection after processing
    listen_for_signal(channel, shared_data , data_lock, pause_event)
    
def process_triggered(shared_data, data_lock):
    """Function to handle the process when the signal is detected."""
    print("Process started...")
    
    # Access shared data
    with data_lock:
        hsv_grid = shared_data.get("hsv_grid")
        grid_array = shared_data.get("grid_array")
        # Perform analysis using hsv_grid and grid_array
        if hsv_grid is not None and grid_array is not None:
            
            with data_lock:
                shared_data["camera_paused"] = True  # Pause the camera
            print("Analyzing data...")
            # Add your analysis logic here
            
            #  TODO ADD PROCESSING LOGIC HERE  
            
            with data_lock:
                shared_data["camera_paused"] = False
        
            # Example: print the shapes of the arrays
            print(f"hsv_grid shape: {hsv_grid.shape}")
            print(f"grid_array shape: {grid_array.shape}")
        else:
            print("No data available for analysis.")

    # Send a signal out
    GPIO.output(GPIO_PIN_OUT, GPIO.HIGH)
    time.sleep(1)  # Keep the signal high for 1 second
    GPIO.output(GPIO_PIN_OUT, GPIO.LOW)
    print("Signal sent out. Listening for the next signal...")

def listen_for_signal(shared_data, data_lock, pause_event):
    """Function to keep the script running and listening for signals."""
    # Add initial event detection on the GPIO pin
    GPIO.add_event_detect(GPIO_PIN_IN, GPIO.RISING, callback=lambda ch: signal_callback(ch, shared_data, data_lock, pause_event))
    try:
        print("Listening for GPIO signals...")
        while True:
            time.sleep(1)  # Keep the script running
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        GPIO.cleanup()  # Clean up GPIO settings