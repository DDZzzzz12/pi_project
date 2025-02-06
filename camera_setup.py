import threading
from picamera2 import Picamera2

class Camera:
    _instance = None  # Singleton instance
    _lock = threading.Lock()  # Lock for thread safety

    def __new__(cls, width=640, height=480):
        """Ensure only one Camera instance exists."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(Camera, cls).__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self, width=640, height=480):
        """Initialize the camera only once."""
        if not self._initialized:
            try:
                self.picam2 = Picamera2()
                self.camera_config = self.picam2.create_video_configuration(
                    main={"size": (width, height)}, buffer_count=3
                )
                self.picam2.configure(self.camera_config)
                self.picam2.start()
                self._initialized = True
                print("✅ Camera initialized successfully")
            except RuntimeError as e:
                print(f"⚠️ Camera initialization failed: {e}")
                self.picam2 = None  # Prevent accessing an invalid camera
    @classmethod
    def get_instance(cls, width=640, height=480):
        """Returns the Singleton instance of the Camera."""
        if cls._instance is None:
            cls._instance = Camera(width, height)
        return cls._instance

    def capture_image(self):
        """Capture an image safely."""
        if not self._initialized or self.picam2 is None:
            raise RuntimeError("Camera is not initialized properly!")
        return self.picam2.capture_array()

    def close(self):
        """Close the camera safely."""
        if self._initialized and self.picam2:
            self.picam2.close()
            self._initialized = False
            print("📷 Camera closed successfully")
