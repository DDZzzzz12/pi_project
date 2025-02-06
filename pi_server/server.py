import http.server
import cv2
import time



class MJPEGStreamHandler(http.server.BaseHTTPRequestHandler):
    """
    Single-threaded MJPEG streaming handler.

    This handler responds to HTTP GET requests at the "/stream" endpoint by 
    capturing images from the camera, processing them to detect circles, 
    and streaming the processed frames as an MJPEG stream to the client.

    Attributes:
        None

    Methods:
        do_GET():
            Handles GET requests to the "/stream" endpoint.
    """
    
    def do_GET(self):
        if self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()

            try:
                while True:
                    
                    
                    with self.server.data_lock:
                        frame = self.server.shared_data.get("image", None)  # Safely access shared data

                    if frame is not None:
                        _, buffer = cv2.imencode(".jpg", frame)  # Encode frame as JPEG

                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                        self.wfile.write(buffer.tobytes())
                        self.wfile.write(b"\r\n")
                    else:
                        time.sleep(0.1)  # Sleep for a short while before checking again
            except (BrokenPipeError, ConnectionResetError):
                pass  # Handle client disconnections gracefully
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")

def run_server(shared_data, data_lock, pause_event, host="0.0.0.0", port=8080):
    """Runs a single-threaded HTTP server with pause capability."""
    class CustomHTTPServer(http.server.HTTPServer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.shared_data = shared_data
            self.data_lock = data_lock
            self.pause_event = pause_event
            
        def serve_forever(self, poll_interval=0.5):
            """Override serve_forever to handle pausing."""
            try:
                while True:
                    if self.pause_event.is_set():
                        time.sleep(0.1)  # Reduce CPU usage while paused
                        continue
                    self._handle_request_noblock()
            except KeyboardInterrupt:
                pass
            finally:
                self.server_close()
                
    server = CustomHTTPServer((host, port), MJPEGStreamHandler)
    print(f"Server running on http://{host}:{port}/stream")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Shutting down server.")
    finally:
        server.server_close()

if __name__ == "__main__":
    run_server()

