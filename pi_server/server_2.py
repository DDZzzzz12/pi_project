import http.server
import cv2
import time
import json
import threading
import numpy as np
from circle_detector import CircleDetector
from game_state_processing import GameStateProcessing  # Import updated class


class MJPEGStreamHandler(http.server.BaseHTTPRequestHandler):
    """
    MJPEG Streaming & Parameter Adjustment API.
    """

    def do_GET(self):
        """Handles MJPEG Streaming."""
        if self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()

            try:
                while True:
                    with self.server.data_lock:
                        frame = self.server.shared_data.get("circled_image", None)

                    if frame is not None:
                        
                        _, buffer = cv2.imencode(".jpg", frame)

                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                        self.wfile.write(buffer.tobytes())
                        self.wfile.write(b"\r\n")
                    else:
                        time.sleep(0.1)
            except (BrokenPipeError, ConnectionResetError):
                pass

        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")
            

    def do_POST(self):
        """Handles POST requests to update circle and color detection settings."""
        if self.path == "/config":
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data)

                with self.server.data_lock:
                    if "hough" in data:
                        self.server.circle_detector.update_settings({
                            "param1": int(data["hough"].get("param1", 90)),
                            "param2": int(data["hough"].get("param2", 20)),
                            "min_radius": int(data["hough"].get("min_radius", 25)),
                            "max_radius": int(data["hough"].get("max_radius", 27)),
                            "dp": float(data["hough"].get("dp", 1.3)),
                            "min_dist": int(data["hough"].get("min_dist", 40)),
                            "clipLimit": float(data["hough"].get("clipLimit", 2.0)),
                            "tileGridSize": tuple(data["hough"].get("tileGridSize", [8, 8]))
                        })

                    if "color" in data:
                        self.server.game_state_processor.update_color_settings({
                            "min_saturation": int(data["color"].get("min_saturation", 130)),
                            "min_brightness": int(data["color"].get("min_brightness", 110)),
                            "red_hue_low": int(data["color"].get("red_hue_low", 0)),
                            "red_hue_high": int(data["color"].get("red_hue_high", 12)),
                            "red_hue_alt_low": int(data["color"].get("red_hue_alt_low", 150)),
                            "red_hue_alt_high": int(data["color"].get("red_hue_alt_high", 180)),
                            "yellow_hue_low": int(data["color"].get("yellow_hue_low", 22)),
                            "yellow_hue_high": int(data["color"].get("yellow_hue_high", 40))
                        })

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "success", "message": "Settings updated"}).encode())
            except Exception as e:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "error", "message": str(e)}).encode())


def run_server(shared_data, data_lock, pause_event, host="0.0.0.0", port=8080):
    """Runs a single-threaded HTTP server with CircleDetector and Color Processing."""

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
                        time.sleep(0.1)
                        continue
                    self._handle_request_noblock()
            except KeyboardInterrupt:
                pass
            finally:
                self.server_close()

    server = CustomHTTPServer((host, port), MJPEGStreamHandler)
    print(f"Server running on http://{host}:{port}/stream")
    print(f"Config API available at http://{host}:{port}/config")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Shutting down server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    shared_data = {}
    data_lock = threading.Lock()
    pause_event = threading.Event()

    run_server(shared_data, data_lock, pause_event)
