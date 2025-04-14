# robopoints_handler.py

import requests

class RoboPointsHandler:
    def __init__(self, endpoint_url):
        """endpoint_url is the remote server that returns points from an image+instruction."""
        self.endpoint_url = endpoint_url

    def call_remote_for_points(self, frame_b64, instruction):
        """Posts {image, instruction} to the remote server, returns 'result' (the points string)."""
        try:
            resp = requests.post(
                self.endpoint_url,
                json={"image": frame_b64, "instruction": instruction},
                timeout=30
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("result", "")
        except Exception as e:
            print("[ERROR in call_remote_for_points]:", e)
            return ""
