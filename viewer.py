import cv2
import time
from utils.grid_view import build_grid

# Safe maximum display size (works everywhere)
MAX_DISPLAY_WIDTH = 1600
MAX_DISPLAY_HEIGHT = 900


def viewer_worker(frame_queue, camera_ids, resize_w=900, resize_h=600):

    latest_frames = {}
    finished_cameras = set()

    win_name = "Multi-Camera Live View"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    while True:
        try:
            while not frame_queue.empty():
                cam_id, frame = frame_queue.get_nowait()

                # ---- DONE SIGNAL ----
                if frame is None:
                    finished_cameras.add(cam_id)
                else:
                    latest_frames[cam_id] = frame

        except Exception:
            pass

        # ---- EXIT CONDITION (ALL CAMS FINISHED) ----
        if len(finished_cameras) == len(camera_ids):
            print("All camera streams finished. Closing viewer.")
            break

        # ---- BUILD & DISPLAY GRID ----
        if latest_frames:
            frames = [
                latest_frames[cid]
                for cid in camera_ids
                if cid in latest_frames
            ]

            if frames:
                grid = build_grid(frames, resize_w, resize_h)

                gh, gw = grid.shape[:2]

                scale = min(
                    MAX_DISPLAY_WIDTH / gw,
                    MAX_DISPLAY_HEIGHT / gh,
                    1.0
                )

                if scale < 1.0:
                    new_w = int(gw * scale)
                    new_h = int(gh * scale)
                    grid = cv2.resize(
                        grid,
                        (new_w, new_h),
                        interpolation=cv2.INTER_AREA
                    )

                cv2.imshow(win_name, grid)

        # ---- MANUAL EXIT ----
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Viewer terminated by user")
            break

        time.sleep(0.01)

    cv2.destroyAllWindows()