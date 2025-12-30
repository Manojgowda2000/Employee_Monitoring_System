import multiprocessing as mp

mp.set_start_method("spawn", force=True)

from multiprocessing import Process, Queue
from inference import load_config, parse_input_sources
from cam_worker import camera_worker
from viewer import viewer_worker
import time

import os
os.environ["PYTORCH_NO_PIN_MEMORY"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

if __name__ == "__main__":
    cfg = load_config("config.properties")
    input_sources = parse_input_sources(cfg)

    frame_queue = Queue(maxsize=20)
    processes = []

    camera_ids = []

    # ---- Start camera processes ----
    for idx, src in enumerate(input_sources, start=1):
        cam_cfg = cfg.copy()
        cam_cfg["input_video_path"] = src
        cam_cfg["camera_id"] = f"cam_{idx}"
        camera_ids.append(cam_cfg["camera_id"])

        p = Process(
            target=camera_worker,
            args=(cam_cfg, frame_queue),
            daemon=True
        )
        p.start()
        processes.append(p)

        print(f"Started {cam_cfg['camera_id']}")

    # ---- Start viewer process ----
    viewer = Process(
        target=viewer_worker,
        args=(frame_queue, camera_ids),
        daemon=True
    )
    viewer.start()
    
    try:
        for p in processes:
            p.join()

        viewer.join()

    
    except KeyboardInterrupt:
        print("Shutting down system...")
