from inference import run_inference

def camera_worker(cam_config, frame_queue):
    try:
        run_inference(cam_config, frame_queue)
    except Exception as e:
        print(f"Camera {cam_config.get('camera_id')} crashed:", e) 