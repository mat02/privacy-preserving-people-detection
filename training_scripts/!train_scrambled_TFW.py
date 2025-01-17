training_params = {
    "epochs": 120,
    "patience": False,
    "batch": 64,
    "seed": 0,
    "task": "scrambledpose",
    "imgsz": 484,
    
    "prefix": "scrambledpose_ablation",
}

extra_params = {
    "rect": False,
    "pretrained": True,
    
    "dsc_gain": 0.01,
    "nip_gain": 0.1,
    "scr_enabled": True,
    "scr_inversion_percentage": 0.5,
    "scr_num_blocks": 4,
    "scr_block_size": 4,
    "scr_unique_blocks": False
}

models = {
    'yolov8-face-lite-t_4x4': {
        "cfg": "privacy_yolov8-face/yolov8-lite-t-pose_scr_4x4_out3.yaml",
        "weights": "yolov8-lite-t-pose.pt",
        "override": {
            "scr_block_size": 4,
            "scr_num_blocks": 4,
            "scr_unique_blocks": False,
            "dsc_gain": 0.01,
        }
    },
    'yolov8-face-lite-t_4x4_unique': {
        "cfg": "privacy_yolov8-face/yolov8-lite-t-pose_scr_4x4_out3_unique.yaml",
        "weights": "yolov8-lite-t-pose.pt",
        "override": {
            "scr_block_size": 4,
            "scr_num_blocks": 4,
            "scr_unique_blocks": True,
            "dsc_gain": 0.01,
        }
    },
    'yolov8-face-lite-t_4x8': {
        "cfg": "privacy_yolov8-face/yolov8-lite-t-pose_scr_4x8_out3.yaml",
        "weights": "yolov8-lite-t-pose.pt",
        "override": {
            "scr_block_size": 4,
            "scr_num_blocks": 8,
            "scr_unique_blocks": False,
            "dsc_gain": 0.1,
        }
    },
    'yolov8-face-lite-t_4x8_unique': {
        "cfg": "privacy_yolov8-face/yolov8-lite-t-pose_scr_4x8_out3_unique.yaml",
        "weights": "yolov8-lite-t-pose.pt",
        "override": {
            "scr_block_size": 4,
            "scr_num_blocks": 8,
            "scr_unique_blocks": True,
            "dsc_gain": 0.1,
        }
    },
    'yolov8-face-lite-t_8x4': {
        "cfg": "privacy_yolov8-face/yolov8-lite-t-pose_scr_8x4_out3.yaml",
        "weights": "yolov8-lite-t-pose.pt",
        "override": {
            "scr_block_size": 8,
            "scr_num_blocks": 4,
            "scr_unique_blocks": False,
            "dsc_gain": 0.01,
        }
    },
    'yolov8-face-lite-t_8x4_unique': {
        "cfg": "privacy_yolov8-face/yolov8-lite-t-pose_scr_8x4_out3_unique.yaml",
        "weights": "yolov8-lite-t-pose.pt",
        "override": {
            "scr_block_size": 8,
            "scr_num_blocks": 4,
            "scr_unique_blocks": True,
            "dsc_gain": 0.01,
        }
    },
    'yolov8-face-lite-t_improved_4x4': {
        "cfg": "privacy_yolov8-face_improved/yolov8-lite-stematt-bifpn-tiny-pose_scr_4x4_out3.yaml",
        "weights": "yolov8-lite-stematt-bifpn-tiny-pose.pt",
        "override": {
            "scr_block_size": 4,
            "scr_num_blocks": 4,
            "scr_unique_blocks": False,
            "dsc_gain": 0.01,
        }
    },
    'yolov8-face-lite-t_improved_4x4_unique': {
        "cfg": "privacy_yolov8-face_improved/yolov8-lite-stematt-bifpn-tiny-pose_scr_4x4_out3_unique.yaml",
        "weights": "yolov8-lite-stematt-bifpn-tiny-pose.pt",
        "override": {
            "scr_block_size": 4,
            "scr_num_blocks": 4,
            "scr_unique_blocks": True,
            "dsc_gain": 0.01,
        }
    },
    'yolov8-face-lite-t_improved_4x8': {
        "cfg": "privacy_yolov8-face_improved/yolov8-lite-stematt-bifpn-tiny-pose_scr_4x8_out3.yaml",
        "weights": "yolov8-lite-stematt-bifpn-tiny-pose.pt",
        "override": {
            "scr_block_size": 4,
            "scr_num_blocks": 8,
            "scr_unique_blocks": False,
            "dsc_gain": 0.1,
        }
    },
    'yolov8-face-lite-t_improved_4x8_unique': {
        "cfg": "privacy_yolov8-face_improved/yolov8-lite-stematt-bifpn-tiny-pose_scr_4x8_out3_unique.yaml",
        "weights": "yolov8-lite-stematt-bifpn-tiny-pose.pt",
        "override": {
            "scr_block_size": 4,
            "scr_num_blocks": 8,
            "scr_unique_blocks": True,
            "dsc_gain": 0.1,
        }
    },
    'yolov8-face-lite-t_improved_8x4': {
        "cfg": "privacy_yolov8-face_improved/yolov8-lite-stematt-bifpn-tiny-pose_scr_8x4_out3.yaml",
        "weights": "yolov8-lite-stematt-bifpn-tiny-pose.pt",
        "override": {
            "scr_block_size": 8,
            "scr_num_blocks": 4,
            "scr_unique_blocks": False,
            "dsc_gain": 0.01,
        }
    },
    'yolov8-face-lite-t_improved_8x4_unique': {
        "cfg": "privacy_yolov8-face_improved/yolov8-lite-stematt-bifpn-tiny-pose_scr_8x4_out3_unique.yaml",
        "weights": "yolov8-lite-stematt-bifpn-tiny-pose.pt",
        "override": {
            "scr_block_size": 8,
            "scr_num_blocks": 4,
            "scr_unique_blocks": True,
            "dsc_gain": 0.01,
        }
    },
}

datasets = {
    "TFW": "./datasets/TFW-outdoor.yaml",
}


import os
import threading
from pathlib import Path
from queue import Queue
from copy import deepcopy
import datetime

if __name__ == "__main__":
    # Select GPUs
    gpus = [0, 1]
    
    # Lock dictionary to ensure only one thread uses each resource at a time
    resource_locks = {resource: threading.Lock() for resource in gpus}

    # Worker function for processing jobs
    def process_job(job_queue):
        while not job_queue.empty():
            job = job_queue.get()
            resource_assigned = None

            for resource, lock in resource_locks.items():
                # Attempt to acquire a lock for the resource
                if lock.acquire(blocking=False):
                    resource_assigned = resource
                    print(f"{threading.current_thread().name} is processing job using GPU {resource_assigned}")
                    
                    model_name = Path(job["model"]["cfg"]).stem
                    date = datetime.datetime.now()
                    # output_log = f'gpu{resource_assigned}_{job["dataset"]["name"]}_{model_name}_{date.isoformat()}'
                    output_log = f'gpu{resource_assigned}_{job["dataset"]["name"]}_{model_name}'
                    print(output_log)
                    
                    extra_params_local = deepcopy(extra_params)
                    
                    if "override" in job["model"]:
                        extra_params_local.update(job["model"]["override"])
                    
                    extra_params_txt = ""
                    for key, value in extra_params_local.items():
                        extra_params_txt += f"{key}={value} "
                    
                    if job["model"]["weights"] is not None:
                        extra_params_txt += f'base_for_scrambling={job["model"]["weights"]} ' 
                    
                    args = (
                        f'''--prefix "{training_params["prefix"]}" --device_id "{resource_assigned}"
                          --model_cfg_name "{job["model"]["cfg"]}" 
                          --dataset "{job["dataset"]["name"]}" "{job["dataset"]["path"]}" 
                          --batch_size {training_params["batch"]} 
                          --patience {training_params["patience"]} 
                          --seed {training_params["seed"]} 
                          --extra 
                            task={training_params["task"]} 
                            epochs={training_params["epochs"]} 
                            imgsz={training_params["imgsz"]} 
                            {extra_params_txt}
                            > {output_log}.log 2>&1'''
                    ).replace('\n', '')
                    
                    script_path = Path(os.path.dirname(os.path.abspath(__file__))) / "train_job.py"
                    cmd = f"python {script_path.relative_to(os.getcwd())} {args}"
                    # print(cmd)
                    r = os.system(cmd)
                    
                    print(f"{threading.current_thread().name} has completed job using  GPU{resource_assigned}")
                    lock.release()  # Release the resource after job is done
                    break

            if resource_assigned is None:
                print(f"{threading.current_thread().name} could not find a free GPU for job")
                job_queue.put(job)  # Requeue the job for future processing if no resource was free

            job_queue.task_done()

    # Create a queue and populate it with jobs
    jobs = []
    for ds_name, ds_path in datasets.items():
        for model_name, model_params in models.items():
            job = {
                "model": model_params,
                "model_name": model_name,
                "dataset": {
                    "name": ds_name,
                    "path": ds_path
                }
            }
            
            jobs.append(job)
    
    job_queue = Queue()
    for job in jobs:
        job_queue.put(job)

    # Create a pool of threads
    num_threads = len(gpus)  # Adjust the number of threads as needed
    threads = []

    for i in range(num_threads):
        thread = threading.Thread(target=process_job, args=(job_queue,), name=f"Thread-{i+1}")
        thread.start()
        threads.append(thread)

    # Wait for all threads to finish
    for thread in threads:
        thread.join()

    print("All jobs have been processed.")
