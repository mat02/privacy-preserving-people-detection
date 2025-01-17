training_params = {
    "epochs": 300,
    "patience": False,
    "batch": 64,
    "seed": 0,
    "task": "detect",
    "imgsz": 484,
    
    "prefix": "base_",
}

models = {
    'yolov8-face': {
        "cfg": "yolov8-lite-t-pose.yaml",
        "weigths": None,
    },
    'yolov8-face_improved': {
        "cfg": "yolov8-lite-stematt-bifpn-tiny-pose.yaml",
        "weigths": None,
    },
}

datasets = {
    "TFW": "./datasets/TFW-outdoor.yaml",
}


import os
import threading
from pathlib import Path
from queue import Queue
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
                    output_log = f'gpu{resource_assigned}_{job["dataset"]["name"]}_{model_name}_{date.isoformat()}'
                    print(output_log)
                    
                    args = \
                        f'--prefix "{training_params["prefix"]}" --device_id "{resource_assigned}" --model_cfg_name "{job["model"]["cfg"]}" --dataset "{job["dataset"]["name"]}" "{job["dataset"]["path"]}" --batch_size {training_params["batch"]} --patience {training_params["patience"]} --seed {training_params["seed"]} --extra task={training_params["task"]} epochs={training_params["epochs"]} imgsz={training_params["imgsz"]}> {output_log}.log 2>&1'
                    
                    
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
