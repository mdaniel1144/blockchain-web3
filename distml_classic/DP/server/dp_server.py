#dp_server.py
import asyncio
import logging
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Dict
import numpy as np
import base64
from collections import defaultdict
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from pydantic_settings import BaseSettings
import random
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

class Settings(BaseSettings):
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    QUORUM_THRESHOLD: float = 0.70
    REDUNDANCY_FACTOR: int = 2  # Redundancy factor defined in settings
    FAILURE_INTERVAL: int = 90  # seconds   


settings = Settings()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - SERVER - %(levelname)s - %(message)s"
)


transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)
mnist_train = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
mnist_test = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)
mnist_full = torch.utils.data.ConcatDataset([mnist_train, mnist_test])


class Worker(BaseModel):
    ip: str
    port: int = 0
    subgroup_id: int = 0
    role: str = "DP" # Role based on DP, TP, or MP
    status: str = "active"
    last_heartbeat: str = None


class GradientData(BaseModel):
    shape: List[int]
    data: str


class Gradients(BaseModel):
    worker_id: int
    subgroup_id: int
    gradients: Dict[str, GradientData]

class HeartbeatRequest(BaseModel):
    worker_id: int


class HealthMonitor:
    def __init__(self, manager, check_interval: int = 10, failure_interval: int = 90):
        self.manager = manager
        self.check_interval = check_interval  # seconds
        self.failure_interval = failure_interval  # seconds

    async def start(self):
        logging.info(f"Starting HealthMonitor with check interval {self.check_interval} seconds.")
        while True:
            await self.health_check()
            await asyncio.sleep(self.check_interval)

    async def health_check(self):
        now = datetime.now()
        for worker_id, worker in self.manager.workers.items():
            time_since_last_heartbeat = now - worker.last_heartbeat
            logging.info(f"Checking worker {worker_id}: Last heartbeat {time_since_last_heartbeat} ago.")
            
            if time_since_last_heartbeat > timedelta(seconds=self.failure_interval):
                logging.warning(f"Worker {worker_id} has exceeded the failure interval ({self.failure_interval} seconds). Marking as inactive.")
                await self.handle_failure(worker_id)
            else:
                logging.info(f"Worker {worker_id} is active.")


    async def handle_failure(self, worker_id):
        logging.warning(f"Worker {worker_id} has failed or is inactive.")
        if worker_id in self.manager.workers:
            del self.manager.workers[worker_id]
            subgroup_id = self.manager.workers[worker_id].subgroup_id
            self.manager.subgroups[subgroup_id].remove(worker_id)
            logging.info(f"Worker {worker_id} removed from subgroup {subgroup_id}")
    
            # if subgroup is empty, handle it
            # handle_empty_subgroup(subgroup_id)

    async def update_heartbeat(self, worker_id: int):
        self.manager.workers[worker_id].last_heartbeat = datetime.now()
        logging.info(f"Heartbeat received from worker {worker_id}")


# Thread-safe class for managing state
class GradientManager:
    def __init__(self):
        self.workers = {}
        self.subgroups = defaultdict(list)  # Track workers by subgroups
        self.gradients_queue = defaultdict(list)
        self.avg_gradients = {}
        self.lock = asyncio.Lock()
        self.global_seed = random.randint(0, 10000)
        self.training_approved = False
        self.redundancy_factor = settings.REDUNDANCY_FACTOR

    async def register_worker(self, worker: Worker):
        async with self.lock:
            try:
                worker_id = len(self.workers) + 1
                subgroup_id = (worker_id - 1) // self.redundancy_factor  # Assign to subgroup
                worker.subgroup_id = subgroup_id
                worker.last_heartbeat = datetime.now()

                # Check if the subgroup is already full
                if len(self.subgroups[subgroup_id]) >= self.redundancy_factor:
                    raise RuntimeError("Subgroup is full. Cannot add more workers.")

                self.subgroups[subgroup_id].append(worker_id)  # Add worker to subgroup
                self.workers[worker_id] = worker
                logging.info(f"Successfully registered worker {worker_id} in subgroup {subgroup_id}")
                return worker_id, subgroup_id, self.global_seed

            except ValueError as ve:
                logging.error(f"Registration failed for worker with role '{worker.role}': {ve}")
                raise HTTPException(status_code=400, detail=str(ve))
            except RuntimeError as re:
                logging.error(f"Registration failed for worker {worker_id}: {re}")
                raise HTTPException(status_code=503, detail=str(re))
            except Exception as e:
                logging.error(f"Unexpected error during registration: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")

        
    async def submit_gradients(self, worker_id: int, gradients: Dict[str, np.ndarray]):
        async with self.lock:
            group_id = self.workers[worker_id].subgroup_id
            self.gradients_queue[group_id].append(gradients)
            logging.info(f"Received gradients from worker {worker_id}, in group {group_id}")

    async def compute_average_gradients(self):
        async with self.lock:
            all_gradients = [
                grad
                for worker_grads in self.gradients_queue.values()
                for grad in worker_grads
            ]
            avg_gradients = {}
            for param_name in all_gradients[0].keys():
                param_gradients = [grad[param_name] for grad in all_gradients]
                avg_gradients[param_name] = np.mean(param_gradients, axis=0)
            self.gradients_queue.clear()
            self.avg_gradients = avg_gradients
            logging.info("Average gradients computed")

    async def get_avg_gradients(self):
        async with self.lock:
            return self.avg_gradients if self.avg_gradients else None

    async def set_training_approval(self, approved: bool):
        async with self.lock:
            self.training_approved = approved
            logging.info(f"Training approval set to: {approved}")

    async def get_training_approval(self):
        async with self.lock:
            return self.training_approved

# Initialize a single instance of GradientManager at startup
gradient_manager = GradientManager()

# Dependency management to ensure single instance of GradientManager
def get_gradient_manager():
    return gradient_manager

# Initialize the health monitor 
health_monitor = HealthMonitor(get_gradient_manager(), check_interval=10, failure_interval=settings.FAILURE_INTERVAL)

#get_health_check_manager() function to return the health_monitor instance
def get_health_check_manager():
    return health_monitor

# Utility functions for serialization/deserialization
def serialize_gradient(grad_array: np.ndarray) -> str:
    return base64.b64encode(grad_array.tobytes()).decode("utf-8")


def deserialize_gradient(shape: List[int], data: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(data), dtype=np.float32).reshape(shape)

# Context manager to start and stop the health monitoring task
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start the health monitoring task
    health_monitor_task = asyncio.create_task(health_monitor.start())
    try:
        # Yield control back to FastAPI
        yield
    finally:
        # Cancel the health monitoring task on shutdown
        health_monitor_task.cancel()
        try:
            await health_monitor_task
        except asyncio.CancelledError:
            logging.info("Health monitor task was cancelled during shutdown.")

# Initialize the FastAPI app with the lifespan context manager
app = FastAPI(lifespan=lifespan)

# API route to register a worker
@app.post("/register")
async def register_worker(
    worker: Worker, manager: GradientManager = Depends(get_gradient_manager)
):
    worker_id, subgroup_id, seed = await manager.register_worker(worker)
    return {"worker_id": worker_id, "subgroup_id": subgroup_id, "seed": seed}

@app.post("/set_training_approval")
async def set_training_approval(
    approved: bool, manager: GradientManager = Depends(get_gradient_manager)
):
    await manager.set_training_approval(approved)
    return {"status": "Training approval updated", "approved": approved}

@app.get("/get_training_approval")
async def get_training_approval(
    manager: GradientManager = Depends(get_gradient_manager)
):
    approved = await manager.get_training_approval()
    return {"approved": approved}

# API route to get data partition for a worker
@app.get("/get_data/{worker_id}")
async def get_data(
    worker_id: int, manager: GradientManager = Depends(get_gradient_manager)
):
    if worker_id not in manager.workers:
        raise HTTPException(status_code=404, detail="Worker not found")

    # Get the worker's subgroup assignment
    worker = manager.workers[worker_id]
    subgroup_id = worker.subgroup_id
    
    num_groups = len(manager.subgroups)

    subgroup_idx = subgroup_id

    samples_per_subgroup = len(mnist_full) // num_groups
    start_idx = subgroup_idx * samples_per_subgroup
    end_idx = (
        start_idx + samples_per_subgroup if (subgroup_id + 1) < num_groups else len(mnist_full)
    )

    subgroup_dataset = Subset(mnist_full, range(start_idx, end_idx))
    subgroup_dataloader = DataLoader(
        subgroup_dataset, batch_size=len(subgroup_dataset), shuffle=False
    )
    data, labels = next(iter(subgroup_dataloader))

    logging.info(
        f"Sending dataset chunk to worker {worker_id} in subgroup {subgroup_id} with {len(labels)} samples"
    )
    return {"data": data.numpy().tolist(), "labels": labels.numpy().tolist()}


# API route to submit gradients from a worker
@app.post("/submit_gradients")
async def submit_gradients(
    gradients: Gradients, manager: GradientManager = Depends(get_gradient_manager)
):
    
    worker_id = gradients.worker_id
    group_id = manager.workers[worker_id].subgroup_id
    if worker_id not in manager.workers:
        raise HTTPException(status_code=404, detail="Worker not found")

    deserialized_gradients = {
        name: deserialize_gradient(grad.shape, grad.data)
        for name, grad in gradients.gradients.items()
    }

    if group_id not in manager.gradients_queue or not manager.gradients_queue[group_id]: # make sure each group submits only once
        await manager.submit_gradients(worker_id, deserialized_gradients)

    # Check if quorum is met to compute average gradients
    if len(manager.gradients_queue) >= settings.QUORUM_THRESHOLD * len(manager.subgroups):
        await manager.compute_average_gradients()

    return {"status": "Gradients received"}


# API route to get average gradients
@app.get("/get_avg_gradients")
async def get_avg_gradients(manager: GradientManager = Depends(get_gradient_manager)):
    avg_gradients = await manager.get_avg_gradients()
    if avg_gradients is None:
        raise HTTPException(
            status_code=400, detail="No gradients available or quorum not met"
        )

    serialized_gradients = {
        name: {"shape": list(grad.shape), "data": serialize_gradient(grad)}
        for name, grad in avg_gradients.items()
    }

    logging.info("Sending average gradients to client")
    return {"avg_gradients": serialized_gradients}

from fastapi import HTTPException, status

@app.post("/update_heartbeat")
async def update_heartbeat(
    heartbeat: HeartbeatRequest,  # Expecting the worker_id in request body
    monitor: HealthMonitor = Depends(get_health_check_manager)  # Get HealthMonitor instance
):
    try:
        await monitor.update_heartbeat(heartbeat.worker_id)
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Worker ID not found")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal Server Error")

    return {"status": "Heartbeat updated"}

# Main block to run the app
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.HOST, port=settings.PORT)
