#dp_client.py
import asyncio
import base64
import logging
import os
from typing import Dict, List, Tuple

import httpx
import numpy as np
import torch
from neural_network import CNN, train, transform_data
from pydantic_settings import BaseSettings
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F


class Settings(BaseSettings):
    LEARNING_RATE: float = 0.001
    EPOCHS: int = 4
    BATCH_SIZE: int = 256
    SERVER_URL: str = "http://localhost:8000"


settings = Settings()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - CLIENT - %(levelname)s - [Worker ID: %(worker_id)s] - %(message)s",
)
SERVER_URL = settings.SERVER_URL
POLL_INTERVAL = 5  # seconds


def serialize_gradient(grad_array: np.ndarray) -> str:
    return base64.b64encode(grad_array.tobytes()).decode("utf-8")


def deserialize_gradient(shape: List[int], data: str) -> np.ndarray:
    try:
        return np.frombuffer(base64.b64decode(data), dtype=np.float32).reshape(shape)
    except Exception as e:
        logging.error(f"Error deserializing gradient: {e}")
        raise ValueError("Invalid gradient data or shape.")


def set_random_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logging.info(f"Random seed set to {seed}")


async def register_node(client: httpx.AsyncClient) -> Tuple[int, int]:
    url = f"{SERVER_URL}/register"
    logging.info(f"Registering node at {url}")
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = await client.post(url, json={"ip": "TBD"}, timeout=100.0)
            logging.info(f"Response: {response.text}")
            if response.status_code == 200:
                data = response.json()
                worker_id = data.get("worker_id")
                subgroup_id = data.get("subgroup_id")  # Get subgroup ID from server
                seed = data.get("seed")
                set_random_seed(seed)
                return worker_id, subgroup_id, seed
            else:
                logging.warning(f"Failed to register node: {response.text}")
        except httpx.RequestError as e:
            logging.warning(f"Request error on attempt {attempt + 1}: {str(e)}")
        if attempt < max_retries - 1:
            await asyncio.sleep(2**attempt)  # Exponential backoff
    raise RuntimeError("Failed to register node after multiple attempts")

async def wait_for_approval(client: httpx.AsyncClient):
    logging.info("Waiting for approval to start training...")
    
    while True:
        try:
            response = await client.get(f"{SERVER_URL}/get_training_approval", timeout=10.0)
            response.raise_for_status()  # Raises an error for non-200 status codes
            
            approval_status = response.json().get("approved", False)
            
            if approval_status:
                logging.info("Training approval received. Starting training session.")
                return
            else:
                logging.info("Training not yet approved. Checking again shortly...")
        
        except httpx.HTTPStatusError as http_err:
            logging.warning(f"HTTP error occurred: {http_err}")
        except httpx.RequestError as req_err:
            logging.error(f"Request error occurred: {req_err}")
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
        
        await asyncio.sleep(POLL_INTERVAL)

async def get_data(client: httpx.AsyncClient, worker_id: int) -> tuple:
    logging.info(f"Fetching dataset for worker ID: {worker_id}")
    url = f"{SERVER_URL}/get_data/{worker_id}"
    response = await client.get(url, timeout=300.0)
    if response.status_code == 200:
        data = response.json()
        images, labels = transform_data(data)
        logging.info(f"Fetched dataset with {len(images)} samples")
        return images, labels
    else:
        raise RuntimeError(f"Failed to fetch dataset chunk: {response.text}")

    
async def submit_gradients(
    client: httpx.AsyncClient, worker_id: int, subgroup_id: int,  gradients: Dict[str, torch.Tensor]
):
    logging.info(f"Submitting gradients for worker ID: {worker_id}")
    url = f"{SERVER_URL}/submit_gradients"

    gradients_dict = {
        name: {
            "shape": list(grad.shape),
            "data": serialize_gradient(grad.cpu().numpy()),
        }
        for name, grad in gradients.items()
    }

    data = {"worker_id": worker_id, "subgroup_id": subgroup_id, "gradients": gradients_dict}

    try:
        response = await client.post(url, json=data, timeout=300.0)
        if response.status_code == 200:
            logging.info(f"Successfully submitted gradients for worker {worker_id}")
        else:
            logging.error(f"Failed to submit gradients: {response.text}")
    except Exception as e:
        logging.error(f"Error while submitting gradients: {e}")


async def get_avg_gradients(client: httpx.AsyncClient) -> Dict[str, torch.Tensor]:
    logging.info("Polling for average gradients...")
    while True:
        try:
            response = await client.get(
                f"{SERVER_URL}/get_avg_gradients", timeout=300.0
            )
            if response.status_code == 200:
                avg_gradients = response.json().get("avg_gradients")
                logging.info("Received average gradients")

                return {
                    name: torch.tensor(
                        deserialize_gradient(grad["shape"], grad["data"])
                    )
                    for name, grad in avg_gradients.items()
                }
            elif response.status_code == 400:
                logging.info("Waiting for quorum to be met.")
            else:
                logging.warning(f"Unexpected response: {response.text}")
        except Exception as e:
            logging.error(f"Error in polling: {e}")
        await asyncio.sleep(POLL_INTERVAL)


async def update_with_smart_average(local_gradients, global_gradients):
    """Combine local and global gradients with smart averaging."""
    smart_avg_gradients = {}
    for name in local_gradients:
        local_grad = local_gradients[name]
        global_grad = global_gradients[name]
        smart_avg_gradients[name] = global_grad  # Smart averaging
    return smart_avg_gradients


async def apply_gradients(model, optimizer, gradients):
    """Apply the given gradients to the model parameters."""
    for name, param in model.named_parameters():
        if param.grad is not None and name in gradients:
            param.grad = gradients[name]
    optimizer.step()  # Apply the updated gradients

# Heartbeat functions
async def send_heartbeat(client: httpx.AsyncClient, worker_id: int):
    url = f"{SERVER_URL}/update_heartbeat"
    data = {"worker_id": worker_id}
    try:
        response = await client.post(url, json=data, timeout=10.0)
        if response.status_code == 200:
            logging.info(f"Heartbeat sent successfully for worker {worker_id}")
        else:
            logging.error(f"Failed to send heartbeat for worker {worker_id}: {response.text}")
    except Exception as e:
        logging.error(f"Error sending heartbeat for worker {worker_id}: {e}")

async def heartbeat_loop(client: httpx.AsyncClient, worker_id: int, interval: int = 30):
    logging.info(f"Starting heartbeat loop for worker {worker_id} with interval {interval} seconds.")
    while True:
        await send_heartbeat(client, worker_id)
        await asyncio.sleep(interval)

# Training functions
async def train_batch(model: CNN, data: torch.Tensor, target: torch.Tensor, optimizer: torch.optim.Optimizer):
    model.train()
    optimizer.zero_grad()
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    return {name: param.grad.clone() for name, param in model.named_parameters()}, loss.item()


async def train_epoch(
    model: CNN,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    client: httpx.AsyncClient,
    worker_id: int,
    subgroup_id: int,
):
    total_loss = 0
    for batch_idx, (data, target) in enumerate(dataloader):
        # 1. Forward pass and 2. Backpropagation
        gradients, loss = await train_batch(model, data, target, optimizer)
        total_loss += loss

        # 3. Submit the gradients
        await submit_gradients(client, worker_id, subgroup_id, gradients)

        # 4. Get the average gradients
        avg_gradients = await get_avg_gradients(client)

        # 5. Replace the computed gradients with the average gradients
        for name, param in model.named_parameters():
            if name in avg_gradients:
                param.grad = avg_gradients[name]

        # 6. Perform optimization on the average gradients
        optimizer.step()

        if batch_idx % 10 == 0:
            logging.info(f"Train Batch: [{batch_idx}/{len(dataloader)}]\tLoss: {loss:.6f}")

    avg_loss = total_loss / len(dataloader)
    logging.info(f"Epoch completed for worker {worker_id}. Average loss: {avg_loss:.6f}")


async def main():
    async with httpx.AsyncClient() as client:
        worker_id, subgroup_id, seed = await register_node(client)
        logging.info(f"Registered as worker {worker_id} to subgroup {subgroup_id} with seed {seed}")
        # Set worker_id in the logger context
        logging.LoggerAdapter(logging.getLogger(), {"worker_id": worker_id})

        # Start the heartbeat loop in the background
        asyncio.create_task(heartbeat_loop(client, worker_id))
        
        #Wait for approval from the server function
        logging.info("Waiting for approval from the server...")

        # Wait for approval before starting the training session
        await wait_for_approval(client)
        
        images, labels = await get_data(client, worker_id)
        dataset = TensorDataset(images, labels)
        dataloader = DataLoader(dataset, batch_size=settings.BATCH_SIZE, shuffle=True)

        model = CNN()
        # optimizer = torch.optim.SGD(model.parameters(), lr=settings.LEARNING_RATE)
        optimizer = torch.optim.Adam(model.parameters(), lr=settings.LEARNING_RATE)
        for epoch in range(settings.EPOCHS):
            logging.info(f"Starting epoch {epoch + 1}/{settings.EPOCHS}")
            await train_epoch(model, dataloader, optimizer, client, worker_id, subgroup_id)
            logging.info(f"Completed epoch {epoch + 1}/{settings.EPOCHS}")

        logging.info("Training completed")


if __name__ == "__main__":
    asyncio.run(main())
