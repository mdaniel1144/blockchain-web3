import json
import asyncio
import numpy as np
from web3 import Web3
from concurrent.futures import ThreadPoolExecutor
from ipfs import IPFS_Store, IPFS_ExtractStore
from datasetInteract import train_loader, test_loader
from neural_network import CnnTrain, CnnTest , AverageGradients , CNN , serialize_gradients
from torch.utils.data import Subset
from config import contract_address, nameABI, epoch, sizeWorker, sizeGroup
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import logging

logging.basicConfig(filename='SmartModelDP.log', 
                        level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s', 
                        filemode='w')  # Overwrite the log file each time

# Connect to Ethereum node
w3 = Web3(Web3.HTTPProvider('http://localhost:8545'))  # Update with your Ethereum node URL

# Load contract ABI and address
with open(nameABI, 'r') as f:
    contract_json = json.load(f)
contractABI = contract_json['abi']
contractAddress = Web3.to_checksum_address(contract_address)

# Create contract instance
contract = w3.eth.contract(address=contractAddress, abi=contractABI)

async def main():
    # Get accounts
    accounts = w3.eth.accounts  # No need for get_accounts, w3.eth.accounts works
    print(f"Total workers {len(accounts)} registered.")

    try:
        # Add Setting in contract
        print("--> Set Setting...")
        await TransactAsync(contract.functions.Setting(epoch, sizeGroup), accounts[0])

        # Register workers
        print("--> Registering workers...")
        for i in range(min(sizeWorker, len(accounts))):
            if i == 0:
                await TransactAsync(contract.functions.register(True), accounts[i])
            else:
                await TransactAsync(contract.functions.register(False), accounts[i])
            print(f"    Worker {i} registered.")
        print("    Workers registration complete.")

        # Split To Group dataset
        await SetGroupsDataset(accounts[0])  # Get dataset size and upload it

        # Start training
        await TransactAsync(contract.functions.startTraining(), accounts[0])

        print("--> Training Model DP started...")
        workers = contract.functions.getWorkerAddresses().call()
        await MakingDP(workers)
        print("--> Training Model DP End")

        print("--> Test Model...")
        MakeTestDP()

    except Exception as error:
        print(f"   An error occurred: {error}")

async def SetGroupsDataset(account):
    # Calculate the total number of samples in the dataset
    total_samples = len(train_loader.dataset)
    
    print(f"--> Using MNIST dataset with {total_samples} samples.")

    try:
        # Upload dataset size to the smart contract
        await TransactAsync(contract.functions.dataDivision(total_samples), account)

        # Retrieve size groups from the contract
        size_groups = contract.functions.getSizeGroups().call()
        for i in range(size_groups):
            range_data = contract.functions.getGroupData(i).call()
            print(f"    Group #{i}: {range_data[0]}-{range_data[1]}")

        print(f"    Dataset size of {total_samples} uploaded to the contract and divided into groups.")
    except Exception as err:
        print(f"    Error uploading dataset size to the contract: {err}")

async def MakingDP(workers):
    finishedGradients = None
    try:
        EPOCH = contract.functions.getEpoch().call()
        
        tasks = [asyncio.create_task(ProcessAccountByEpoch(EPOCH , address)) for address in workers]
        #tasks = [asyncio.create_task(ProcessAccountByBatch(EPOCH , address)) for address in workers]    # Create tasks for all workers
        await asyncio.gather(*tasks)                                                              # Wait for all tasks to complete

        isFinished = contract.functions.allGroupsComplete().call()
        if isFinished:
            print("--> All accounts finished training")
            currentGradientHash = contract.functions.getAvgGradients().call()
            finishedGradients = await asyncio.to_thread(IPFS_ExtractStore, currentGradientHash)
        else:
            raise Exception("Model-DP Fail")

    except Exception as err: 
        print(f"Error processing: {str(err)}")
    return finishedGradients

async def ProcessAccountByBatch(EPOCH,account):
    currentGradient = None
    rangeSet = contract.functions.getRangeDatasetByWorkerID().call({'from': account})
    start_index, end_index = rangeSet[0], rangeSet[1]
    dataset_training = Subset(train_loader.dataset, range(start_index, end_index))
    workerID = contract.functions.getWorkerID().call({'from': account})
    print(f"    Processing-DP workerID: {workerID} - Dataset range: {start_index} - {end_index}")

    for i in range(EPOCH):
        print(f"    Worker {workerID} - Epoch {i}")
        newGradients, overall_accuracy, average_loss = await CnnTrainByBatch(dataset_training, currentGradient, workerID, i , account)
        currentGradientHash = contract.functions.getAvgGradients().call()
        currentGradient = IPFS_ExtractStore(currentGradientHash)

    print(f"    Worker {workerID}: ACCURACY: {overall_accuracy}, Avg_Loss: {average_loss}")
    await TransactAsync(contract.functions.setTrainningComplete(), account)



def MakeTestDP():
    # Create a new CNN model instance
    currentGradientHash = contract.functions.getAvgGradients().call()
    averaged_gradients = IPFS_ExtractStore(currentGradientHash)

    accuracy, test_loss = CnnTest(averaged_gradients, test_loader)
    print(f'   Test set: '
          f'   - Average loss: {test_loss:.4f}'
          f'   - Accuracy: {accuracy:.2f}%')
    print(f"--> Test completed")
    
async def TransactAsync(function, account):
    loop = asyncio.get_running_loop()     # Get the current event loop that is running in the background
    tx_hash = await loop.run_in_executor(None, function.transact, {'from': account})     # Use the event loop to run the transaction function asynchronously in a separate thread
    await loop.run_in_executor(None, w3.eth.wait_for_transaction_receipt, tx_hash)     # This ensures the transaction has been confirmed on the blockchain

#Make Average by Epoch
async def ProcessAccountByEpoch(EPOCH,account):
    currentGradient = None
    rangeSet = contract.functions.getRangeDatasetByWorkerID().call({'from': account})
    start_index, end_index = rangeSet[0], rangeSet[1]
    dataset_training = Subset(train_loader.dataset, range(start_index, end_index))
    workerID = contract.functions.getWorkerID().call({'from': account})
    print(f"    Processing-DP workerID: {workerID} - Dataset range: {start_index} - {end_index}")

    for i in range(EPOCH):
        print(f"    Worker {workerID} - Epoch {i}")
        newGradients, overall_accuracy, average_loss =  CnnTrain(dataset_training, currentGradient, workerID, i)
        gradients_for_contract = AverageGradients(currentGradient, newGradients)

        try:
            print(f"--> Uploading gradients for Worker {workerID}...")
            dataGradientHash = IPFS_Store(gradients_for_contract)
            await TransactAsync(contract.functions.uploadNewGradient(dataGradientHash), account)
            print(f"    Gradients uploaded successfully for Worker {workerID}.")
        except Exception as error:
            print(f"   An error occurred for Worker {workerID}: {error}")

        currentGradientHash = contract.functions.getAvgGradients().call()
        currentGradient = IPFS_ExtractStore(currentGradientHash)

    print(f"    Worker {workerID}: ACCURACY: {overall_accuracy}, Avg_Loss: {average_loss}")
    await TransactAsync(contract.functions.setTrainningComplete(), account)


async def CnnTrainByBatch(datasetTrain, initial_gradients, worker_id, EPOCH , account):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataLoader = DataLoader(datasetTrain, batch_size=64, shuffle=True)
    model = CNN().to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    if initial_gradients:
        model.load_state_dict(initial_gradients)

    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(dataLoader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        try:
            current_gradients = {name: param.grad.cpu().numpy().tolist() for name, param in model.named_parameters() if param.grad is not None}     # Get the current gradients
            currentGradientHash = contract.functions.getAvgGradients().call()                      # Get the contract gradients
            if currentGradientHash != "":
                ipfs_gradients = IPFS_ExtractStore(currentGradientHash)
                averaged_gradients = AverageGradients(current_gradients, ipfs_gradients)              # make avg gradient
                # Update model parameters with averaged gradients
                for name, param in model.named_parameters():
                    param.grad = torch.tensor(averaged_gradients[name]).to(device)
                #Upload to IPFS

                #print(f"--> Uploading gradients for Worker {workerID}...")
                dataGradientHash = IPFS_Store(averaged_gradients)
                await TransactAsync(contract.functions.uploadNewGradient(dataGradientHash), account)
                logging.info(f'Worker-{worker_id} EPOCH-{EPOCH} - Batch {batch_idx} - Upload Gradient')
            else:
                dataGradientHash = IPFS_Store(current_gradients)
                await TransactAsync(contract.functions.uploadNewGradient(dataGradientHash), account)
                logging.info(f'Worker-{worker_id} EPOCH-{EPOCH} - Batch {batch_idx} - Upload Gradient')

        except Exception as error:
            print(f"   An error occurred for Worker {worker_id}: {error}")
            raise Exception(error)
        
        optimizer.step()

        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        if batch_idx % 100 == 0:
            logging.info(f'Worker-{worker_id} EPOCH-{EPOCH} - Batch {batch_idx}: Loss: {loss.item():.6f}, '
                        f'Accuracy: {100. * correct / total:.2f}%')

    average_loss = total_loss / len(dataLoader)
    accuracy = 100. * correct / total
    logging.info(f'Worker-{worker_id} EPOCH-{EPOCH} - Training complete. '
                f'Average loss: {average_loss:.6f}, Accuracy: {accuracy:.2f}%')
    
    
    gradients = serialize_gradients(model.state_dict())
    return gradients , accuracy , average_loss


if __name__ == "__main__":
    asyncio.run(main())
