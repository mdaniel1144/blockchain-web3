// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract SmartModelDp {

    struct Range{
        uint minRange;
        uint maxRange;
    }

    struct Worker {
        uint workerID;
        uint groupID;
        bool isOwner;
        bool trainingComplete;
    }

    struct Group {
        Range rangeDataset;
    }

    struct ModelMetrics {
        uint256 loss;
        uint256 accuracy;
    }

    uint private epoch_count;
    uint private size_group;
    uint private totalWorkers;
    bool private uploadDatasetRange = false;
    bool private trainingInProgress = false;
    string private avgGradientsHash;

    mapping(address => Worker) private workers;
    Group[] private groups;
    address[] private workerAddresses;
    ModelMetrics private globalMetrics;


    event TrainingStarted();
    event GlobalTrainingCompleted();
    event Reset();
    event GroupRange(uint groupID, uint minRange, uint maxRange);
    event WorkerRegistered(address indexed worker, uint workerID, uint groupID);
    event DatasetDivided(string message);

    function register(bool _isOwner) external returns (uint, uint) {
        require(workers[msg.sender].workerID == 0, "Worker already registered");

        totalWorkers++;
        workers[msg.sender] = Worker({ 
            workerID: totalWorkers,
            groupID: (totalWorkers - 1) % size_group,
            isOwner: _isOwner,
            trainingComplete: false
        });

        workerAddresses.push(msg.sender);

        emit WorkerRegistered(msg.sender, workers[msg.sender].workerID, workers[msg.sender].groupID);
        return (workers[msg.sender].workerID, workers[msg.sender].groupID);
    }

    function dataDivision(uint sizeRow) external {
        require(workers[msg.sender].isOwner, "Only owner can divide dataset");
        require(sizeRow > 0, "Dataset size must be greater than zero");

        uint groupSize = sizeRow / size_group;
        uint remainder = sizeRow % size_group;  // To distribute any remaining rows
        uint num = 0;

        for (uint i = 0; i < size_group; i++) {
            uint minRange = num;
            num += groupSize;

            // Distribute the remainder among the first few groups
            if (remainder > 0) {
                num += 1;
                remainder -= 1;
            }
            uint maxRange = num - 1; // Set maxRange to the last row in the group

            groups.push(Group(Range(minRange, maxRange)));
            emit GroupRange(i, groups[i].rangeDataset.minRange, groups[i].rangeDataset.maxRange);
        }

        uploadDatasetRange = true;
        emit DatasetDivided("Data successfully divided equally among groups.");
    }


    function startTraining() external {
        require(workers[msg.sender].isOwner, "Only owner can start training");
        require(uploadDatasetRange, "Training Has no set Range of dataset");
        require(!trainingInProgress, "Training already in progress");
        
        trainingInProgress = true;
        for (uint i = 0; i < workerAddresses.length ; i++) {
            workers[workerAddresses[i]].trainingComplete = false;
        }
        
        emit TrainingStarted();
    }

    function uploadNewGradient(string memory newGradients) external {
        require(trainingInProgress, "Training not in progress");
        require(workers[msg.sender].workerID != 0, "Worker is not registered");
        avgGradientsHash = newGradients;
    }

    function getGroupData(uint _groupID) external view returns (Range memory _range) {
        require(workers[msg.sender].workerID != 0, "Worker not registered");
        require(_groupID < size_group, "Invalid group ID");
    
        return groups[_groupID].rangeDataset;
    }

    function allGroupsComplete() external returns (bool) {

        for (uint i = 0; i < workerAddresses.length ; i++) {
            if (!workers[workerAddresses[i]].trainingComplete) {
                return false;
            }
        }

        trainingInProgress = false;
        return true;
    }

    function Setting(uint epoch, uint sizeGroup) external {
        require(epoch > 0 && sizeGroup > 0 , "Must be unsign number");

        epoch_count = epoch;
        size_group = sizeGroup;
    }
    function reset() public {
        // Reset dynamic arrays and mappings
        delete groups;
        for (uint i = 0; i < workerAddresses.length; i++) {
            delete workers[workerAddresses[i]];
        }
        delete workerAddresses;

        // Reset other state variables
        avgGradientsHash = "";
        globalMetrics = ModelMetrics(0, 0);
        totalWorkers = 0;
        uploadDatasetRange = false;
        trainingInProgress = false;

        epoch_count = 0;
        size_group = 0;

        emit Reset();
    }


    function setTrainningComplete() external{
        require(workers[msg.sender].workerID != 0, "Worker not registered");
        workers[msg.sender].trainingComplete = true;
    }

    function getWorkerAddresses() external view returns (address[] memory) {
        require(workers[msg.sender].workerID != 0, "Worker not registered");
        return workerAddresses;
    }

    function getWorkerID() external view returns (uint) {
        require(workers[msg.sender].workerID != 0, "Worker not registered");
        return workers[msg.sender].workerID;
    }

    function getRangeDatasetByWorkerID() external view returns (Range memory) {
        require(workers[msg.sender].workerID != 0, "Worker not registered");
        return groups[workers[msg.sender].groupID].rangeDataset;
    }

    function getAvgGradients () external view returns (string memory) {
        return avgGradientsHash;
    }

    function getSizeGroups() external view returns (uint) {
        return size_group;
    }

    function getEpoch() external view returns (uint) {
        return epoch_count;
    }
    function getTotalWorkers() external view returns (uint) {
        return totalWorkers;
    }
    
}