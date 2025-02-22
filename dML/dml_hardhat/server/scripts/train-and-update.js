const { ethers } = require("hardhat");
const tf = require('@tensorflow/tfjs-node');

async function main() {
  const [signer] = await ethers.getSigners();
  console.log("Running script with address:", signer.address);

  const DistributedLearningContract = await ethers.getContractFactory("SmartModelDP");
  const contract = await DistributedLearningContract.attach("0x5FbDB2315678afecb367f032d93F642f64180aa3");

  // Get worker info
  const [workerID, groupID] = await contract.workers(signer.address);
  console.log(`Worker ID: ${workerID}, Group ID: ${groupID}`);



// Wait for training to start
contract.on("TrainingStarted", async () => {
    console.log("Training started. Fetching group data...");
  
    // Get group data
    const [keys, values] = await contract.getGroupData(groupID);
    
    const y_TrainKey = ""; // Specify your label key
    const processedData = processData(keys, values, y_TrainKey);
    const [xTrain, yTrain] = processedData;
  
    // Define and compile the model
    const model = tf.sequential();
    model.add(tf.layers.conv2d({ filters: 32, kernelSize: 3, activation: 'relu', inputShape: [28, 28, 1] }));
    model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
    model.add(tf.layers.flatten());
    model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 10, activation: 'softmax' }));
  
    model.compile({
      optimizer: 'adam',
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy'],
    });
  
    // Number of epochs for training
    const epochs = contract.GetEpoch();
  
    for (let epoch = 0; epoch < epochs; epoch++) {
      console.log(`Starting epoch ${epoch + 1}...`);
  
      // Train the model for one epoch
      const history = await model.fit(xTrain, yTrain, {
        epochs: 1, // Train one epoch at a time
        validationSplit: 0.2,
      });
  
      console.log(`Epoch ${epoch + 1} complete. Loss: ${history.history.loss[0]}, Accuracy: ${history.history.acc[0]}`);
  
      // Get the weights
      const weights = await model.getWeights();
      const weightData = await tf.io.encodeWeights(weights);
      const weightsBuffer = Buffer.concat(weightData.map(w => Buffer.from(w.data)));
  
      // Upload results
      try {
        const tx = await contract.uploadGroupResults(
          groupID,
          weightsBuffer,
          Math.floor(history.history.loss[0] * 10000), // Using loss from the current epoch
          Math.floor(history.history.acc[0] * 10000)   // Using accuracy from the current epoch
        );
        await tx.wait();
        console.log("Results uploaded successfully for epoch", epoch + 1);
        
        // Fetch updated weights from the contract if necessary
        // const updatedWeightsBuffer = await contract.getUpdatedWeights(groupID);
        // Process the updated weights as needed
      } catch (error) {
        console.error("Error uploading results:", error);
      }
    }
  
    console.log("All epochs complete. Training finished.");
  });
}

function processData(keys, values, y_TrainKey) {
    const xTrain = []; // Array for features
    const yTrain = []; // Array for labels
  
    // Find the index of the label key
    const labelIndex = keys.indexOf(y_TrainKey);
  
    for (let i = 0; i < values.length; i++) {
      const dataPoint = {};
      for (let j = 0; j < keys.length; j++) {
        // Map each key to its corresponding value, except for the label key
        if (j !== labelIndex) {
          dataPoint[keys[j]] = values[i][j];
        }
      }
      // Push the data point to xTrain
      xTrain.push(dataPoint);
  
      // Push the corresponding label to yTrain, if it exists
      if (labelIndex !== -1) {
        yTrain.push(values[i][labelIndex]);
      }
    }
  
    return [xTrain, yTrain]; // Return the processed dataset
  }

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });