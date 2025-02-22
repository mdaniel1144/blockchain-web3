const hre = require("hardhat");
const fs = require("fs");

async function main() {
  const contractAddress = "0x5fbdb2315678afecb367f032d93f642f64180aa3";
  const smartModelDP = await hre.ethers.getContractFactory("SmartModelDp");
  const contract = smartModelDP.attach(contractAddress); // Connect to deployed contract

  // Get signers
  const signers = await hre.ethers.getSigners();
  console.log(`Total worker ${signers.length} registered.`);

  try {
    // Register workers
    console.log("Registering workers...");
    for (let i = 0; i < 6 && i < signers.length; i++) {
      let tx;
      if (i === 0) {
        tx = await contract.connect(signers[i]).register(true);
      } else {
        tx = await contract.connect(signers[i]).register(false);
      }
      await tx.wait();
      console.log(`Worker ${i} registered.`);
    }
    console.log("Workers registration complete.");

    // Load the MNIST dataset
    const name = "mnist_train";
    
    // Get dataset size and upload it
    await SetSizeRowOfDataset(name ,contract, signers[0]);

    // Start training
    await contract.connect(signers[0]).startTraining();
    console.log("Training Model DP started...");
  } catch (error) {
    //console.log(error);
  }
}

async function SetSizeRowOfDataset(name, contract, signer) {
  const datasetPath = `./dataset/${name}.csv`; // Update with your dataset path
  let dataset;

  try {
    dataset = fs.readFileSync(datasetPath, { encoding: "utf-8" });
  } catch (err) {
    console.error(`Error reading dataset file: ${err}`);
    return;
  }

  // Prepare keys and values from the dataset
  const rows = dataset.split("\n").slice(1); // Assuming CSV format with headers
  const keys = [];
  const values = [];

  rows.forEach((row) => {
    const columns = row.split(",");
    const label = columns[0]; // Label (e.g., digit 0-9)
    const data = columns.slice(1); // Pixel data

    // Add label to keys if not already present
    if (!keys.includes(label)) {
      keys.push(label);
      values.push([]);
    }

    // Add data to corresponding key in values
    const index = keys.indexOf(label);
    values[index].push(data.join(",")); // Join pixel values as a string
  });

  console.log(`Using Dataset ${name}`);
  console.log(`Uploading dataset with ${keys.length} classes and ${rows.length} samples.`);

  try {
    // Upload dataset size to the smart contract
    const tx1 = await contract.connect(signer).dataDivision(rows.length);
    await tx1.wait();

    const sizeGroups = await contract.getSizeGroups()
    for (let i = 0; i < sizeGroups; i++) {
      const range = await contract.connect(signer).getGroupData(i); // Await the contract call
      console.log(`Group #${i}: ${range.minRange}-${range.maxRange}`);
    }

    console.log(`Dataset size of ${rows.length} uploaded to contract.`);
  } catch (err) {
    console.error(`Error uploading dataset size to contract: ${err}`);
  }
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
