const { ethers } = require("hardhat");

async function main() {
    const [deployer] = await ethers.getSigners();
    console.log("Deploying contracts with the account:", deployer.address);

    //check if connect to net chain
    const network = await ethers.provider.getNetwork();
    console.log("Network name:", network.name);
}

main().catch((error) => {
    console.error("Error:", error);
    process.exitCode = 1;
});