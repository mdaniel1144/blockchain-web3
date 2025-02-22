const { ethers } = require("hardhat");

async function main() {
    const SmartModel = await ethers.getContractFactory("SmartModelDp");
    const contract = await SmartModel.deploy();

    // Use `smartModel.address` instead of `smartModel.getAddress()`
    console.log("SmartModel-DP deployed to:", await contract.getAddress());

}

main().catch((error) => {
    console.error("Error:", error);
    process.exitCode = 1;
});
