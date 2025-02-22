import React, { createContext, useState, useContext, useEffect } from 'react';
import { JsonRpcProvider, formatEther , ContractFactory ,Contract} from 'ethers';
import SmartModel from './contract/SmartModel.json'

// Create a context for ethers
const EtherContext = createContext();

export const EthProvider = ({ children }) => {
  const [provider, setProvider] = useState(null);
  const [addressSignersList , SetaddressSignersList] = useState([])
  const [selectedSigner, setSelectedSigner] = useState();
  const [signerInfo, setSignerInfo] = useState({ balance: '0 ETH', name: 'Unknown' });
  const [contract , setContract] = useState(null)
  const [contractAddress, setContractAddress] = useState('0x5fbdb2315678afecb367f032d93f642f64180aa3');
  const [isConnected , setIsConnected] = useState(false)
  const [functionContract , setFunctionContract] = useState()
  const [error, setError] = useState('');


    const ConnectToHardHat = async () => {
        try {
            const newProvider = new JsonRpcProvider("http://127.0.0.1:8545");
            const network = await newProvider.getNetwork();
            const addressSignersRPC =  await newProvider.listAccounts();

            setProvider(newProvider)
            for (const item of addressSignersRPC) {
                addressSignersList.push(item.address);
            }

            setIsConnected(true)
            SetaddressSignersList(addressSignersList)
            setSelectedSigner(addressSignersList[0])
            const newContract = new Contract(contractAddress, SmartModel.abi, selectedSigner);

            setContract(newContract)
            setFunctionContract(Object.keys(contract.interface.functions));
            console.log(`Connected to network: ${network.name} (Chain ID: ${network.chainId})`);

        } 
        catch (err) {
            setError(`Provider connection error: ${err.message}`); 
        }
    };

    useEffect(() => {
      if (provider && addressSignersList.length > 0) {
        setSelectedSigner(addressSignersList[0]);
      }
  }, [provider, addressSignersList]);
  

  useEffect(() => {
    const fetchAccountInfo = async () => {
      if (selectedSigner && provider) {
        try {
          console.log(selectedSigner)
          const balance = await provider.getBalance(selectedSigner);
          const formattedBalance = formatEther(balance);
          
          // Only call contract methods if the contract is properly initialized
          const  accountStatus = true
  
          // Simulating fetching name from a hardcoded map
          const accountNames = { 
            "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266": "Alice",
            "0x70997970C51812dc3A010C7d01b50e0d17dc79C8": "Bob",
            "0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC": "Charlie",
            // Add more accounts and names as needed
          };
          const name = accountNames[selectedSigner] || "Unknown";
            
          setSignerInfo({
            accountStatus,
            balance: `${formattedBalance} ETH`,
            name,
          });
  
        } catch (err) {
          setError(`Error fetching account info: ${err.message}`);
        }
      }
    };
    if (selectedSigner)
      fetchAccountInfo();
  }, [selectedSigner]);


  const MakeEvent = async (nameEvent) =>{

  }


  const GetAllLogs = async () => {
    
    const logs = await provider.send("eth_getLogs", [{
      fromBlock: "earliest",   // You can specify a block range if needed
      toBlock: "latest"        // "latest" to get all logs up to the current block
    }]);
  
    return logs;
  }


  return (
    <EtherContext.Provider value={{addressSignersList , selectedSigner, signerInfo ,error ,isConnected,contract,functionContract,
    setError,
    ConnectToHardHat,
    GetAllLogs,
    setSelectedSigner, 
}}>
      {children}
    </EtherContext.Provider>
  );
};

export const useEth = () => useContext(EtherContext);
