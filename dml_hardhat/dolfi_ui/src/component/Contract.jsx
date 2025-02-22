import React, { useState }  from 'react';
import { useEth } from './context/EtherContext'; 
import './css/Contract.css'

const Contract = () => {

    const {contract} = useEth()
    const [solFile, setSolFile] = useState(null);
    const [contractAddress, setContractAddress] = useState('');
    const [error, setError] = useState('');

    const handleFileChange = (event) => {
        setSolFile(event.target.files[0]);
    };

    const deployContract = async () => {
        if (!solFile) {
            setError('Please upload a Solidity file.');
            return;
        }

        try {
            
        }
        catch(error){

        }
    };

    return (
        <div>
            <input type="file" accept=".sol" onChange={handleFileChange} />
            <button onClick={deployContract}>Deploy Contract</button>
            {contractAddress && <p>Contract deployed at: {contractAddress}</p>}
            {error && <p style={{ color: 'red' }}>{error}</p>}
        </div>
    );
};

export default Contract;