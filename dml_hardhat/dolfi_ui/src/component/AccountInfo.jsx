import React, { useEffect, useState } from 'react';
import { useEth } from './context/EtherContext';

const AccountInfo = () => {
  const {addressSignersList, setSelectedSigner , selectedSigner, signerInfo, setError } = useEth();
  const [selectedIndex, setSelectedIndex] = useState('');

  const handleSelect = (event) => {
    const index = event.target.value;
    setSelectedSigner(addressSignersList[index])
    setSelectedIndex(index); // Update selected index
  };



  return (
    <div>
      <label htmlFor="accountSelect">Select Account:</label>
      <select id="accountSelect" value={selectedIndex} onChange={handleSelect}>
        <option value="" disabled>Select an account</option>
        {addressSignersList.map((signer, index) => (
          <option key={index} value={index}>
            Account #{index}
          </option>
        ))}
      </select>
      {selectedSigner && signerInfo && (
        <div>
          <h2>Account Information</h2>
          <p><strong>Address:</strong> {selectedSigner}</p>
          <p><strong>Name:</strong> {signerInfo.name}</p>
          <p><strong>Balance:</strong> {signerInfo.balance}</p>
          <p>Status: {signerInfo.accountStatus ? 'Associated with the contract' : 'Not associated'}</p>
          <h3>Events for this account:</h3>
          <ul>
            {/* Add logic to display events for this account */}
          </ul>
        </div>
      )}
    </div>
  );
};

export default AccountInfo;
