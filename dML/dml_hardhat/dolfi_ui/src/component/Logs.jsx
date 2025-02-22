import React, { useEffect, useState } from 'react';
import { useEth } from './context/EtherContext';
import './css/Logs.css';

const Logs = () => {
  const { GetAllLogs } = useEth();
  const [logs, setLogs] = useState([]); // State to store the fetched logs

  useEffect(() => {
    const logsData = GetAllLogs(); // Fetch the logs
    setLogs(logsData); // Store the logs in state
  }, []); // Empty dependency array ensures this only runs once

  return (
    <div className='Logs-container'>
      <main>
        <h1>Event Logs</h1>
        {logs.length > 0 ? (
          <ul>
            {logs.map((log, index) => (
              <li key={index}>{JSON.stringify(log)}</li> // Render logs
            ))}
          </ul>
        ) : (
          <div>No logs available</div>
        )}
      </main>
    </div>
  );
};

export default Logs;
