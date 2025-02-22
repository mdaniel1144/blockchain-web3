import React, { useState, useCallback } from "react";
import { useEth } from './component/context/EtherContext';
import {Route, Routes , BrowserRouter as Router} from 'react-router-dom';
import AccountInfo from "./component/AccountInfo";
import WellcomeBlockcahin from "./component/WellcomeBlockchain"
import Page404 from "./component/Page404"
import Logs from "./component/Logs"
import NavigationBar from "./component/NavigationBar";
import Contract from "./component/Contract";
import './App.css'


function App() {
  const {error , isConnected, setError ,ConnectToHardHat } = useEth();

  return (
    <Router>
      <div>
        <NavigationBar/>
        {error && (<p style={{ color: 'red' }}>{error}</p>)}
        <div className="App-content">
            <Routes>
              <Route path="/" element={<WellcomeBlockcahin />}/>
              <Route path="/logs" element={<Logs />}/>
              <Route path="/contract" element={<Contract/>} />
              <Route path="/accounts" element={<AccountInfo/>} />
              <Route path="*" element={<Page404/>} />
            </Routes>
          </div>
      </div>
    </Router>
   
  );
}

export default App;
