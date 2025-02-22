import React, { useState } from 'react';
import { FaChevronRight, FaTimes , FaTags, FaCog } from 'react-icons/fa';
import {useEth} from './context/EtherContext'
import { Link } from 'react-router-dom'
import './css/NavigationBar.css'


const NavigationBar = () => {

  const {isConnected ,ConnectToHardHat } = useEth();
  const [isOpenNavigitonBar , setIsNavigitonBar] = useState(false)
  const [animationStep, setAnimationStep] = useState(0);


  const handleClick = () => {
    ConnectToHardHat()
    setTimeout(() => {
      setAnimationStep(1);
    }, 700); 
  };

  return (
    <div className={`navigation-container ${isOpenNavigitonBar? 'open' : ''}`}>  
      <div className='navigation-content-container'>
        <button className={`navigation-button-connect ${isConnected ? 'connect' : ''}`} onClick={handleClick}>{isConnected ? 'Conntect to 8454 Port' : 'Conntect to hardhat' }</button>
        {isConnected && (
          <ul className={`navigation-menu-container ${animationStep === 1 ? 'show' : ''}`}>
            <li>
              <Link to="/accounts">
                <FaTags className='navigation-menu-icon' />
                <label>Accounts</label>
              </Link>
            </li>
            <li>
              <Link to="/contract" >
              <FaCog className='navigation-menu-icon'/>
              <label>Contract</label>
              </Link>
            </li> 
            <li>
              <Link to="/logs" >
              <FaCog className='navigation-menu-icon'/>
              <label>Logs</label>
              </Link>
            </li>           
          </ul>)}
      </div>
      <div className='navigation-group-container' style={{ verticalAlign: 'top' , top: '20px'}}>
          <button className='navigation-buttom' type='button' onClick={(e) => setIsNavigitonBar(!isOpenNavigitonBar)}>
          {isOpenNavigitonBar ? <FaTimes /> : <FaChevronRight />} {/* Toggle icon */}
          </button>
      </div>
      {isOpenNavigitonBar? (<div className='navigation-close' onClick={(e) => setIsNavigitonBar(!isOpenNavigitonBar)}></div>) : (null)}
    </div>
  );
};

export default NavigationBar;