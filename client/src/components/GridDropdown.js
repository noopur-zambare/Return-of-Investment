/* 
Component - Enables selection of ML model
Functionality
- supports workking of MLdropdown.js
*/

import React, { useState } from 'react';
import './GridDropdown.css'

const Dropdown = ({ options, onSelect }) => {
  const [selectedOption, setSelectedOption] = useState('');

  const handleSelect = (option) => {
    setSelectedOption(option.value);
    onSelect(option.value);
  };

  return (
    <div className="grid-dropdown">
      {options.map((option) => (
        <div
          key={option.value}
          className={`grid-option ${selectedOption === option.value ? 'selected' : ''}`}
          onClick={() => handleSelect(option)}
        >
          {option.label}
        </div>
      ))}
    </div>
  );
};

export default Dropdown;
