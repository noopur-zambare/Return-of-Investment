/* Function - keeps track of the steps */
       

import React from 'react';
import './ProgressBar.css';

const ProgressBar = ({ steps, currentStep, onNext }) => {
  const stepWidth = 100 / (steps.length - 1);

  return (
    <div className="progress-bar-container">
      <div className="progress-bar">
        {steps.map((step, index) => (
          <div
            className={`step ${index <= currentStep ? 'active' : ''}`}
            key={index}
            style={{ width: `${stepWidth}%` }}
          >
            {index !== steps.length - 1 && (
              <div className="step-line"></div>
            )}
            <div className="step-circle">
              <span className="step-label">{step}</span>
            </div>
          </div>
        ))}
      </div>
      <button className="next-button" onClick={onNext}>
        Next
      </button>
    </div>
  );
};

export default ProgressBar;
