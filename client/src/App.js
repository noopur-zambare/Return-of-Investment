import React, { useState } from 'react';
import ImportCSV from './components/ImportCSV';
import MLdropdown from './components/MLdropdown';
import ProgressBar from './components/ProgressBar';
import Results from './components/Results';
import './App.css';

function App() {
  
  const [currentStep, setCurrentStep] = useState(0);
  const steps = ['Data', 'ML Model', 'Results', 'Dependency Graphs', 'ROI Analysis'];
  const pages = [ImportCSV, MLdropdown, Results];

  const handleNext = () => {
    if (currentStep < steps.length - 1) {
      setCurrentStep(currentStep + 1);
    }
  };

  const handleModelSelect = (selectedModel) => {
    console.log('Selected model:', selectedModel);
  
  };

  const CurrentPage = pages[currentStep];

  return (
    <div className="App">
      <header className="App-header">
        <div className="app-container">
          <ProgressBar steps={steps} currentStep={currentStep} onNext={handleNext} />
        </div>
      </header>
      <div className="App-content">
        {CurrentPage && (
          <CurrentPage onModelSelect={handleModelSelect} />
        )}
      </div>
    </div>
  );
}

export default App;
