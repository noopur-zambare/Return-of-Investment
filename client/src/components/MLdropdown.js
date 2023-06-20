/* Function - ML model selector
        Pass In: Data (.csv) -> dataframe
        Pass Out: f1 score, accuracy, confusion matrix, execution time
    Endfunction */

import React, { useState } from 'react';
import Dropdown from './GridDropdown';
import axios from 'axios';
import './MLdropdown.css'


const MLdropdown = ({ onModelSelect, trainData, testData }) => {
  const options = [
    { label: 'Logistic Regression', value: 'logistic_regression' },
    { label: 'Naive Bayes', value: 'naive_bayes' },
    { label: 'Random Forest', value: 'random_forest' },
    { label: 'Support Vector Machine', value: 'support_vector_machine' },
    { label: 'Decision Tree', value: 'decision_tree' },
  ];
  const [report, setReport] = useState(null);
  const [accuracy, setAccuracy] = useState(null);
  const [stopTime, setStopTime] = useState(null);
  const [f1_score, setF1Score] = useState(null);
  const [graph, setGraph] = useState(false); 
  const [confusionMatrix, setConfusionMatrix] = useState(false); 


  const handleOptionSelect = async (selectedValue) => {
    console.log('Selected option:', selectedValue);
    onModelSelect(selectedValue);
  
    if (selectedValue === 'logistic_regression') {
      try {
        const response = await axios.post('/logistic-regression');
        console.log(response.data);
        if (response.data.success) {
          setReport(response.data.report);
          setAccuracy(response.data.accuracy);
          setStopTime(response.data.stop);
          setGraph(response.data.graph)
          setConfusionMatrix(response.data.cm)
          setF1Score(response.data.f1)
        }
      } catch (error) {
        console.error(error);
      }
    }
    else if (selectedValue === 'naive_bayes') {
      try {
        const response = await axios.post('/naive-bayes');
        console.log(response.data);
        if (response.data.success) {
          setReport(response.data.report);
          setAccuracy(response.data.accuracy);
          setStopTime(response.data.stop);
          setGraph(response.data.graph)
          setConfusionMatrix(response.data.cm)
          setF1Score(response.data.f1)
          
        }
      } catch (error) {
        console.error(error);
      }
    }
    else if (selectedValue === 'random_forest') {
      try {
        const response = await axios.post('/random-forest');
        console.log(response.data);
        if (response.data.success) {
          setReport(response.data.report);
          setAccuracy(response.data.accuracy);
          setStopTime(response.data.stop);
          setGraph(response.data.graph)
          setConfusionMatrix(response.data.cm)
          setF1Score(response.data.f1)
        }
      } catch (error) {
        console.error(error);
      }
    }
    else if (selectedValue === 'support_vector_machine') {
      try {
        const response = await axios.post('/support-vector-machine');
        console.log(response.data);
        if (response.data.success) {
          setReport(response.data.report);
          setAccuracy(response.data.accuracy);
          setStopTime(response.data.stop);
          setGraph(response.data.graph)
          setConfusionMatrix(response.data.cm)
          setF1Score(response.data.f1)
        }
      } catch (error) {
        console.error(error);
      } 
    }
    else if (selectedValue === 'decision_tree') {
      try {
        const response = await axios.post('/decision-tree');
        console.log(response.data);
        if (response.data.success) {
          setReport(response.data.report);
          setAccuracy(response.data.accuracy);
          setStopTime(response.data.stop);
          setGraph(response.data.graph)
          setConfusionMatrix(response.data.cm)
          setF1Score(response.data.f1)
        }
      } catch (error) {
        console.error(error);
      }
    }
  };
  

  
  

  return (
    <div>
      <h1>ML Model</h1>
      <Dropdown options={options} onSelect={handleOptionSelect} />
      
      {report && (
        <div className="classification-report">
          
          <div className="report-content">
         
            <pre><u><b>Training Accuracy</b></u>: {accuracy}</pre>
            <pre><u><b>Execution Time</b></u>: {stopTime} seconds</pre>
            <pre><u><b>F1 Score</b></u>: {f1_score} </pre>
            <pre><u><b>Classification Report</b></u><br></br><br></br>{report}</pre>
       

            <img src={graph}/>
            <img src={confusionMatrix}/>
          
          </div>
        </div>
      )}
    </div>
  );
  
};

export default MLdropdown;
