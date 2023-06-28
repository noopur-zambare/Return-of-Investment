/* Function - Imports test and train data, display plots, enables column selection
        Pass In: train data & test data (.csv)
        Pass Out: interactive graphs, multiselection bar to choose columns of data
    Endfunction */

import React, { useState } from 'react';
import axios from 'axios';
import './ImportCSV.css';
import Slider from 'rc-slider';
import 'rc-slider/assets/index.css';
import Charts from './ChartComponent';
import Filter from './ColumnFilter';

function ImportCSV() {
  const [trainData, setTrainData] = useState(null);
  const [trainingSize, setTrainingSize] = useState(0.8);
  const [train_data_RowCount, settrain_data_RowCount] = useState(null);
  const [train_data_ColCount, settrain_data_ColCount] = useState(null);
  const [trainDataUploaded, setTrainDataUploaded] = useState(false);
  const [showColumnSelection, setShowColumnSelection] = useState(false); 

  const [testData, setTestData] = useState(null);
  const [test_data_RowCount, settest_data_RowCount] = useState(null);
  const [test_data_ColCount, settest_data_ColCount] = useState(null);
  const [testDataUploaded, setTestDataUploaded] = useState(false);

  const handleTrainDataUpload = (event) => {
    setTrainData(event.target.files[0]);
  };

  const handleTrainingSizeChange = (value) => {
    setTrainingSize(value);
  };

  const handleTestDataUpload = (event) => {
    setTestData(event.target.files[0]);
  };

  const handleGraphUpload = () => {
    const formData = new FormData();
    formData.append('file', trainData);
    formData.append('training_size', trainingSize);  
  
    axios
      .post('https://34.201.93.116:8080/upload/train_data', formData)
      .then((response) => {
        console.log(response.data);
        if (response.data.success) {
          settrain_data_RowCount(response.data.rows);
          settrain_data_ColCount(response.data.columns);
          setTrainData(response.data.csv_data);
          setTrainDataUploaded(true);
          setShowColumnSelection(true); 
        }
      })
      .catch((error) => {
        console.error(error);
      });
  };
  

  const handleTestData = () => {
    const formData = new FormData();
    formData.append('file', testData);

    axios
      .post('http://34.201.93.116:8080/upload/test_data', formData)
      .then((response) => {
        console.log(response.data);
        if (response.data.success) {
          settest_data_RowCount(response.data.rows);
          settest_data_ColCount(response.data.columns);
          setTestDataUploaded(true);
        }
      })
      .catch((error) => {
        console.error(error);
      });
  };

  const handlePreprocessData = () => {
    axios
      .post('http://34.201.93.116:8080//trim_data', { trainingSize})
      .then((response) => {
        console.log(response.data);
        window.alert('Trimmed Data Saved at Backend!');
      })
      .catch((error) => {
        console.error(error);
      });
  };
  

  return (
    <div className="container">
      <div className="section">
        <h2>Train Data</h2>
        <div className="slider-container">
          <p style={{ fontFamily: 'inherit', fontSize: '16px' }}>Training Size: {trainingSize}</p>
          <Slider
            min={0.1}
            max={1}
            step={0.1}
            value={trainingSize}
            onChange={handleTrainingSizeChange}
          />
        </div>
        <div className="input-section">
          <input type="file" onChange={handleTrainDataUpload} />
          <button onClick={handleGraphUpload}>Upload</button>
        </div>
        
        {trainDataUploaded && (
          <div className="result">
            <p>Data Rows: {train_data_RowCount}</p>
            <p>Data Columns: {train_data_ColCount}</p>
            
          </div>
        )}
      </div>

      <div className="section">
        <h2>Test Data</h2>
        <div className="input-section">
          <input type="file" onChange={handleTestDataUpload} />
          <button onClick={handleTestData}>Upload</button>
        </div>
        {test_data_RowCount !== null && test_data_ColCount !== null && (
          <div className="result">
            <p>Data Rows: {test_data_RowCount}</p>
            <p>Data Columns: {test_data_ColCount}</p>
          </div>
        )}
        {testDataUploaded && <h6>Test Data Uploaded</h6>}
      </div>
      <div className="graphs-container">
      {showColumnSelection && (
        <div className="graph-section">
          <h4>Select the required columns</h4>
          <Filter trainData={trainData} /> 
        </div>
      )}</div>

      <div className="graphs-container">
        {trainDataUploaded && (
          <div className="graph-section">
            <Charts />

          </div>
        )}
      </div>
      <button className="button" onClick={handlePreprocessData}>
        Trim & Preprocess Data
      </button>
    </div>
  );
}

export default ImportCSV;
