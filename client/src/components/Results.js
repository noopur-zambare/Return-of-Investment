import React, { useEffect, useState } from 'react';

const Results = () => {
  const [graph, setGraph] = useState(false);
  const [graph1, setGraphRecall] = useState(false);
  const [graph2, setGraphPrecision] = useState(false);

  useEffect(() => {
    fetchF1Score();
  }, []);

  const fetchF1Score = async () => {
    const response = await fetch('http://34.201.93.116:8080//f1score', {
      method: 'POST',
    });
    const data = await response.json();
    setGraph(data.graph);
    setGraphRecall(data.graph1);
    setGraphPrecision(data.graph2);
  };

  return (
    <div>
      <h3>Results</h3>
      <div className="graph-container">
        <div className="graph">
          <h5>F1 Score Graph</h5>
          <img src={graph} alt="F1 Score Graph" />
        </div>
        <div className="graph">
          <h5>Recall Graph</h5>
          <img src={graph1} alt="Recall Graph" />
        </div>
      </div>
      <div className="graph-container">
        <div className="graph">
          <h5>Precision Graph</h5>
          <img src={graph2} alt="Precision Graph" />
        </div>
      </div>
    </div>
  );
};

export default Results;
