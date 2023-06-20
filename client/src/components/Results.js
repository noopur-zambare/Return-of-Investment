/* Function - Display graphs for F1 score, Recall & Precision over variable training size
        Pass In: validation data
        Pass Out: graphs
    Endfunction */
import React, { useEffect, useState } from 'react';

const Results = () => {
  const [graph, setGraph] = useState(false);
  const [graph1, setGraph_recall] = useState(false);
  const [graph2, setGraph_precision] = useState(false);
  const [f1Score_lg, setF1Score_lg] = useState('');
  const [f1Score_nb, setF1Score_nb] = useState('');
  const [f1Score_rf, setF1Score_rf] = useState('');
  const [f1Score_svc, setF1Score_svc] = useState('');
  const [f1Score_dt, setF1Score_dt] = useState('');

  useEffect(() => {
    fetchF1Score();
  }, []);

  const fetchF1Score = async () => {
    const response = await fetch('/f1score', {
      method: 'POST',
    });
    const data = await response.json();
    setGraph(data.graph);
    setGraph_recall(data.graph1);
    setGraph_precision(data.graph2);
    setF1Score_lg(data.f1_score_lg);
    setF1Score_nb(data.f1_score_nb);
    setF1Score_rf(data.f1_score_rf);
    setF1Score_svc(data.f1_score_svc);
    setF1Score_dt(data.f1_score_dt);

  }

  return (
    <div>
      <h3>Results</h3>
      <img src={graph}/>
      <img src={graph1}/>
      <img src={graph2}/> 
    </div>
  );
};

export default Results;
