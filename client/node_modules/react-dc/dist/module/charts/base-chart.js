import React from 'react';

let BaseChart = class BaseChart extends React.Component {
  render() {
    return React.createElement('div', { ref: chart => this.chart = chart });
  }
};
export { BaseChart as default };