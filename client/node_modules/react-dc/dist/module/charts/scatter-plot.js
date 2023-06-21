var _class;

import React from 'react';
import dc from 'dc';
import BaseChart from './base-chart';
import coordinateGridMixin from '../mixins/coordinate-grid-mixin';
import scatterMixin from '../mixins/scatter-mixin';

let ScatterPlot = scatterMixin(_class = coordinateGridMixin(_class = class ScatterPlot extends BaseChart {

  componentDidMount() {
    console.log('scatter');
    this.chart = dc.scatterPlot(this.chart);
    this.configure();
    this.chart.render();
  }
}) || _class) || _class;

ScatterPlot.displayName = 'ScatterPlot';
export { ScatterPlot as default };