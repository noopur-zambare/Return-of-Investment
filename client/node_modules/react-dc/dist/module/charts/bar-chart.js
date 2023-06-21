var _class;

import React from 'react';
import dc from 'dc';
import BaseChart from './base-chart';
import coordinateGridMixin from '../mixins/coordinate-grid-mixin';
import stackMixin from '../mixins/stack-mixin';
import barMixin from '../mixins/bar-mixin';

let BarChart = barMixin(_class = stackMixin(_class = coordinateGridMixin(_class = class BarChart extends BaseChart {

  componentDidMount() {
    this.chart = dc.barChart(this.chart);
    this.configure();
    this.chart.render();
  }
}) || _class) || _class) || _class;

BarChart.displayName = 'BarChart';
export { BarChart as default };