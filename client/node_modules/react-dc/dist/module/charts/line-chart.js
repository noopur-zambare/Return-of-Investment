var _class;

import React from 'react';
import dc from 'dc';
import BaseChart from './base-chart';
import coordinateGridMixin from '../mixins/coordinate-grid-mixin';
import stackMixin from '../mixins/stack-mixin';
import lineMixin from '../mixins/line-mixin';

const { arrayOf, bool, func, number, oneOfType, shape, string } = React.PropTypes;

let LineChart = stackMixin(_class = coordinateGridMixin(_class = lineMixin(_class = class LineChart extends BaseChart {

  componentDidMount() {
    this.chart = dc.lineChart(this.chart);
    this.configure();
    this.chart.render();
  }
}) || _class) || _class) || _class;

LineChart.displayName = 'LineChart';
export { LineChart as default };