var _class;

import React from 'react';
import dc from 'dc';
import BaseChart from './base-chart';
import coordinateGridMixin from '../mixins/coordinate-grid-mixin';
import stackMixin from '../mixins/stack-mixin';

const { arrayOf, bool, instanceOf, number, object, string } = React.PropTypes;

let CompositeChart = coordinateGridMixin(_class = class CompositeChart extends BaseChart {

  componentDidMount() {
    this.chart = dc.compositeChart(this.chart);
    this.configure();
    this.chart.render();
  }
}) || _class;

CompositeChart.displayName = 'CompositeChart';
export { CompositeChart as default };