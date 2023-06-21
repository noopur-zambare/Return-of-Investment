var _class;

import React from 'react';
import dc from 'dc';
import BaseChart from './base-chart';
import baseMixin from '../mixins/base-mixin';
import capMixin from '../mixins/cap-mixin';
import colorMixin from '../mixins/color-mixin';
import pieMixin from '../mixins/pie-mixin';

const { bool, number, string } = React.PropTypes;

let PieChart = pieMixin(_class = colorMixin(_class = capMixin(_class = baseMixin(_class = class PieChart extends BaseChart {

  componentDidMount() {
    this.chart = dc.pieChart(this.chart);
    this.configure();
    this.chart.render();
  }
}) || _class) || _class) || _class) || _class;

PieChart.displayName = 'PieChart';
export { PieChart as default };