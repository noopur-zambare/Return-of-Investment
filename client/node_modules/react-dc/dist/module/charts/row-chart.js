var _class;

import React from 'react';
import dc from 'dc';
import BaseChart from './base-chart';
import baseMixin from '../mixins/base-mixin';
import marginMixin from '../mixins/margin-mixin';
import capMixin from '../mixins/cap-mixin';
import colorMixin from '../mixins/color-mixin';
import rowMixin from '../mixins/row-mixin';

const { any, bool, number, oneOfType } = React.PropTypes;

let RowChart = rowMixin(_class = colorMixin(_class = capMixin(_class = marginMixin(_class = baseMixin(_class = class RowChart extends BaseChart {

  componentDidMount() {
    this.chart = dc.rowChart(this.chart);
    this.configure();
    this.chart.render();
  }
}) || _class) || _class) || _class) || _class) || _class;

RowChart.displayName = 'RowChart';
export { RowChart as default };