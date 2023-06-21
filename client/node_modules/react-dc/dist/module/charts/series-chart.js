var _class;

import React from 'react';
import dc from 'dc';
import CompositeChart from './composite-chart';
import seriesMixin from '../mixins/series-mixin';

let SeriesChart = seriesMixin(_class = class SeriesChart extends CompositeChart {

  componentDidMount() {
    this.chart = dc.seriesChart(this.chart);
    this.configure();
    this.chart.render();
  }
}) || _class;

SeriesChart.displayName = 'SeriesChart';
export { SeriesChart as default };