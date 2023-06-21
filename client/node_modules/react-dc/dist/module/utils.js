var _extends = Object.assign || function (target) { for (var i = 1; i < arguments.length; i++) { var source = arguments[i]; for (var key in source) { if (Object.prototype.hasOwnProperty.call(source, key)) { target[key] = source[key]; } } } return target; };

export const compose = (...fns) => fns.reduce((f, g) => (...args) => f(g(...args)));

export const intersect = (obj1, obj2) => {
  const o = {};
  Object.entries(keys2).forEach((key, val) => {
    if (obj1.hasOwnProperty(key)) {
      o[key] = val;
    }
  });
  return o;
};

// map over the object and make sure each value is a React PropType
const extractPropTypes = props => Object.entries(props).reduce((acc, [prop, val]) => {
  acc[prop] = val.propTypes || val;
  return acc;
}, {});

export const withProps = propTypes => Component => {
  var _class, _temp;

  return _temp = _class = class extends Component {

    configure() {
      if (super.configure) {
        super.configure();
      }
      Object.entries(this.props).forEach(([prop, val]) => {
        if (propTypes[prop]) {
          if (propTypes[prop].setter) {
            propTypes[prop].setter(this.chart[prop], val);
          } else {
            this.chart[prop](val);
          }
        }
      });
    }
  }, _class.propTypes = _extends({}, Component.propTypes, extractPropTypes(propTypes)), _temp;
};