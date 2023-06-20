/* Function - Filters out columns as per requirements
        Pass In: takes input from user via multi-selection-bar 
        Pass Out: selected columns are stored at backend,
                  once the apply button is clicked, it freezes the selection and disables further chnages in multi-selection-bar
    Endfunction */

import React, { Component } from "react";
import ReactDOM from "react-dom";
import { default as ReactSelect, components } from "react-select";
import { parse } from "papaparse"; 
import "./ColumnFilter.css";

const Option = (props) => {
  const { value, label } = props.data;

  if (value === "select-all") {
    return (
      <components.Option {...props}>
        <div style={{ display: "flex", alignItems: "center" }}>
          <input
            type="checkbox"
            checked={props.isSelected}
            onChange={() => null}
          />
          <label
            style={{
              color: "black",
              marginLeft: "8px",
              textAlign: "left",
              flex: "1",
              fontSize: "14px",
            }}
          >
            {label}
          </label>
        </div>
      </components.Option>
    );
  } else {
    return (
      <components.Option {...props}>
        <div style={{ display: "flex", alignItems: "center" }}>
          <input
            type="checkbox"
            checked={props.isSelected}
            onChange={() => null}
          />
          <label
            style={{
              color: "black",
              marginLeft: "8px",
              textAlign: "left",
              flex: "1",
              fontSize: "14px",
            }}
          >
            {props.label}
          </label>
        </div>
      </components.Option>
    );
  }
};

const customStyles = {
  control: (provided) => ({
    ...provided,
    backgroundColor: "white",
    color: "black",
    fontSize: "16px",
  }),
  option: (provided, state) => ({
    ...provided,
    backgroundColor: state.isFocused ? "#e6f7ff" : "transparent", 
    color: "black",
    fontSize: "14px",
  }),
};

export default class Filter extends Component {
  constructor(props) {
    super(props);
    this.state = {
      optionSelected: null,
      columnNames: [],
      csvData: [],
      frozenColumns: [], 
    };
  }

  componentDidMount() {
  
    const csvFilePath = process.env.PUBLIC_URL + "/static/data.csv"

    fetch(csvFilePath)
      .then((response) => response.text())
      .then((csvText) => {
        const { data } = parse(csvText, { header: true });
        const columnNames = Object.keys(data[0]);
        const csvData = data.map((row) => Object.values(row));
        this.setState({ columnNames, csvData });
      })
      .catch((error) => {
        console.error("Error loading CSV file:", error);
      });
  }

  handleChange = (selected) => {
    const { columnNames } = this.state;

    if (
      selected &&
      selected.some((option) => option.value === "select-all")
    ) {

      const selectedOptions = [
        { value: "select-all", label: "Select All" },
        ...columnNames.map((name) => ({ value: name, label: name })),
      ];
      this.setState({
        optionSelected: selectedOptions,
      });
    } else {
      this.setState({
        optionSelected: selected,
      });
    }
  };

  handleSelectAll = () => {
    const { columnNames } = this.state;
    const selectedOptions = [
      { value: "select-all", label: "Select All" },
      ...columnNames.map((name) => ({ value: name, label: name })),
    ];
    this.setState({
      optionSelected: selectedOptions,
    });
  };

  handleApply = () => {
    const { optionSelected } = this.state;
    const frozenColumns = optionSelected
      .filter((option) => option.value !== "select-all")
      .map((option) => option.value);
  

    fetch('/filter_columns', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ columns: frozenColumns }),
    })
      .then(response => response.json())
      .then(data => {
        console.log(data);
      })
      .catch(error => {
        console.error('Error sending column values:', error);
      });
  
    this.setState({
      frozenColumns,
    });
  };
  

  render() {
    const { columnNames, optionSelected, csvData, frozenColumns } = this.state;
    const topRows = csvData.slice(0, 3); 
  
    return (
      <div>
        {columnNames.length > 0 && (
          <ReactSelect
            styles={customStyles}
            options={[
              { value: "select-all", label: "Select All" },
              ...columnNames.map((name) => ({ value: name, label: name })),
            ]}
            isMulti
            closeMenuOnSelect={false}
            hideSelectedOptions={false}
            components={{
              Option,
            }}
            onChange={this.handleChange}
            allowSelectAll={true}
            value={optionSelected}
            placeholder="Select columns..."
            isDisabled={!!frozenColumns.length} 
          />
        )}
  
        <div className="button-container">
          <button onClick={this.handleApply}>Apply</button>
        </div>
  
        {topRows.length > 0 && (
          <table className="csv-table">
            <thead>
              <tr>
                {columnNames.map((name) => (
                  <th
                    key={name}
                    className={frozenColumns.includes(name) ? "frozen" : ""}
                  >
                    {name}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {topRows.map((row, index) => (
                <tr key={index}>
                  {row.map((value, colIndex) => (
                    <td
                      key={colIndex}
                      className={frozenColumns.includes(columnNames[colIndex]) ? "frozen" : ""}
                    >
                      {value}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    );
  }
  
}

const rootElement = document.getElementById("root");
ReactDOM.render(<Filter />, rootElement);
