import React from "react";

function DashboardCard({ title, value, unit, status }) {
  return (
    <div className={`card ${status ? `card-${status}` : ""}`}>
      <h2>{title}</h2>

      {React.isValidElement(value) ? (
        <div className="card-value-wrap">
          {value}
        </div>
      ) : (
        <div className="card-value-wrap">
          <span className="card-value">{value}</span>
          {unit && <span className="card-unit">{unit}</span>}
        </div>
      )}
    </div>
  );
}

export default DashboardCard;