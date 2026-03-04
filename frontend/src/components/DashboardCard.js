import React from "react";

function DashboardCard({ title, value }) {
  return (
    <div className="card">
      <h2>{title}</h2>
      <p className="card-value">{value}</p>
    </div>
  );
}

export default DashboardCard;