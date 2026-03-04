import React, { useEffect, useState } from "react";
import "./styles/Dashboard.css";
import { fetchEnergy } from "./services/api";
import DashboardCard from "./components/DashboardCard";

function App() {
  const [energy, setEnergy] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const getData = async () => {
      try {
        const energyData = await fetchEnergy();
        setEnergy(energyData);
      } catch (error) {
        console.error("API 호출 실패:", error);
      } finally {
        setLoading(false);
      }
    };

    getData();
  }, []);

  if (loading || !energy) {
    return <p style={{ textAlign: "center", marginTop: "100px" }}>데이터 불러오는 중...</p>;
  }

  return (
    <div className="app-container">
      <h1 className="main-title">🌱 ESG 스마트 에너지 대시보드</h1>

      <div className="card-grid">
        <DashboardCard title="⚡ 현재 발전량" value={`${energy.power} kW`} />
        <DashboardCard title="🔋 배터리 충전율" value={`${energy.soc} %`} />
        <DashboardCard title="🌿 탄소 절감량" value={`${energy.carbon_saved} kg`} />
      </div>
    </div>
  );
}

export default App;