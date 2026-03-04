import React, { useEffect, useState } from "react";
import {
  Chart as ChartJS,
  LineElement,
  CategoryScale,
  LinearScale,
  PointElement,
  Legend,
  Tooltip
} from "chart.js";
import { Line } from "react-chartjs-2";

ChartJS.register(
  LineElement,
  CategoryScale,
  LinearScale,
  PointElement,
  Legend,
  Tooltip
);

function App() {
  const [data, setData] = useState([]);
  const [stats, setStats] = useState({});
  const [latest, setLatest] = useState(null);

  // 📡 5초마다 데이터 가져오기
  useEffect(() => {
    fetchData();
    fetchStats();

    const interval = setInterval(() => {
      fetchData();
      fetchStats();
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  const fetchData = async () => {
    try {
      const res = await fetch("http://127.0.0.1:5000/data");
      const json = await res.json();
      setData(json);
      if (json.length > 0) {
        setLatest(json[json.length - 1]);
      }
    } catch (err) {
      console.error(err);
    }
  };

  const fetchStats = async () => {
    try {
      const res = await fetch("http://127.0.0.1:5000/stats");
      const json = await res.json();
      setStats(json);
    } catch (err) {
      console.error(err);
    }
  };

  const getModeText = (mode) => {
    if (mode === 0) return "🔴 긴급절전";
    if (mode === 1) return "🟡 절약모드";
    return "🟢 풀가동";
  };

  // 📈 차트 데이터 생성
  const labels = data.map(d => d.timestamp);

  const powerData = {
    labels,
    datasets: [
      {
        label: "발전량",
        data: data.map(d => d.power),
        borderColor: "blue"
      },
      {
        label: "AI 예측",
        data: data.map(d => d.pred_1h),
        borderColor: "red"
      }
    ]
  };

  const socData = {
    labels,
    datasets: [
      {
        label: "SOC",
        data: data.map(d => d.soc),
        borderColor: "green"
      }
    ]
  };

  const soilData = {
    labels,
    datasets: [
      {
        label: "토양습도",
        data: data.map(d => d.soil),
        borderColor: "brown"
      },
      {
        label: "기준선 (40%)",
        data: data.map(() => 40),
        borderColor: "orange",
        borderDash: [5, 5]
      }
    ]
  };

  return (
    <div style={{ padding: 30, fontFamily: "sans-serif" }}>
      <h1>🌱 AI 스마트 에너지 대시보드</h1>

      {latest && (
        <div style={{ marginBottom: 30 }}>
          <h2>📊 실시간 데이터</h2>
          <p>⚡ 발전량: {latest.power} kW</p>
          <p>🔋 SOC: {latest.soc}%</p>
          <p>🌡 토양습도: {latest.soil}%</p>
          <p>🤖 AI 1시간 예측: {latest.pred_1h} kW</p>
          <p>⚙ 에너지 모드: {getModeText(latest.mode)}</p>
        </div>
      )}

      <div>
        <h2>🌍 ESG 통계</h2>
        <p>누적 발전량: {stats.total_generation || 0} kW</p>
        <p>탄소 절감량: {stats.carbon_reduction_g || 0} g</p>
      </div>

      <h2>📈 발전량 vs AI 예측</h2>
      <Line data={powerData} />

      <h2>🔋 SOC 변화</h2>
      <Line data={socData} />

      <h2>🌡 토양습도 모니터링</h2>
      <Line data={soilData} />
    </div>
  );
}

export default App;