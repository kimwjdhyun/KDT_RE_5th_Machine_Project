import React, { useEffect, useMemo, useState } from "react";
import "./styles/Dashboard.css";
import { fetchDataLog, fetchStats } from "./services/api";
import DashboardCard from "./components/DashboardCard";

import {
  Chart as ChartJS,
  LineElement,
  CategoryScale,
  LinearScale,
  PointElement,
  Legend,
  Tooltip,
  Filler
} from "chart.js";
import { Line } from "react-chartjs-2";

ChartJS.register(LineElement, CategoryScale, LinearScale, PointElement, Legend, Tooltip, Filler);

function getModeMeta(mode) {
  if (mode === 0) {
    return { text: "🔴 긴급절전", pillCls: "mode mode-red", rowCls: "mode-red" };
  }
  if (mode === 1) {
    return { text: "🟡 절약모드", pillCls: "mode mode-yellow", rowCls: "mode-yellow" };
  }
  return { text: "🟢 풀가동", pillCls: "mode mode-green", rowCls: "mode-green" };
}

function formatCarbon(carbon_g) {
  const n = Number(carbon_g) || 0;
  if (n >= 1000) return `${(n / 1000).toFixed(2)} kg`;
  return `${n.toFixed(2)} g`;
}

function formatNumber(n, digits = 2) {
  const x = Number(n);
  if (Number.isNaN(x)) return "-";
  return x.toFixed(digits);
}

function App() {
  const [log, setLog] = useState([]);
  const [latest, setLatest] = useState(null);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [errorMsg, setErrorMsg] = useState("");

  const [dark, setDark] = useState(false);

  const missionPool = useMemo(
    () => [
      "🔋 배터리 지키기: SOC 20% 아래로 떨어지면 절전!",
      "💧 급수 타이밍 체크: 토양습도 40% 기준선 지키기",
      "📈 예측 vs 실제: 오차가 커지면 센서값부터 의심!",
      "🌿 ESG 미션: 오늘 탄소 절감량 1kg 찍어보기",
      "🛰 데이터 수집: 5초 주기 로그가 끊기지 않게!"
    ],
    []
  );

  const [mission, setMission] = useState(missionPool[0]);
  const [popHistoryKey, setPopHistoryKey] = useState("");
  const [prevLatestTs, setPrevLatestTs] = useState("");
  const [prevMode, setPrevMode] = useState(null);
  const [toast, setToast] = useState({ show: false, text: "", type: "info" });

  useEffect(() => {
    const pick = () => {
      const idx = Math.floor(Math.random() * missionPool.length);
      setMission(missionPool[idx]);
    };

    pick();
    const id = setInterval(pick, 25000);
    return () => clearInterval(id);
  }, [missionPool]);

  const showToast = (text, type = "info") => {
    setToast({ show: true, text, type });
    window.clearTimeout(showToast._t);
    showToast._t = window.setTimeout(() => {
      setToast((v) => ({ ...v, show: false }));
    }, 1700);
  };

  const refresh = async () => {
    try {
      setErrorMsg("");

      const [logData, statsData] = await Promise.all([fetchDataLog(), fetchStats()]);
      const safeLog = Array.isArray(logData) ? logData : [];

      setLog(safeLog);

      const newLatest = safeLog.length > 0 ? safeLog[safeLog.length - 1] : null;
      setLatest(newLatest);

      if (newLatest?.timestamp && newLatest.timestamp !== prevLatestTs) {
        setPopHistoryKey(newLatest.timestamp);
        setPrevLatestTs(newLatest.timestamp);
      }

      if (newLatest && typeof newLatest.mode !== "undefined") {
        if (prevMode === null) {
          setPrevMode(newLatest.mode);
        } else if (newLatest.mode !== prevMode) {
          const from = getModeMeta(prevMode).text;
          const to = getModeMeta(newLatest.mode).text;
          showToast(`⚙ 모드 변경: ${from} → ${to}`, "mode");
          setPrevMode(newLatest.mode);
        }
      }

      setStats(statsData || null);
    } catch (e) {
      console.error(e);
      setErrorMsg("백엔드 연결 실패: Flask 서버(5000)가 켜져있는지 확인해줘!");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 5000);
    return () => clearInterval(id);
  }, []);

  const waterNeeded = latest ? Number(latest.soil) < 40 : false;
  const energyLow = latest ? Number(latest.soc) < 20 || Number(latest.mode) === 0 : false;

  const iconKey = latest?.timestamp || "static";

  const systemOnline = latest !== null;
  const sensorActive = log.length > 0;
  const aiActive = !!(latest && latest.pred_1h !== undefined);

  const chartPalette = useMemo(() => {
    if (dark) {
      return {
        power: "#60a5fa",
        pred: "#fca5a5",
        soc: "#86efac",
        soil: "#fbbf24",
        baseline: "#fdba74",
        grid: "rgba(255,255,255,0.12)",
        tick: "rgba(229,231,235,0.85)"
      };
    }

    return {
      power: "#2563eb",
      pred: "#dc2626",
      soc: "#16a34a",
      soil: "#a16207",
      baseline: "#f97316",
      grid: "rgba(15,23,42,0.10)",
      tick: "rgba(15,23,42,0.75)"
    };
  }, [dark]);

  const chartOptions = useMemo(
    () => ({
      responsive: true,
      maintainAspectRatio: false,
      layout: { padding: { bottom: 12, right: 10, left: 6 } },
      plugins: {
        legend: {
          labels: { color: chartPalette.tick, boxWidth: 10, boxHeight: 10 }
        },
        tooltip: { enabled: true }
      },
      scales: {
        x: { ticks: { color: chartPalette.tick }, grid: { color: chartPalette.grid } },
        y: { ticks: { color: chartPalette.tick }, grid: { color: chartPalette.grid } }
      },
      elements: {
        point: { radius: 2.2, hoverRadius: 4 },
        line: { tension: 0.35, borderWidth: 2 }
      }
    }),
    [chartPalette]
  );

  const labels = useMemo(() => log.map((d) => d.timestamp), [log]);

  const powerChart = useMemo(() => {
    return {
      labels,
      datasets: [
        {
          label: "발전량(W)",
          data: log.map((d) => d.solar_power ?? 0),
          borderColor: chartPalette.power,
          backgroundColor: dark ? "rgba(96,165,250,0.10)" : "rgba(37,99,235,0.08)",
          fill: true
        },
        {
          label: "AI 예측(1h, W)",
          data: log.map((d) => d.pred_1h ?? 0),
          borderColor: chartPalette.pred,
          backgroundColor: dark ? "rgba(252,165,165,0.08)" : "rgba(220,38,38,0.06)",
          borderDash: [6, 6],
          fill: true
        }
      ]
    };
  }, [labels, log, chartPalette, dark]);

  const socChart = useMemo(() => {
    return {
      labels,
      datasets: [
        {
          label: "SOC(%)",
          data: log.map((d) => d.soc ?? 0),
          borderColor: chartPalette.soc,
          backgroundColor: dark ? "rgba(134,239,172,0.10)" : "rgba(22,163,74,0.08)",
          fill: true
        }
      ]
    };
  }, [labels, log, chartPalette, dark]);

  const soilChart = useMemo(() => {
    return {
      labels,
      datasets: [
        {
          label: "토양습도(%)",
          data: log.map((d) => d.soil ?? 0),
          borderColor: chartPalette.soil,
          backgroundColor: dark ? "rgba(251,191,36,0.10)" : "rgba(161,98,7,0.08)",
          fill: true
        },
        {
          label: "기준선(40%)",
          data: log.map(() => 40),
          borderColor: chartPalette.baseline,
          borderDash: [5, 5],
          fill: false
        }
      ]
    };
  }, [labels, log, chartPalette, dark]);

  const modeHistory = useMemo(() => {
    const last10 = log.slice(-10);

    return last10
      .map((d) => ({
        timestamp: d.timestamp ?? "-",
        mode: d.mode ?? 2,
        soc: d.soc ?? 0,
        solar_power: d.solar_power ?? 0,
        pred_1h: d.pred_1h ?? 0,
        soil: d.soil ?? 0,
        temperature: d.temperature ?? 0,
        humidity: d.humidity ?? 0,
        light: d.light ?? 0
      }))
      .reverse();
  }, [log]);

  const latestMode = getModeMeta(latest?.mode);

  const Title = ({ emoji, text }) => (
    <span className="card-title-wrap">
      <span key={`${emoji}-${iconKey}`} className="card-icon" aria-hidden="true">
        {emoji}
      </span>
      <span>{text}</span>
    </span>
  );

  if (loading) {
    return <p style={{ textAlign: "center", marginTop: 100 }}>불러오는 중...</p>;
  }

  return (
    <div className={`app-container capture ${dark ? "theme-dark" : "theme-light"}`}>
      <div className="bg-doodles" aria-hidden="true" />

      <div className={`toast ${toast.show ? "toast-show" : ""} toast-${toast.type}`}>
        {toast.text}
      </div>

      <div className="topbar">
        <div className="title-wrap">
          <span className="live-dot" title="LIVE" />
          <h1 className="main-title">🌱 AI · ESG 스마트 에너지 대시보드</h1>
          <span className="tiny-tag">eco + iot + ml</span>

          <span className="mission-badge" title="오늘의 미션">
            {mission}
          </span>
        </div>

        <button
          className="theme-toggle"
          type="button"
          onClick={() => setDark((v) => !v)}
          aria-label="Toggle dark mode"
          title="다크모드"
        >
          {dark ? "🌙" : "☀️"}
        </button>
      </div>

      <div className="system-status">
        <span className={`status-pill ${systemOnline ? "status-on" : "status-off"}`}>
          {systemOnline ? "🟢 System Online" : "🔴 System Offline"}
        </span>

        <span className={`status-pill ${sensorActive ? "status-on" : "status-off"}`}>
          {sensorActive ? "📡 Sensor Stream Active" : "📡 Sensor Waiting"}
        </span>

        <span className={`status-pill ${aiActive ? "status-on" : "status-off"}`}>
          {aiActive ? "🤖 AI Prediction Running" : "🤖 AI Idle"}
        </span>
      </div>

      {errorMsg && <div className="error-banner">⚠ {errorMsg}</div>}

      {(waterNeeded || energyLow) && (
        <div className="status-badges">
          {waterNeeded && <div className="badge badge-water">💧 급수 필요! (토양습도 40% 미만)</div>}
          {energyLow && <div className="badge badge-energy">⚡ 에너지 부족! (SOC 20% 미만)</div>}
        </div>
      )}

      <div className="main-layout">
        <div className="left-panel">
          <DashboardCard
            title={<Title emoji="⚡" text="현재 발전량" />}
            value={latest ? `${formatNumber(latest.solar_power)} W` : "-"}
          />
          <DashboardCard
            title={<Title emoji="🤖" text="1시간 예측" />}
            value={latest ? `${formatNumber(latest.pred_1h)} W` : "-"}
          />
          <DashboardCard
            title={<Title emoji="🔋" text="배터리 SOC" />}
            value={latest ? `${formatNumber(latest.soc, 0)} %` : "-"}
          />
          <DashboardCard
            title={<Title emoji="🌡" text="토양습도" />}
            value={latest ? `${formatNumber(latest.soil, 0)} %` : "-"}
          />
          <DashboardCard
            title={<Title emoji="🌤" text="온도" />}
            value={latest ? `${formatNumber(latest.temperature, 1)} °C` : "-"}
          />
          <DashboardCard
            title={<Title emoji="💧" text="공기습도" />}
            value={latest ? `${formatNumber(latest.humidity, 1)} %` : "-"}
          />
          <DashboardCard
            title={<Title emoji="💡" text="조도" />}
            value={latest ? `${formatNumber(latest.light, 0)} %` : "-"}
          />
          <DashboardCard
            title={<Title emoji="⚙" text="에너지 모드" />}
            value={latest ? <span className={latestMode.pillCls}>{latestMode.text}</span> : "-"}
          />
          <DashboardCard
            title={<Title emoji="📦" text="총 발전량" />}
            value={stats ? `${formatNumber(stats.total_solar_generation)} W` : "0.00 W"}
          />
          <DashboardCard
            title={<Title emoji="🌿" text="탄소 절감량" />}
            value={stats ? formatCarbon(stats.carbon_reduction_g) : "0.00 g"}
          />
        </div>

        <div className="right-panel">
          <div className="chart-card chart-h">
            <h2 className="section-title">📈 발전량 vs AI 예측</h2>
            <div className="chart-box">
              <Line data={powerChart} options={chartOptions} />
            </div>
          </div>

          <div className="chart-card chart-h">
            <h2 className="section-title">🔋 SOC 변화</h2>
            <div className="chart-box">
              <Line data={socChart} options={chartOptions} />
            </div>
          </div>

          <div className="chart-card chart-h chart-span">
            <h2 className="section-title">🌡 토양습도 모니터링</h2>
            <div className="chart-box">
              <Line data={soilChart} options={chartOptions} />
            </div>
          </div>
        </div>
      </div>

      <div className="bottom-panel">
        <h2 className="section-title">🧾 최근 모드 변화 히스토리 (최신 10개)</h2>

        <div className="chart-card">
          {modeHistory.length === 0 ? (
            <p style={{ opacity: 0.7 }}>
              아직 데이터가 없습니다. Flask 시리얼 수신이 정상인지 확인해주세요!
            </p>
          ) : (
            <ul className="history-list">
              {modeHistory.map((row) => {
                const m = getModeMeta(row.mode);
                const isNew = row.timestamp === popHistoryKey;

                return (
                  <li
                    className={`history-item ${m.rowCls} ${isNew ? "history-pop" : ""}`}
                    key={row.timestamp}
                  >
                    <span className="history-time">
                      {row.timestamp}
                      {isNew && <span className="new-badge">NEW ✨</span>}
                    </span>

                    <strong className="history-mode">{m.text}</strong>

                    <span className="history-meta">
                      SOC {formatNumber(row.soc, 0)}% · 발전 {formatNumber(row.solar_power)}W · 예측{" "}
                      {formatNumber(row.pred_1h)}W · 토양 {formatNumber(row.soil, 0)}% · 온도{" "}
                      {formatNumber(row.temperature, 1)}°C · 습도 {formatNumber(row.humidity, 1)}% · 조도{" "}
                      {formatNumber(row.light, 0)}%
                    </span>
                  </li>
                );
              })}
            </ul>
          )}
        </div>

        <p className="footer-note">5초마다 자동 갱신됩니다 ⏱</p>
      </div>
    </div>
  );
}

export default App;