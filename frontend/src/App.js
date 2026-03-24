import React, { useEffect, useMemo, useState } from "react";
import "./styles/Dashboard.css";
import { fetchDataLog, fetchStats, fetchLatest, fetchHealth } from "./services/api";
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
  if (Number(mode) === 0) {
    return {
      text: "긴급절전",
      emoji: "🔴",
      pillCls: "mode mode-red",
      rowCls: "mode-red"
    };
  }
  if (Number(mode) === 1) {
    return {
      text: "절약모드",
      emoji: "🟡",
      pillCls: "mode mode-yellow",
      rowCls: "mode-yellow"
    };
  }
  return {
    text: "풀가동",
    emoji: "🟢",
    pillCls: "mode mode-green",
    rowCls: "mode-green"
  };
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

function getCardStatus(type, latest) {
  if (!latest) return "";

  if (type === "soc") {
    const soc = Number(latest.soc);
    if (soc < 20) return "danger";
    if (soc < 40) return "warning";
    return "normal";
  }

  if (type === "soil") {
    const soil = Number(latest.soil);
    if (soil < 40) return "danger";
    if (soil < 55) return "warning";
    return "normal";
  }

  if (type === "pump") {
    return Number(latest.pump) === 1 ? "normal" : "warning";
  }

  if (type === "led") {
    return Number(latest.led) === 1 ? "normal" : "warning";
  }

  if (type === "battery_voltage") {
    const bv = Number(latest.battery_voltage);
    if (bv < 11.2) return "danger";
    if (bv < 12.0) return "warning";
    return "normal";
  }

  if (type === "pred_1h") {
    const pred = Number(latest.pred_1h);
    if (pred <= 0) return "warning";
    return "normal";
  }

  if (type === "ai_confidence") {
    const pred = Number(latest?.pred_1h || 0);
    if (pred <= 0) return "warning";
    return "normal";
  }

  return "";
}

function App() {
  const [log, setLog] = useState([]);
  const [latest, setLatest] = useState(null);
  const [stats, setStats] = useState(null);
  const [health, setHealth] = useState(null);
  const [loading, setLoading] = useState(true);
  const [errorMsg, setErrorMsg] = useState("");
  const [dark, setDark] = useState(false);

  const missionPool = useMemo(
    () => [
      "🔋 배터리 지키기: SOC 20% 아래로 떨어지면 절전!",
      "💧 급수 타이밍 체크: 토양습도 40% 기준선 지키기",
      "📈 예측 vs 실제: 오차가 커지면 센서값부터 의심!",
      "🌿 ESG 미션: 오늘 탄소 절감량 1kg 찍어보기",
      "🛰 데이터 수집: 10초 주기 로그가 끊기지 않게!"
    ],
    []
  );

  const [mission, setMission] = useState(missionPool[0]);
  const [popHistoryKey, setPopHistoryKey] = useState("");
  const [prevLatestTs, setPrevLatestTs] = useState("");
  const [prevMode, setPrevMode] = useState(null);
  const [toast, setToast] = useState({ show: false, text: "", type: "info" });
  const [expandedHistory, setExpandedHistory] = useState({});

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

  const toggleHistory = (key) => {
    setExpandedHistory((prev) => ({
      ...prev,
      [key]: !prev[key]
    }));
  };

  const refresh = async () => {
    try {
      setErrorMsg("");

      const [logData, statsData, latestData, healthData] = await Promise.all([
        fetchDataLog(),
        fetchStats(),
        fetchLatest(),
        fetchHealth()
      ]);

      const safeLog = Array.isArray(logData) ? logData : [];
      setLog(safeLog);

      const newLatest =
        latestData && Object.keys(latestData).length > 0
          ? latestData
          : safeLog.length > 0
            ? safeLog[safeLog.length - 1]
            : null;

      setLatest(newLatest);
      setStats(statsData || null);
      setHealth(healthData || null);

      if (newLatest?.timestamp && newLatest.timestamp !== prevLatestTs) {
        setPopHistoryKey(newLatest.timestamp);
        setPrevLatestTs(newLatest.timestamp);
      }

      if (newLatest && typeof newLatest.mode !== "undefined") {
        if (prevMode === null) {
          setPrevMode(newLatest.mode);
        } else if (Number(newLatest.mode) !== Number(prevMode)) {
          const from = `${getModeMeta(prevMode).emoji} ${getModeMeta(prevMode).text}`;
          const to = `${getModeMeta(newLatest.mode).emoji} ${getModeMeta(newLatest.mode).text}`;
          showToast(`⚙ 모드 변경: ${from} → ${to}`, "mode");
          setPrevMode(newLatest.mode);
        }
      }
    } catch (e) {
      console.error(e);
      setErrorMsg("백엔드 연결 실패: Flask 서버(5000)가 켜져있는지 확인해줘!");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 10000);
    return () => clearInterval(id);
  }, []);

  const waterNeeded = latest ? Number(latest.soil) < 40 || Number(latest.water_alert) === 1 : false;
  const energyLow = latest ? Number(latest.soc) < 20 || Number(latest.mode) === 0 : false;

  const iconKey = latest?.timestamp || "static";

  const systemOnline = health?.status === "ok";
  const sensorActive = !!health?.latest_exists;
  const aiActive = !!health?.xgb_ready;

  const predGap = latest
    ? Math.abs(Number(latest.solar_power || 0) - Number(latest.pred_1h || 0))
    : null;

  const predictionPending =
    latest && Number(latest.pred_1h) === 0 && !!health?.xgb_ready;

  const predictionInsight = useMemo(() => {
    if (!latest) return "-";

    const current = Number(latest.solar_power || 0);
    const pred = Number(latest.pred_1h || 0);
    const diff = pred - current;

    if (!health?.xgb_ready) return "AI 모델 비활성";
    if (pred === 0) return "예측 준비 중";

    if (diff > 0.2) return "📈 1시간 후 발전 증가 예상";
    if (diff < -0.2) return "📉 1시간 후 발전 감소 예상";
    return "➡️ 1시간 후 발전 유지 예상";
  }, [latest, health]);

  const predGapStatus = useMemo(() => {
    if (predGap === null) return "-";
    if (predGap < 0.2) return "🟢 안정";
    if (predGap < 0.5) return "🟡 변동 주의";
    return "🔴 급변 가능";
  }, [predGap]);

  const aiConfidence = useMemo(() => {
    if (!health?.xgb_ready || !latest) {
      return { text: "비활성", score: "-" };
    }

    const pred = Number(latest.pred_1h || 0);
    const current = Number(latest.solar_power || 0);
    const gap = Math.abs(pred - current);

    if (pred === 0) {
      return { text: "계산 대기", score: "-" };
    }

    if (gap < 0.2) {
      return { text: "높음", score: "90" };
    }

    if (gap < 0.5) {
      return { text: "보통", score: "75" };
    }

    return { text: "낮음", score: "55" };
  }, [latest, health]);

  const systemSummary = useMemo(() => {
    if (!latest) return "데이터 수신 대기 중";

    const parts = [];

    if (Number(latest.soc) >= 40) parts.push("배터리 안정");
    else if (Number(latest.soc) >= 20) parts.push("배터리 주의");
    else parts.push("배터리 부족");

    if (Number(latest.soil) >= 40) parts.push("토양 양호");
    else parts.push("급수 필요");

    if (Number(latest.pred_1h) > Number(latest.solar_power) + 0.2) {
      parts.push("발전 증가 예상");
    } else if (Number(latest.pred_1h) < Number(latest.solar_power) - 0.2) {
      parts.push("발전 감소 예상");
    } else if (Number(latest.pred_1h) > 0) {
      parts.push("발전 유지 예상");
    }

    return parts.join(" · ");
  }, [latest]);

  const chartPalette = useMemo(() => {
    if (dark) {
      return {
        power: "#60a5fa",
        pred: "#fca5a5",
        gap: "#f59e0b",
        soc: "#86efac",
        soil: "#fbbf24",
        baseline: "#fdba74",
        temp: "#c084fc",
        hum: "#22d3ee",
        grid: "rgba(255,255,255,0.12)",
        tick: "rgba(229,231,235,0.85)"
      };
    }

    return {
      power: "#2563eb",
      pred: "#dc2626",
      gap: "#f59e0b",
      soc: "#16a34a",
      soil: "#a16207",
      baseline: "#f97316",
      temp: "#7c3aed",
      hum: "#0891b2",
      grid: "rgba(15,23,42,0.10)",
      tick: "rgba(15,23,42,0.75)"
    };
  }, [dark]);

  const chartOptions = useMemo(
    () => ({
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      interaction: {
        mode: "index",
        intersect: false
      },
      layout: { padding: { bottom: 4, right: 6, left: 4 } },
      plugins: {
        legend: {
          labels: { color: chartPalette.tick, boxWidth: 10, boxHeight: 10 }
        },
        tooltip: { enabled: true }
      },
      scales: {
        x: {
          ticks: {
            color: chartPalette.tick,
            maxTicksLimit: 10
          },
          grid: { color: chartPalette.grid }
        },
        y: {
          ticks: { color: chartPalette.tick },
          grid: { color: chartPalette.grid }
        }
      },
      elements: {
        point: { radius: 2.2, hoverRadius: 4 },
        line: { tension: 0.35, borderWidth: 2 }
      }
    }),
    [chartPalette]
  );

  // 최근 1시간(10초 간격 기준 360개)만 가져오고
  // 그래프에는 최대 60포인트 정도만 샘플링해서 표시
  const displayLog = useMemo(() => {
    if (!Array.isArray(log) || log.length === 0) return [];

    const recent1h = log.slice(-360);
    const step = Math.max(1, Math.ceil(recent1h.length / 60));

    return recent1h.filter((_, i) => i % step === 0);
  }, [log]);

  const labels = useMemo(
    () => displayLog.map((d) => (d.timestamp ? d.timestamp.slice(11, 19) : "-")),
    [displayLog]
  );

  const powerChart = useMemo(() => {
    return {
      labels,
      datasets: [
        {
          label: "발전량(W)",
          data: displayLog.map((d) => d.solar_power ?? 0),
          borderColor: chartPalette.power,
          backgroundColor: dark ? "rgba(96,165,250,0.10)" : "rgba(37,99,235,0.08)",
          fill: true
        },
        {
          label: "AI 예측(1h, W)",
          data: displayLog.map((d) => d.pred_1h ?? 0),
          borderColor: chartPalette.pred,
          backgroundColor: dark ? "rgba(252,165,165,0.08)" : "rgba(220,38,38,0.06)",
          borderDash: [6, 6],
          fill: true
        }
      ]
    };
  }, [labels, displayLog, chartPalette, dark]);

  const errorChart = useMemo(() => {
    return {
      labels,
      datasets: [
        {
          label: "예측 오차(|실제-예측|, W)",
          data: displayLog.map((d) =>
            Math.abs(Number(d.solar_power ?? 0) - Number(d.pred_1h ?? 0))
          ),
          borderColor: chartPalette.gap,
          backgroundColor: dark ? "rgba(245,158,11,0.16)" : "rgba(245,158,11,0.12)",
          fill: true
        }
      ]
    };
  }, [labels, displayLog, chartPalette, dark]);

  const socChart = useMemo(() => {
    return {
      labels,
      datasets: [
        {
          label: "SOC(%)",
          data: displayLog.map((d) => d.soc ?? 0),
          borderColor: chartPalette.soc,
          backgroundColor: dark ? "rgba(134,239,172,0.10)" : "rgba(22,163,74,0.08)",
          fill: true
        }
      ]
    };
  }, [labels, displayLog, chartPalette, dark]);

  const soilChart = useMemo(() => {
    return {
      labels,
      datasets: [
        {
          label: "토양습도(%)",
          data: displayLog.map((d) => d.soil ?? 0),
          borderColor: chartPalette.soil,
          backgroundColor: dark ? "rgba(251,191,36,0.10)" : "rgba(161,98,7,0.08)",
          fill: true
        },
        {
          label: "기준선(40%)",
          data: displayLog.map(() => 40),
          borderColor: chartPalette.baseline,
          borderDash: [5, 5],
          fill: false
        }
      ]
    };
  }, [labels, displayLog, chartPalette, dark]);

  const envChart = useMemo(() => {
    return {
      labels,
      datasets: [
        {
          label: "온도(°C)",
          data: displayLog.map((d) => d.temperature ?? 0),
          borderColor: chartPalette.temp,
          backgroundColor: dark ? "rgba(192,132,252,0.08)" : "rgba(124,58,237,0.08)",
          fill: true
        },
        {
          label: "습도(%)",
          data: displayLog.map((d) => d.humidity ?? 0),
          borderColor: chartPalette.hum,
          backgroundColor: dark ? "rgba(34,211,238,0.08)" : "rgba(8,145,178,0.08)",
          fill: true
        }
      ]
    };
  }, [labels, displayLog, chartPalette, dark]);

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
        light: d.light ?? 0,
        solar_voltage: d.solar_voltage ?? 0,
        solar_current: d.solar_current ?? 0,
        battery_voltage: d.battery_voltage ?? 0,
        battery_current: d.battery_current ?? 0,
        pump: d.pump ?? 0,
        led: d.led ?? 0,
        led_brightness: d.led_brightness ?? 0,
        water_alert: d.water_alert ?? 0
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

        {!!health?.feature_count && (
          <span className="status-pill status-on">
            🧠 XGBoost · {health.feature_count} features
          </span>
        )}
      </div>

      <div className="status-badges">
        <div className="badge badge-energy">🧾 상태 요약: {systemSummary}</div>
      </div>

      {errorMsg && <div className="error-banner">⚠ {errorMsg}</div>}

      {(waterNeeded || energyLow) && (
        <div className="status-badges">
          {waterNeeded && <div className="badge badge-water">💧 급수 필요! (토양습도 40% 미만 또는 water_alert 감지)</div>}
          {energyLow && <div className="badge badge-energy">⚡ 에너지 부족! (SOC 20% 미만 또는 긴급절전 모드)</div>}
        </div>
      )}

      {predictionPending && (
        <div className="status-badges">
          <div className="badge badge-energy">🤖 AI 예측 준비 중이거나 야간 구간일 수 있습니다.</div>
        </div>
      )}

      <div className="main-layout">
        <div className="left-panel">
          <DashboardCard
            title={<Title emoji="⚡" text="현재 발전량" />}
            value={latest ? formatNumber(latest.solar_power) : "-"}
            unit="W"
          />

          <DashboardCard
            title={<Title emoji="🤖" text="1시간 후 예측 발전량" />}
            value={latest ? formatNumber(latest.pred_1h) : "-"}
            unit="W"
            status={getCardStatus("pred_1h", latest)}
          />

          <DashboardCard
            title={<Title emoji="📏" text="현재-예측 차이" />}
            value={predGap !== null ? formatNumber(predGap) : "-"}
            unit="W"
          />

          <DashboardCard
            title={<Title emoji="🎯" text="AI 신뢰도" />}
            value={aiConfidence.score === "-" ? aiConfidence.text : `${aiConfidence.score}% (${aiConfidence.text})`}
            status={getCardStatus("ai_confidence", latest)}
          />

          <DashboardCard
            title={<Title emoji="🧠" text="AI 해석" />}
            value={predictionInsight}
          />

          <DashboardCard
            title={<Title emoji="📊" text="예측 차이 상태" />}
            value={predGapStatus}
          />

          <DashboardCard
            title={<Title emoji="🔋" text="배터리 SOC" />}
            value={latest ? formatNumber(latest.soc, 0) : "-"}
            unit="%"
            status={getCardStatus("soc", latest)}
          />

          <DashboardCard
            title={<Title emoji="🌱" text="토양습도" />}
            value={latest ? formatNumber(latest.soil, 0) : "-"}
            unit="%"
            status={getCardStatus("soil", latest)}
          />

          <DashboardCard
            title={<Title emoji="🌤" text="온도" />}
            value={latest ? formatNumber(latest.temperature, 1) : "-"}
            unit="°C"
          />

          <DashboardCard
            title={<Title emoji="💧" text="공기습도" />}
            value={latest ? formatNumber(latest.humidity, 1) : "-"}
            unit="%"
          />

          <DashboardCard
            title={<Title emoji="💡" text="조도" />}
            value={latest ? formatNumber(latest.light, 0) : "-"}
            unit="%"
          />

          <DashboardCard
            title={<Title emoji="⚙" text="에너지 모드" />}
            value={latest ? <span className={latestMode.pillCls}>{`${latestMode.emoji} ${latestMode.text}`}</span> : "-"}
          />

          <DashboardCard
            title={<Title emoji="🔆" text="태양광 전압" />}
            value={latest ? formatNumber(latest.solar_voltage) : "-"}
            unit="V"
          />

          <DashboardCard
            title={<Title emoji="🔌" text="태양광 전류" />}
            value={latest ? formatNumber(latest.solar_current, 3) : "-"}
            unit="A"
          />

          <DashboardCard
            title={<Title emoji="🔋" text="배터리 전압" />}
            value={latest ? formatNumber(latest.battery_voltage) : "-"}
            unit="V"
            status={getCardStatus("battery_voltage", latest)}
          />

          <DashboardCard
            title={<Title emoji="🔌" text="배터리 전류" />}
            value={latest ? formatNumber(latest.battery_current, 3) : "-"}
            unit="A"
          />

          <DashboardCard
            title={<Title emoji="💧" text="펌프 상태" />}
            value={latest ? (Number(latest.pump) === 1 ? "ON" : "OFF") : "-"}
            status={getCardStatus("pump", latest)}
          />

          <DashboardCard
            title={<Title emoji="💡" text="LED 상태" />}
            value={
              latest
                ? Number(latest.led) === 1
                  ? `ON (${formatNumber(latest.led_brightness, 0)}%)`
                  : "OFF"
                : "-"
            }
            status={getCardStatus("led", latest)}
          />

          <DashboardCard
            title={<Title emoji="🚨" text="급수 경고" />}
            value={latest ? (Number(latest.water_alert) === 1 ? "ALERT" : "정상") : "-"}
            status={latest && Number(latest.water_alert) === 1 ? "danger" : "normal"}
          />

          <DashboardCard
            title={<Title emoji="📦" text="누적 발전 지표" />}
            value={stats ? formatNumber(stats.total_solar_generation) : "0.00"}
            unit="W"
          />

          <DashboardCard
            title={<Title emoji="🌿" text="탄소 절감량" />}
            value={stats ? formatCarbon(stats.carbon_reduction_g) : "0.00 g"}
          />
        </div>

        <div className="right-panel">
          <div className="chart-card chart-h">
            <h2 className="section-title">📈 발전량 vs AI 예측 (최근 1시간)</h2>
            <div className="chart-box">
              <Line key="power" data={powerChart} options={chartOptions} />
            </div>
          </div>

          <div className="chart-card chart-h">
            <h2 className="section-title">📉 AI 예측 오차 강조 (최근 1시간)</h2>
            <div className="chart-box">
              <Line key="error" data={errorChart} options={chartOptions} />
            </div>
          </div>

          <div className="chart-card chart-h">
            <h2 className="section-title">🔋 SOC 변화 (최근 1시간)</h2>
            <div className="chart-box">
              <Line key="soc" data={socChart} options={chartOptions} />
            </div>
          </div>

          <div className="chart-card chart-h">
            <h2 className="section-title">🌱 토양습도 모니터링 (최근 1시간)</h2>
            <div className="chart-box">
              <Line key="soil" data={soilChart} options={chartOptions} />
            </div>
          </div>

          <div className="chart-card chart-h">
            <h2 className="section-title">🌤 온도 · 습도 변화 (최근 1시간)</h2>
            <div className="chart-box">
              <Line key="env" data={envChart} options={chartOptions} />
            </div>
          </div>
        </div>
      </div>

      <div className="bottom-panel">
        <div className="history-header">
          <h2 className="section-title">🧾 최근 모드 변화 히스토리 (최신 10개)</h2>
          <button
            type="button"
            className="history-toggle-all"
            onClick={() => {
              const shouldOpenAll = modeHistory.some((row) => !expandedHistory[row.timestamp]);
              const nextState = {};
              modeHistory.forEach((row) => {
                nextState[row.timestamp] = shouldOpenAll;
              });
              setExpandedHistory(nextState);
            }}
          >
            {modeHistory.some((row) => !expandedHistory[row.timestamp]) ? "전체 펼치기" : "전체 접기"}
          </button>
        </div>

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
                const isOpen = !!expandedHistory[row.timestamp];

                return (
                  <li
                    className={`history-item ${m.rowCls} ${isNew ? "history-pop" : ""}`}
                    key={row.timestamp}
                  >
                    <button
                      type="button"
                      className="history-card-btn"
                      onClick={() => toggleHistory(row.timestamp)}
                      aria-expanded={isOpen}
                    >
                      <div className="history-top">
                        <span className="history-time">
                          {row.timestamp}
                          {isNew && <span className="new-badge">NEW ✨</span>}
                        </span>

                        <span className="history-expand-icon">{isOpen ? "▲" : "▼"}</span>
                      </div>

                      <div className="history-mode-card">
                        {m.emoji} {m.text}
                      </div>

                      <div className="history-summary">
                        SOC {formatNumber(row.soc, 0)}% · 발전 {formatNumber(row.solar_power)}W · 예측{" "}
                        {formatNumber(row.pred_1h)}W
                      </div>
                    </button>

                    {isOpen && (
                      <div className="history-detail">
                        <div className="history-detail-row">
                          <span className="history-chip">🌱 토양 {formatNumber(row.soil, 0)}%</span>
                          <span className="history-chip">🌤 온도 {formatNumber(row.temperature, 1)}°C</span>
                          <span className="history-chip">💧 습도 {formatNumber(row.humidity, 1)}%</span>
                        </div>

                        <div className="history-detail-row">
                          <span className="history-chip">💡 조도 {formatNumber(row.light, 0)}%</span>
                          <span className="history-chip">🔆 태양광 {formatNumber(row.solar_voltage)}V</span>
                          <span className="history-chip">🔌 전류 {formatNumber(row.solar_current, 3)}A</span>
                        </div>

                        <div className="history-detail-row">
                          <span className="history-chip">🔋 배터리 {formatNumber(row.battery_voltage)}V</span>
                          <span className="history-chip">🔌 배터리전류 {formatNumber(row.battery_current, 3)}A</span>
                          <span className="history-chip">💧 펌프 {Number(row.pump) === 1 ? "ON" : "OFF"}</span>
                        </div>

                        <div className="history-detail-row">
                          <span className="history-chip">
                            💡 LED {Number(row.led) === 1 ? `ON (${formatNumber(row.led_brightness, 0)}%)` : "OFF"}
                          </span>
                          <span className={`history-chip ${Number(row.water_alert) === 1 ? "chip-danger" : ""}`}>
                            🚨 급수경고 {Number(row.water_alert) === 1 ? "ALERT" : "정상"}
                          </span>
                        </div>
                      </div>
                    )}
                  </li>
                );
              })}
            </ul>
          )}
        </div>

        <p className="footer-note">10초마다 자동 갱신됩니다 ⏱</p>
      </div>
    </div>
  );
}

export default App;