const API_BASE = process.env.REACT_APP_API_BASE || "http://192.168.201.138:5000";

async function fetchJSON(url) {
  const res = await fetch(url);

  if (!res.ok) {
    throw new Error(`API 요청 실패: ${res.status}`);
  }

  return res.json();
}

export async function fetchDataLog() {
  return fetchJSON(`${API_BASE}/api/history`);
}

export async function fetchStats() {
  return fetchJSON(`${API_BASE}/api/stats`);
}

export async function fetchLatest() {
  return fetchJSON(`${API_BASE}/api/latest`);
}

export async function fetchHealth() {
  return fetchJSON(`${API_BASE}/api/health`);
}