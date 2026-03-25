async function fetchJSON(url) {
  const res = await fetch(url, {
    headers: {
      "Content-Type": "application/json",
    },
  });

  if (!res.ok) {
    throw new Error(`API 요청 실패: ${res.status}`);
  }

  return res.json();
}

export async function fetchDataLog() {
  return fetchJSON("/api/history");
}

export async function fetchStats() {
  return fetchJSON("/api/stats");
}

export async function fetchLatest() {
  return fetchJSON("/api/latest");
}

export async function fetchHealth() {
  return fetchJSON("/api/health");
}

export async function fetchPredict() {
  return fetchJSON("/api/predict");
}