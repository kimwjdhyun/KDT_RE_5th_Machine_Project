import axios from "axios";

const API_BASE_URL = "http://127.0.0.1:5000";

export const fetchDataLog = async () => {
  const res = await axios.get(`${API_BASE_URL}/data`);
  return res.data; // 최근 50개 배열
};

export const fetchStats = async () => {
  const res = await axios.get(`${API_BASE_URL}/stats`);
  return res.data; // { total_generation, carbon_reduction_g }
};