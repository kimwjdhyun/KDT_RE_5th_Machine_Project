import axios from "axios";

const API_BASE_URL = "http://127.0.0.1:5000";

export const fetchEnergy = async () => {
  const response = await axios.get(`${API_BASE_URL}/api/energy`);
  return response.data;
};