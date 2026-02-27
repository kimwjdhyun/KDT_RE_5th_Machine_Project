import { useState, useEffect } from "react"
import axios from "axios"
import "./Dashboard.css"

function Dashboard() {
  const [data, setData] = useState({
    power: 0,
    soc: 0,
    temp: 0,
    soil: 0,
  })

  const fetchData = async () => {
    const res = await axios.get("http://localhost:5000/predict")
    setData({
      power: res.data.pred_1h,
      soc: 58,
      temp: 24,
      soil: 42,
    })
  }

  useEffect(() => {
    fetchData()
  }, [])

  return (
    <div className="container">
      <h1>🌱 AI 스마트팜 대시보드</h1>
      <p className="status">🟢 실시간 연결 중</p>

      <div className="grid">
        <div className="card">
          <h2>⚡ 발전량</h2>
          <p>{data.power} W</p>
        </div>

        <div className="card">
          <h2>🔋 SOC</h2>
          <p>{data.soc} %</p>
        </div>

        <div className="card">
          <h2>🌡 온도</h2>
          <p>{data.temp} ℃</p>
        </div>

        <div className="card">
          <h2>💧 토양습도</h2>
          <p>{data.soil} %</p>
        </div>
      </div>
    </div>
  )
}

export default Dashboard