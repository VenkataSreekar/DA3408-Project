// Frontend configuration. The page talks to the backend ONLY through API_URL.
// Edit these URLs to point at your running services (no rebuild of the page needed).
window.APP_CONFIG = {
  API_URL: "http://localhost:8000",
  MLFLOW_URL: "http://localhost:5000",
  GRAFANA_URL: "http://localhost:3001",
  PROMETHEUS_URL: "http://localhost:9090",
};
