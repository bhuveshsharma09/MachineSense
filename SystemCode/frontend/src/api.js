import axios from "axios";

const api = axios.create({ baseURL: "/api" });

export const errorText = (e) => e?.response?.data?.detail || e?.message || "request failed";
export const getHealth = () => api.get("/health").then((r) => r.data);
export const getRecordings = () => api.get("/recordings").then((r) => r.data.recordings);
export const diagnoseRecording = (rec_id) => api.post("/diagnose", { rec_id }).then((r) => r.data);
export const diagnoseUpload = (file, nominalSpeedHz) => {
  const form = new FormData();
  form.append("file", file);
  if (nominalSpeedHz) form.append("nominal_speed_hz", nominalSpeedHz);
  return api.post("/diagnose", form).then((r) => r.data);
};
export const getCase = (id) => api.get(`/cases/${id}`).then((r) => r.data);
export const getReport = (id) => api.get(`/cases/${id}/report`).then((r) => r.data);
