import axios from "axios";

const api = axios.create({ baseURL: "/api" });

export const errorInfo = (e) => ({ status: e?.response?.status ?? 0, message: e?.response?.data?.detail || e?.message || "request failed" });
export const getHealth = () => api.get("/health").then((r) => r.data);
export const getRecordings = () => api.get("/recordings").then((r) => r.data.recordings);
export const getShowcase = () => api.get("/showcase").then((r) => r.data);
export const diagnoseRecording = (rec_id) => api.post("/diagnose", { rec_id }).then((r) => r.data);
export const diagnoseUpload = (file) => {
  const form = new FormData();
  form.append("file", file);
  return api.post("/diagnose", form).then((r) => r.data);
};
export const getCase = (id) => api.get(`/cases/${id}`).then((r) => r.data);
export const getActivity = (id) => api.get(`/cases/${id}/activity`).then((r) => r.data);
export const getReport = (id) => api.get(`/cases/${id}/report`).then((r) => r.data);
export const getChannels = (id) => api.get(`/cases/${id}/channels`).then((r) => r.data);
export const getSignals = (id) => api.get(`/cases/${id}/signals`).then((r) => r.data);
