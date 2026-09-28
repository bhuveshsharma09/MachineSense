import axios from "axios";

export { errorInfo } from "./lib/session.js";

const api = axios.create({ baseURL: "/api" });

// Every diagnose / rerun request carries a request id (header) and its input id; the server echoes both in the
// response headers, and the session drops a response whose ids do not match the current selection (App-2a.3).
const echoed = (r) => ({ record: r.data, requestId: r.headers["x-request-id"], inputId: r.headers["x-input-id"] });

export const getHealth = () => api.get("/health").then((r) => r.data);
export const getRecordings = () => api.get("/recordings").then((r) => r.data.recordings);
export const getShowcase = () => api.get("/showcase").then((r) => r.data);
export const diagnoseRecording = (rec_id, { signal, requestId } = {}) =>
  api.post("/diagnose", { rec_id, request_id: requestId }, { signal, headers: { "X-Request-Id": requestId } }).then(echoed);
export const diagnoseUpload = (file, { signal, requestId, uploadId } = {}) => {
  const form = new FormData();
  form.append("file", file);
  form.append("upload_id", uploadId);
  form.append("request_id", requestId);
  return api.post("/diagnose", form, { signal, headers: { "X-Request-Id": requestId } }).then(echoed);
};
export const rerunCase = (caseId, { signal, requestId } = {}) =>
  api.post(`/cases/${caseId}/rerun`, null, { signal, headers: { "X-Request-Id": requestId } }).then(echoed);
export const getCase = (id, { signal } = {}) => api.get(`/cases/${id}`, { signal }).then((r) => r.data);
export const getCaseMeta = (id, { signal } = {}) => api.get(`/cases/${id}/meta`, { signal }).then((r) => r.data);
export const getActivity = (id, { signal } = {}) => api.get(`/cases/${id}/activity`, { signal }).then((r) => r.data);
export const getReport = (id, { signal } = {}) => api.get(`/cases/${id}/report`, { signal }).then((r) => r.data);
export const getChannels = (id, { signal } = {}) => api.get(`/cases/${id}/channels`, { signal }).then((r) => r.data);
export const getSignals = (id, { signal } = {}) => api.get(`/cases/${id}/signals`, { signal }).then((r) => r.data);
