async function fetchJson(url, options) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || "Request failed");
  }
  return data;
}

function setStat(id, value) {
  const node = document.getElementById(id);
  if (node) node.textContent = value;
}

function renderLogs(rows) {
  const body = document.getElementById("logsBody");
  body.innerHTML = "";
  rows.forEach((row) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${row.created_at || ""}</td>
      <td>${row.direction || ""}</td>
      <td>${row.risk_level || ""}</td>
      <td>${row.risk_category || ""}</td>
      <td>${row.final_action || ""}</td>
      <td>${row.score ?? ""}</td>
      <td>${row.original_text_preview || ""}</td>
    `;
    body.appendChild(tr);
  });
}

function renderRules(sectionId, rows) {
  const body = document.getElementById(sectionId);
  body.innerHTML = "";
  rows.forEach((row) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${row.rule_id || row.example_id || ""}</td>
      <td>${row.category || ""}</td>
      <td>${row.subcategory || ""}</td>
      <td>${row.level || ""}</td>
      <td>${row.action || ""}</td>
    `;
    body.appendChild(tr);
  });
}

async function loadDashboard() {
  const stats = await fetchJson("/api/stats");
  const totalLogs = Object.values(stats.level_distribution || {}).reduce((sum, count) => sum + count, 0);
  setStat("totalLogs", totalLogs);
  setStat("highRiskCount", stats.level_distribution?.high || 0);
  setStat("blockedCount", stats.action_distribution?.block || 0);
  setStat("replacedCount", stats.action_distribution?.replace || 0);
}

async function loadLogs() {
  const direction = document.getElementById("directionFilter").value;
  const riskLevel = document.getElementById("riskLevelFilter").value;
  const query = new URLSearchParams({
    page: "1",
    per_page: "50",
    direction,
    risk_level: riskLevel,
  });
  const data = await fetchJson(`/api/logs?${query.toString()}`);
  renderLogs(data.data || []);
}

async function loadRules() {
  const data = await fetchJson("/api/rules?type=all");
  renderRules("inputRulesBody", data.input || []);
}

async function clearAllLogs() {
  if (!confirm("确定清空所有聊天记录和风险日志吗？此操作不可恢复。")) return;
  await fetchJson("/api/logs", { method: "DELETE" });
  await loadDashboard();
  await loadLogs();
}

document.addEventListener("DOMContentLoaded", async () => {
  document.getElementById("refreshStatsBtn").addEventListener("click", loadDashboard);
  document.getElementById("clearLogsBtn").addEventListener("click", clearAllLogs);
  document.getElementById("refreshLogsBtn").addEventListener("click", loadLogs);
  document.getElementById("refreshRulesBtn").addEventListener("click", loadRules);
  document.getElementById("directionFilter").addEventListener("change", loadLogs);
  document.getElementById("riskLevelFilter").addEventListener("change", loadLogs);

  await loadDashboard();
  await loadLogs();
  await loadRules();
});
