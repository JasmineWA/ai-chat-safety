const SESSION_STORAGE_KEY = "safety_chat_session_id";
const MOBILE_WIDTH = 1100;
const TRASH_ICON = `
  <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
    <path d="M9 3h6l1 2h4v2H4V5h4l1-2Zm1 6h2v8h-2V9Zm4 0h2v8h-2V9ZM7 9h2v8H7V9Zm-1 12-1-14h14l-1 14H6Z"></path>
  </svg>
`;

function escapeHtml(value) {
  return String(value || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function riskClass(level) {
  if (!level) return "safe";
  if (level === "high") return "high";
  if (level === "medium") return "medium";
  if (level === "low") return "low";
  return "safe";
}

function getSessionId() {
  return localStorage.getItem(SESSION_STORAGE_KEY);
}

function setSessionId(sessionId) {
  localStorage.setItem(SESSION_STORAGE_KEY, sessionId);
}

function clearSessionId() {
  localStorage.removeItem(SESSION_STORAGE_KEY);
}

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || "Request failed");
  }
  return data;
}

function clearMessages() {
  document.getElementById("messages").innerHTML = "";
}

function appendMeta(wrap, rows) {
  const validRows = rows.filter(Boolean);
  if (!validRows.length) return;

  const meta = document.createElement("div");
  meta.className = "message-meta";

  validRows.forEach((row) => {
    const line = document.createElement("div");
    line.className = `meta-line ${row.multiline ? "multiline" : ""}`.trim();

    const label = document.createElement("span");
    label.textContent = row.label;

    const value = document.createElement("strong");
    value.textContent = row.value;

    line.appendChild(label);
    line.appendChild(value);
    meta.appendChild(line);
  });

  wrap.appendChild(meta);
}

function scrollMessagesToBottom() {
  const messages = document.getElementById("messages");
  messages.scrollTop = messages.scrollHeight;
}

function updateScrollBottomButton() {
  const messages = document.getElementById("messages");
  const button = document.getElementById("scrollBottomBtn");
  const distance = messages.scrollHeight - messages.scrollTop - messages.clientHeight;
  button.classList.toggle("visible", distance > 120);
}

function buildDeleteMessageButton(messageId) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "message-delete-btn";
  button.innerHTML = TRASH_ICON;
  button.title = "删除消息";
  button.setAttribute("aria-label", "删除消息");
  button.addEventListener("click", async () => {
    if (!confirm("确定删除这条消息吗？")) return;
    try {
      const result = await fetchJson(`/api/messages/${messageId}`, { method: "DELETE" });
      if (result.session_id) {
        await loadConversation(result.session_id);
        await loadSessions();
      }
    } catch (err) {
      document.getElementById("errorText").textContent = `删除失败: ${err.message}`;
    }
  });
  return button;
}

function addMessage(role, text, options = {}) {
  const messages = document.getElementById("messages");
  const row = document.createElement("div");
  row.className = `msg ${role}`;
  if (options.messageId) {
    row.dataset.messageId = String(options.messageId);
  }

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = role === "user" ? "ME" : "AI";

  const wrap = document.createElement("div");
  wrap.className = "bubble-wrap";

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text || "";
  wrap.appendChild(bubble);

  appendMeta(wrap, options.meta || []);

  if (options.riskLevel) {
    const tag = document.createElement("div");
    tag.className = `risk-tag ${riskClass(options.riskLevel)}`;
    tag.textContent = `风险等级: ${options.riskLevel}`;
    wrap.appendChild(tag);
  }

  if (options.messageId) {
    wrap.appendChild(buildDeleteMessageButton(options.messageId));
  }

  row.appendChild(avatar);
  row.appendChild(wrap);
  messages.appendChild(row);
  scrollMessagesToBottom();
  updateScrollBottomButton();
  return { row, wrap, bubble };
}

function fillMessageMeta(targetWrap, options = {}) {
  if (!targetWrap) return;

  targetWrap.querySelectorAll(".message-meta, .risk-tag").forEach((node) => node.remove());
  appendMeta(targetWrap, options.meta || []);

  if (options.riskLevel) {
    const tag = document.createElement("div");
    tag.className = `risk-tag ${riskClass(options.riskLevel)}`;
    tag.textContent = `风险等级: ${options.riskLevel}`;
    targetWrap.appendChild(tag);
  }
}

function renderWelcome() {
  addMessage("assistant", "这里是新的安全对话入口。你发送的内容会先经过输入检测，再发送给大模型；模型返回后还会经过输出检测。");
}

function closeSidebarOnMobile() {
  if (window.innerWidth >= MOBILE_WIDTH) return;
  document.querySelector(".sidebar")?.classList.remove("open");
  document.getElementById("sidebarBackdrop")?.classList.remove("visible");
}

function openSidebarOnMobile() {
  document.querySelector(".sidebar")?.classList.add("open");
  document.getElementById("sidebarBackdrop")?.classList.add("visible");
}

function renderSessionList(items, activeSessionId) {
  const container = document.getElementById("sessionList");
  container.innerHTML = "";

  items.forEach((item) => {
    const wrapper = document.createElement("div");
    wrapper.className = "session-item-row";

    const button = document.createElement("button");
    button.type = "button";
    button.className = `session-item ${item.session_id === activeSessionId ? "active" : ""}`;
    button.innerHTML = `<strong>${escapeHtml(item.title || "新对话")}</strong>`;
    button.addEventListener("click", async () => {
      setSessionId(item.session_id);
      await loadConversation(item.session_id);
      await loadSessions();
      closeSidebarOnMobile();
    });

    const deleteButton = document.createElement("button");
    deleteButton.type = "button";
    deleteButton.className = "session-delete-btn";
    deleteButton.innerHTML = TRASH_ICON;
    deleteButton.title = "删除整个会话";
    deleteButton.setAttribute("aria-label", "删除整个会话");
    deleteButton.addEventListener("click", async (event) => {
      event.stopPropagation();
      if (!confirm("确定删除整个会话吗？该会话下的所有消息和日志都会删除。")) return;
      try {
        await fetchJson(`/api/sessions/${encodeURIComponent(item.session_id)}`, { method: "DELETE" });
        if (getSessionId() === item.session_id) {
          clearSessionId();
          const newSession = await fetchJson("/api/sessions/new", { method: "POST" });
          setSessionId(newSession.session_id);
          clearMessages();
          renderWelcome();
        }
        await loadSessions();
      } catch (err) {
        document.getElementById("errorText").textContent = `删除会话失败: ${err.message}`;
      }
    });

    wrapper.appendChild(button);
    wrapper.appendChild(deleteButton);
    container.appendChild(wrapper);
  });
}

async function loadSessions() {
  const keyword = document.getElementById("sessionSearchInput")?.value?.trim() || "";
  const query = new URLSearchParams({ keyword });
  const data = await fetchJson(`/api/sessions?${query.toString()}`);
  renderSessionList(data.data || [], getSessionId());
}

async function ensureActiveSession() {
  let sessionId = getSessionId();
  if (!sessionId) {
    const data = await fetchJson("/api/sessions/new", { method: "POST" });
    sessionId = data.session_id;
    setSessionId(sessionId);
  }
  return sessionId;
}

async function loadConversation(sessionId) {
  clearMessages();
  const data = await fetchJson(`/api/sessions/${encodeURIComponent(sessionId)}`);
  const messages = data.messages || [];
  if (!messages.length) {
    renderWelcome();
    return;
  }

  messages.forEach((message) => {
    const meta = [];
    if (message.role === "user") {
      if (message.risk_category) meta.push({ label: "输入风险类别", value: message.risk_category });
      if (message.action) meta.push({ label: "输入处理动作", value: message.action });
      if (["replace", "mask"].includes(message.action) && message.safe_text) {
        meta.push({ label: "替换后文本", value: message.safe_text, multiline: true });
      }
    }

    addMessage(message.role, message.text, {
      messageId: message.id,
      riskLevel: message.risk_level,
      meta,
    });
  });
}

async function createNewConversation() {
  const data = await fetchJson("/api/sessions/new", { method: "POST" });
  setSessionId(data.session_id);
  clearMessages();
  renderWelcome();
  await loadSessions();
  closeSidebarOnMobile();
}

async function sendMessage() {
  const input = document.getElementById("messageInput");
  const button = document.getElementById("sendBtn");
  const error = document.getElementById("errorText");
  const value = input.value.trim();
  if (!value || button.disabled) return;

  error.textContent = "";
  button.disabled = true;
  input.value = "";
  const pendingUserMessage = addMessage("user", value);

  try {
    const sessionId = await ensureActiveSession();
    const data = await fetchJson("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: value,
        session_id: sessionId,
      }),
    });

    setSessionId(data.session_id);

    const userMeta = [];
    if (data.input?.risk_category) userMeta.push({ label: "输入风险类别", value: data.input.risk_category });
    if (data.input?.action) userMeta.push({ label: "输入处理动作", value: data.input.action });
    if (data.input && ["replace", "mask"].includes(data.input.action) && data.input.safe_text) {
      userMeta.push({ label: "替换后文本", value: data.input.safe_text, multiline: true });
    }

    fillMessageMeta(pendingUserMessage.wrap, {
      riskLevel: data.input?.risk_level || null,
      meta: userMeta,
    });

    addMessage("assistant", data.reply || "", {
      riskLevel: data.output?.risk_level || null,
      meta: [],
    });

    await loadSessions();
  } catch (err) {
    error.textContent = `请求失败: ${err.message}`;
  } finally {
    button.disabled = false;
    input.focus();
  }
}

document.addEventListener("DOMContentLoaded", async () => {
  const input = document.getElementById("messageInput");
  const button = document.getElementById("sendBtn");
  const newChatButton = document.getElementById("newChatBtn");
  const searchInput = document.getElementById("sessionSearchInput");
  const messages = document.getElementById("messages");
  const scrollBottomBtn = document.getElementById("scrollBottomBtn");
  const sidebarToggle = document.getElementById("sidebarToggle");
  const sidebarBackdrop = document.getElementById("sidebarBackdrop");

  button.addEventListener("click", sendMessage);
  newChatButton.addEventListener("click", createNewConversation);
  searchInput.addEventListener("input", loadSessions);
  messages.addEventListener("scroll", updateScrollBottomButton);
  scrollBottomBtn.addEventListener("click", scrollMessagesToBottom);
  sidebarToggle.addEventListener("click", openSidebarOnMobile);
  sidebarBackdrop.addEventListener("click", closeSidebarOnMobile);

  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    }
  });

  const sessionId = await ensureActiveSession();
  await loadSessions();
  await loadConversation(sessionId);
  updateScrollBottomButton();
});
