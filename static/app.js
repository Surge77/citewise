const form = document.getElementById("ask-form");
const input = document.getElementById("question");
const sendButton = document.getElementById("send");
const chat = document.getElementById("chat");

// Everything is rendered with textContent, never innerHTML, so model output can't inject markup.
function addMessage(role, text) {
  const message = document.createElement("div");
  message.className = `message ${role}`;
  message.textContent = text;
  chat.appendChild(message);
  chat.scrollTop = chat.scrollHeight;
  return message;
}

function addCitations(message, citations) {
  if (citations.length === 0) return;
  const list = document.createElement("ol");
  list.className = "citations";
  for (const c of citations) {
    const item = document.createElement("li");
    item.value = c.number;
    const source = document.createElement("strong");
    source.textContent = `${c.doc_name}, p. ${c.page}`;
    const snippet = document.createElement("span");
    snippet.textContent = ` — ${c.snippet}`;
    item.append(source, snippet);
    list.appendChild(item);
  }
  message.appendChild(list);
}

// Reads the NDJSON stream: "delta" lines carry text as it is generated, then one "done" or "error" line ends it.
async function streamAnswer(question, history, onDelta) {
  const response = await fetch("/api/ask/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, history }),
  });
  if (!response.ok || !response.body) {
    const body = await response.json().catch(() => null);
    throw new Error(body?.error?.message ?? `Request failed (${response.status}).`);
  }
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffered = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffered += value;
    const lines = buffered.split("\n");
    buffered = lines.pop();  // A network chunk can end mid-line; keep the partial line for the next read.
    for (const line of lines) {
      if (!line) continue;
      const event = JSON.parse(line);
      if (event.type === "delta") onDelta(event.text);
      else if (event.type === "done") return event;
      else if (event.type === "error") throw new Error(event.message);
    }
  }
  throw new Error("The answer stream ended unexpectedly.");
}

const MAX_UPLOAD_BYTES = 20 * 1024 * 1024;
// Recent turns sent with each question so follow-ups ("explain that more simply") have context; the API accepts 8.
const MAX_HISTORY_TURNS = 6;
const history = [];
const uploadInput = document.getElementById("upload");
const uploadStatus = document.getElementById("upload-status");
const documentList = document.getElementById("documents");

function setUploadStatus(text, kind = "") {
  uploadStatus.className = `upload-status ${kind}`;
  uploadStatus.textContent = text;
}

async function loadDocuments() {
  const response = await fetch("/api/documents");
  const body = await response.json().catch(() => null);
  documentList.replaceChildren();
  if (!body?.success || body.data.length === 0) {
    const empty = document.createElement("li");
    empty.className = "empty";
    empty.textContent = "No documents yet. Add a PDF to start asking questions.";
    documentList.appendChild(empty);
    return;
  }
  for (const doc of body.data) {
    const item = document.createElement("li");
    item.textContent = `${doc.name} · ${doc.chunks} passages`;
    documentList.appendChild(item);
  }
}

function describeUpload(data) {
  if (data.already_indexed) return `${data.name} is already in your library.`;
  const pages = `${data.pages} page${data.pages === 1 ? "" : "s"}`;
  const note = data.empty_pages.length
    ? ` No text found on page(s) ${data.empty_pages.join(", ")} — they may be scanned images.`
    : "";
  return `Added ${data.name}: ${pages}, ${data.chunks_added} passages.${note}`;
}

uploadInput.addEventListener("change", async () => {
  const file = uploadInput.files?.[0];
  uploadInput.value = "";
  if (!file) return;
  if (file.size > MAX_UPLOAD_BYTES) {
    setUploadStatus("PDFs can be at most 20 MB.", "error");
    return;
  }
  setUploadStatus(`Reading ${file.name}…`, "pending");
  try {
    const response = await fetch(`/api/documents?name=${encodeURIComponent(file.name)}`, {
      method: "POST",
      headers: { "Content-Type": "application/pdf" },
      body: file,
    });
    const body = await response.json().catch(() => null);
    if (!body?.success) {
      throw new Error(body?.error?.message ?? `Upload failed (${response.status}).`);
    }
    setUploadStatus(describeUpload(body.data), body.data.empty_pages.length ? "warning" : "ok");
    await loadDocuments();
  } catch (error) {
    setUploadStatus(error.message, "error");
  }
});

loadDocuments().catch(() => setUploadStatus("Could not load your documents.", "error"));

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = input.value.trim();
  if (!question) return;

  addMessage("user", question);
  input.value = "";
  sendButton.disabled = true;
  const pending = addMessage("bot pending", "Searching your documents…");

  try {
    let streamed = "";
    const data = await streamAnswer(question, history.slice(-MAX_HISTORY_TURNS), (text) => {
      if (!streamed) pending.className = "message bot";
      streamed += text;
      pending.textContent = streamed;
      chat.scrollTop = chat.scrollHeight;
    });
    pending.className = data.found ? "message bot" : "message bot not-found";
    pending.textContent = data.answer;
    addCitations(pending, data.citations);
    chat.scrollTop = chat.scrollHeight;
    history.push({ role: "user", text: question }, { role: "assistant", text: data.answer });
  } catch (error) {
    pending.className = "message bot error";
    pending.textContent = error.message;
  } finally {
    sendButton.disabled = false;
    input.focus();
  }
});
