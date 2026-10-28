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

async function ask(question) {
  const response = await fetch("/api/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  const body = await response.json().catch(() => null);
  if (!body?.success) {
    throw new Error(body?.error?.message ?? `Request failed (${response.status}).`);
  }
  return body.data;
}

const MAX_UPLOAD_BYTES = 20 * 1024 * 1024;
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
    const data = await ask(question);
    pending.className = data.found ? "message bot" : "message bot not-found";
    pending.textContent = data.answer;
    addCitations(pending, data.citations);
  } catch (error) {
    pending.className = "message bot error";
    pending.textContent = error.message;
  } finally {
    sendButton.disabled = false;
    input.focus();
  }
});
