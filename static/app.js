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
