async function failure(response) {
  let detail = { type: "HTTPError", message: `The studio answered ${response.status}.` };
  try {
    const data = await response.json();
    if (data && data.error) detail = data.error;
  } catch { /* not JSON */ }
  const error = new Error(detail.message);
  error.type = detail.type;
  error.status = response.status;
  return error;
}

async function call(method, path, body) {
  const response = await fetch(path, {
    method,
    credentials: "same-origin",
    headers: body === undefined ? {} : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!response.ok) throw await failure(response);
  return response;
}

export const getState = async () => (await call("GET", "/api/state")).json();
export const setOrbital = async (side, name, enabled) =>
  (await call("PATCH", `/api/orbitals/${encodeURIComponent(side)}/${encodeURIComponent(name)}`, { enabled })).json();
export const resetConversation = async () => (await call("POST", "/api/conversation/reset", {})).json();

/** Run a turn and call onEvent for every event of its stream, in order. */
export async function sendTurn(message, onEvent) {
  const response = await call("POST", "/api/turns", { message });
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value;
    let end;
    while ((end = buffer.indexOf("\n\n")) >= 0) {
      const block = buffer.slice(0, end);
      buffer = buffer.slice(end + 2);
      const data = block.split("\n").find((line) => line.startsWith("data: "));
      if (data) onEvent(JSON.parse(data.slice(6)));
    }
  }
}
