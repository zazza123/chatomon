async function failure(response) {
  let detail = { type: "HTTPError", message: `The studio answered ${response.status}.` };
  try {
    const data = await response.json();
    if (data && data.error) detail = data.error;
  } catch { /* not JSON */ }
  const error = new Error(detail.message);
  error.type = detail.type;
  error.status = response.status;
  error.setting = detail.setting;
  error.problems = detail.problems;
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

const send = async (method, path, body) => (await call(method, path, body)).json();

export const getState = () => send("GET", "/api/state");
export const resetConversation = () => send("POST", "/api/conversation/reset", {});
export const patchOrbital = (side, name, change) => send("PATCH", "/api/orbitals", { side, name, ...change });
export const addOrbital = (side, kind, settings) => send("POST", "/api/orbitals", { side, kind, settings });
export const deleteOrbital = (side, name) => send("DELETE", "/api/orbitals", { side, name });
export const resetOrbitals = (target = {}) => send("POST", "/api/orbitals/reset", target);
export const undoOrbitals = () => send("POST", "/api/orbitals/undo", {});
export const getCode = () => send("GET", "/api/code");
export const getPresetText = async () => (await call("GET", "/api/preset")).text();
export const importPreset = (text) => send("POST", "/api/preset", { text });

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
