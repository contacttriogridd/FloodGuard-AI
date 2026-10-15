/* FloodGuard AI — Streamlit bridge, injected at the top of api/static/index.html by streamlit_app/app.py.
 *
 * The page runs inside a Streamlit component iframe, where there is no HTTP API. Its fetch() calls to the
 * API paths are queued and sent one at a time to Python as the component value; Python answers them with
 * the FastAPI app (in-process) and returns the response in the component args. Everything else in the
 * page is unchanged. Protocol: Streamlit custom components v1 (postMessage).
 */
(() => {
  const API_PATHS = ["/health", "/model-info", "/predict", "/predict/batch", "/predict/csv", "/explain"];
  const API_BASE = __API_BASE__;  // public FastAPI deployment, used for the /ui and /docs links and API snippets
  const TIMEOUT_MS = 120000;
  window.FG_API_BASE = API_BASE;

  const post = (type, data = {}) => window.parent.postMessage({ isStreamlitMessage: true, type, ...data }, "*");
  const realFetch = window.fetch.bind(window);
  const queue = [];
  let current = null, ready = false, seq = 0;

  const toB64 = async (blob) => {
    const bytes = new Uint8Array(await blob.arrayBuffer());
    let s = "";
    for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
    return btoa(s);
  };

  async function encodeBody(body) {
    if (body instanceof FormData) {
      const files = [];
      for (const [name, v] of body.entries()) {
        if (v instanceof Blob) files.push({ name, filename: v.name || "upload.csv", type: v.type || "text/csv", b64: await toB64(v) });
      }
      return { files };
    }
    return { body: body == null ? null : String(body) };
  }

  function settle(job, fn) {
    clearTimeout(job.timer);
    if (current === job) current = null;
    if (!job.done) { job.done = true; fn(); }
    pump();
  }

  function pump() {
    if (current || !ready) return;
    while (queue.length && queue[0].done) queue.shift();
    const job = queue.shift();
    if (!job) return;
    current = job;
    job.timer = setTimeout(() => settle(job, () => job.reject(new Error("The model server did not answer in time."))), TIMEOUT_MS);
    post("streamlit:setComponentValue", { value: job.msg, dataType: "json" });
  }

  window.fetch = async (input, init = {}) => {
    const url = new URL(typeof input === "string" ? input : input.url, location.href);
    if (!API_PATHS.includes(url.pathname)) return realFetch(input, init);
    const signal = init.signal;
    if (signal && signal.aborted) throw new DOMException("Aborted", "AbortError");
    const msg = { id: `${Date.now().toString(36)}-${++seq}`, method: (init.method || "GET").toUpperCase(), path: url.pathname,
                  ...(await encodeBody(init.body)) };
    return new Promise((resolve, reject) => {
      const job = { msg, resolve, reject, done: false };
      if (signal) signal.addEventListener("abort", () => settle(job, () => reject(new DOMException("Aborted", "AbortError"))));
      queue.push(job);
      pump();
    });
  };

  window.addEventListener("message", (e) => {
    const d = e.data;
    if (!d || d.type !== "streamlit:render") return;
    ready = true;
    const r = d.args && d.args.response;
    if (r && current && r.id === current.msg.id) {
      const job = current;
      settle(job, () => job.resolve(new Response(r.body, { status: r.status, headers: { "Content-Type": r.content_type } })));
    }
    pump();
  });

  // Fill the Streamlit viewport; the page scrolls inside the frame so its sticky nav, toasts and anchors work.
  const fit = () => {
    let height = 900;
    try { height = window.parent.innerHeight; } catch (_) { /* parent not reachable */ }
    post("streamlit:setFrameHeight", { height });
  };
  try { window.parent.addEventListener("resize", fit); } catch (_) { /* parent not reachable */ }
  window.addEventListener("load", fit);

  // The Gradio UI and the OpenAPI docs live on the FastAPI deployment, not in Streamlit.
  document.addEventListener("DOMContentLoaded", () => {
    document.querySelectorAll('a[href="/ui"], a[href="/docs"]').forEach((a) => {
      a.href = API_BASE + a.getAttribute("href");
      a.target = "_blank";
      a.rel = "noreferrer";
    });
  });

  post("streamlit:componentReady", { apiVersion: 1 });
  fit();
})();
