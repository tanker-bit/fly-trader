// BACKUP COPY kept in the fly-trader repository (handover task 2). It is not run from here: copy it into the cloned
// investopedia_simulator_api folder (next to its package.json and node_modules with puppeteer) and run it there. Since the lab
// session (2026-11-27) the fly reads its account and places orders through the website itself (broker/site_account.py,
// broker/trade_ui.py, Playwright); this bridge is only needed for the older API route (broker.daily_run --portfolio live).
// The copy contains no secret: the bridge key is random at every start and written to bridge_key.txt (git-ignored).

// The "bridge": keeps ONE real browser window logged in to the Investopedia simulator and lets the Python program send the library's API calls
// THROUGH that window (the site blocks plain Python requests with a 403, but lets your own logged-in browser in).
//   node token_keeper.mjs                 read-only: portfolio, quotes, order PREVIEWS. Order submission and cancelling are refused here.
//   node token_keeper.mjs --allow-orders  also lets orders through (only use when you mean it)
// Put this file in the library folder (next to package.json) and run it from there. First run: log in in the window (email + emailed code);
// the saved profile ./browser_profile keeps the login for about a week. Leave it running while the fly works; Ctrl+C stops it.
// Safety: it listens on 127.0.0.1 only, every request needs the secret in ./bridge_key.txt (new at every start), and only
// https://api.investopedia.com/simulator/ addresses are forwarded. Nothing is typed for you and no password is stored.
// browser_profile and bridge_key.txt are a login: keep them in the library folder, never commit them, never send them to anyone.
import puppeteer from "puppeteer";
import fs from "fs";
import path from "path";
import http from "http";
import crypto from "crypto";

const SIM = "https://www.investopedia.com/simulator";
const TOKEN_URL = "https://www.investopedia.com/simulator/api/auth/token";
const ALLOWED_PREFIX = "https://api.investopedia.com/simulator/";
const PORT = 8765;
const EVERY_MS = 4 * 60 * 1000;
const ALLOW_ORDERS = process.argv.includes("--allow-orders");
const KEY = crypto.randomBytes(16).toString("hex");
let token = null, lastOk = 0, lastMsg = 0;

const browser = await puppeteer.launch({ headless: false, defaultViewport: null, userDataDir: path.resolve("./browser_profile") });
const page = (await browser.pages())[0] || (await browser.newPage());
const stamp = () => new Date().toLocaleTimeString();

function save(data, how) {
  token = data.accessToken;
  lastOk = Date.now();
  console.log(stamp() + "  token ok (" + how + "), good for " + data.expiresIn + " s");
}

page.on("response", async (response) => {
  if (response.url().split("?")[0] !== TOKEN_URL || response.status() !== 200) return;
  try { const d = JSON.parse(await response.text()); if (d.accessToken) save(d, "from the page"); } catch (e) { /* ignore */ }
});

process.on("SIGINT", async () => { console.log("\nstopping"); await browser.close().catch(() => {}); process.exit(0); });

let queue = Promise.resolve();                      // one call at a time through the window
function forward(url, body) {
  const run = async () => {
    if (!token) return { status: 503, text: "no token yet: log in in the browser window" };
    return await page.evaluate(async (u, b, t) => {
      const r = await fetch(u, { method: "POST", headers: { Authorization: "Bearer " + t, "Content-Type": "application/json" }, body: b });
      return { status: r.status, text: await r.text() };
    }, url, body, token);
  };
  const p = queue.then(run, run);
  queue = p.catch(() => {});
  return p;
}

http.createServer((req, res) => {
  const send = (code, obj) => { res.writeHead(200, { "Content-Type": "application/json" }); res.end(JSON.stringify(Object.assign({ status: code }, obj))); };
  if (req.headers["x-bridge-key"] !== KEY) return send(403, { text: "bad bridge key" });
  let raw = "";
  req.on("data", (c) => { raw += c; });
  req.on("end", async () => {
    try {
      const { url, body } = JSON.parse(raw);
      if (typeof url !== "string" || !url.startsWith(ALLOWED_PREFIX)) return send(403, { text: "address not allowed" });
      const b = typeof body === "string" ? body : JSON.stringify(body || {});
      if (!ALLOW_ORDERS && /submit(Stock|Option|Cancel)?Trade|submitCancel/i.test(b)) return send(403, { text: "orders are blocked: the bridge was started without --allow-orders" });
      const r = await forward(url, b);
      send(r.status, { text: r.text });
    } catch (e) { send(502, { text: "bridge error: " + String(e.message).slice(0, 200) }); }
  });
}).listen(PORT, "127.0.0.1", () => {
  fs.writeFileSync("./bridge_key.txt", KEY);
  console.log("bridge listening on 127.0.0.1:" + PORT + (ALLOW_ORDERS ? "   ORDERS ALLOWED" : "   read-only (orders blocked)"));
});

await page.goto(SIM, { waitUntil: "load", timeout: 60000 });
console.log("Browser open. If it asks you to log in: click LOG IN, enter your email and the emailed code. Leave this window running.");

while (true) {
  await new Promise((r) => setTimeout(r, 5000));
  if (Date.now() - lastOk < EVERY_MS) continue;
  try {
    const res = await page.evaluate(async (u) => {
      const r = await fetch(u, { method: "POST", credentials: "include" });
      let j = null; try { j = await r.json(); } catch (e) { /* not json */ }
      return { status: r.status, j };
    }, TOKEN_URL);
    if (res.status === 200 && res.j && res.j.accessToken) save(res.j, "refreshed");
    else if (Date.now() - lastMsg > 30000) { lastMsg = Date.now(); console.log(stamp() + "  not logged in yet (answer " + res.status + "): log in in the browser window"); }
  } catch (e) {
    if (Date.now() - lastMsg > 30000) { lastMsg = Date.now(); console.log(stamp() + "  waiting (" + String(e.message).slice(0, 80) + ")"); }
  }
}
