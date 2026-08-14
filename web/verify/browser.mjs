// Real-browser verification of the wasm playground. Headless Chrome is the only way
// to actually exercise the mt chdb-wasm bundle under cross-origin isolation — jsdom/node
// --test cannot: SharedArrayBuffer + worker threads + COOP/COEP behave differently there.
// This directory is deliberately NOT under test/, so `node --test` skips it; run it
// explicitly via `npm run verify:browser`.
//
// Expects `npm --prefix web run serve` already running on http://localhost:8099 — this
// script does not start its own server.
import puppeteer from "puppeteer-core";

const CHROME_PATH = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const URL = "http://localhost:8099/";
const TIMEOUT = 120_000;

async function waitForText(page, selector, needle, timeout = TIMEOUT) {
  await page.waitForFunction(
    (sel, text) => (document.querySelector(sel)?.textContent ?? "").includes(text),
    { timeout },
    selector,
    needle,
  );
}

async function main() {
  const browser = await puppeteer.launch({
    executablePath: CHROME_PATH,
    headless: true,
    args: ["--no-sandbox"],
  });
  const page = await browser.newPage();

  const pageErrors = [];
  page.on("pageerror", (err) => pageErrors.push(String(err)));
  page.on("console", (msg) => {
    // Resource 404s are expected locally: favicon, and app.js's HEAD probe for
    // ./engine/ (which only exists on the deployed gh-pages bundle).
    if (msg.type() === "error" && !/404/.test(msg.text())) pageErrors.push(msg.text());
  });

  try {
    await page.goto(URL, { waitUntil: "networkidle0", timeout: TIMEOUT });

    await page.click("#load");
    await waitForText(page, "#loadmsg", "engine ready");
    // "engine ready" is set before initBench enables the act buttons — clicking a
    // still-disabled button would silently no-op, so wait for enablement, not text.
    await page.waitForFunction(
      () => !document.querySelector('button[data-n="100000"]')?.disabled,
      { timeout: TIMEOUT },
    );
    console.log("engine ready");

    await page.click('button[data-n="100000"]');
    await waitForText(page, "#genmsg", "rows live");
    console.log("generated:", await page.$eval("#genmsg", (e) => e.textContent));

    await page.click("#draw");
    await waitForText(page, "#drawstat", "rows drawn");
    const drawstat = await page.$eval("#drawstat", (e) => e.textContent);
    console.log("drawstat:", drawstat);

    // The first frame renders before any training step (ratio 0.00x) — wait for a
    // real tick so this also proves the learning loop is alive.
    await page.waitForFunction(
      () => {
        const m = (document.getElementById("frame")?.textContent ?? "")
          .match(/\((\d+\.\d+)x\)/);
        return m && Number(m[1]) > 0;
      },
      { timeout: TIMEOUT },
    );
    const frame = await page.$eval("#frame", (e) => e.textContent);
    const ratioMatch = frame.match(/\(([\d.]+x)\)/);
    console.log("ratio:", ratioMatch ? ratioMatch[1] : "(not found)");

    if (pageErrors.length) {
      throw new Error(`page reported errors during a successful run: ${pageErrors.join(" | ")}`);
    }

    await browser.close();
    process.exit(0);
  } catch (err) {
    console.error("verification FAILED:", err?.message ?? err);
    if (pageErrors.length) console.error("page errors:", pageErrors.join(" | "));
    // Dump the page's visible state so a timeout is diagnosable from the log alone.
    const state = await page.evaluate(() =>
      ["loadmsg", "genmsg", "drawstat"].map((id) =>
        `${id}=${JSON.stringify(document.getElementById(id)?.textContent ?? null)}`,
      ).join(" "),
    ).catch(() => "(state dump failed)");
    console.error("page state:", state);
    await browser.close();
    process.exit(1);
  }
}

main();
