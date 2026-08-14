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
    console.log("engine ready");

    // Act 1 (memory/spotlight) autostarts — wait for real steps through the
    // store: experiences inserted and both students evaluated.
    await page.waitForFunction(
      () => /% sharp/.test(document.getElementById("memstat")?.textContent ?? ""),
      { timeout: TIMEOUT },
    );
    console.log("memory:", await page.$eval("#memstat", (e) => e.textContent));

    // Act 2 (the race) starts on click, taking the stage from act 1.
    await page.click("#rpause");
    await page.waitForFunction(
      () => {
        const t = document.getElementById("racestat")?.textContent ?? "";
        const m = t.match(/step (\d+)/);
        return m && Number(m[1]) >= 5 && t.includes("% sharp");
      },
      { timeout: TIMEOUT },
    );
    console.log("race:", await page.$eval("#racestat", (e) => e.textContent));

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
      ["loadmsg", "memstat", "racestat"].map((id) =>
        `${id}=${JSON.stringify(document.getElementById(id)?.textContent ?? null)}`,
      ).join(" "),
    ).catch(() => "(state dump failed)");
    console.error("page state:", state);
    await browser.close();
    process.exit(1);
  }
}

main();
