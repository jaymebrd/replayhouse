import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { extname, join } from "node:path";

const TYPES = { ".html": "text/html", ".js": "text/javascript",
  ".mjs": "text/javascript", ".wasm": "application/wasm", ".json": "application/json" };
createServer(async (req, res) => {
  const path = join(".", decodeURIComponent(new URL(req.url, "http://x").pathname))
    .replace(/\/$/, "/index.html");
  // decodeURIComponent runs after URL dot-segment normalization, so %2F-encoded
  // ".." can survive into `path` — refuse anything escaping the web root.
  if (path.split("/").includes("..")) { res.writeHead(404); res.end("not found"); return; }
  try {
    const body = await readFile(path === "." ? "index.html" : path);
    res.writeHead(200, {
      "content-type": TYPES[extname(path)] ?? "application/octet-stream",
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    });
    res.end(body);
  } catch { res.writeHead(404); res.end("not found"); }
}).listen(8099, () => console.log("http://localhost:8099"));
