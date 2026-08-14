#!/usr/bin/env bash
# Assemble and push the playground to the gh-pages branch.
# The ~99MB wasm lives ONLY on gh-pages, never on main.
set -euo pipefail
cd "$(dirname "$0")/.."

test -d web/node_modules/chdb-wasm || { echo "run: npm --prefix web install"; exit 1; }

WT=$(mktemp -d)
git worktree add "$WT" gh-pages 2>/dev/null || {
  git worktree add --detach "$WT"
  git -C "$WT" checkout --orphan gh-pages
  git -C "$WT" rm -rf --quiet . 2>/dev/null || true
}

rm -rf "$WT"/{index.html,app.js,bench.js,data.js,race.js,learn.js,replayhouse.js,engine,coi-serviceworker.min.js}
cp web/index.html web/app.js web/bench.js web/data.js web/race.js web/replayhouse.js "$WT/"
cp web/coi-serviceworker.min.js "$WT/"
mkdir -p "$WT/engine"
cp web/node_modules/chdb-wasm/dist/*.js "$WT/engine/"                          # glue: index/async/status/platform/bindings/worker/protocol
cp web/node_modules/chdb-wasm/dist/chdb.mjs web/node_modules/chdb-wasm/dist/chdb.wasm "$WT/engine/"  # mt bundle lives at dist root
# Real-browser testing showed the st (single-threaded) bundle can't run MergeTree at all
# (CANNOT_SCHEDULE_TASK on CREATE TABLE) — so we ship mt-only. GitHub Pages can't set the
# COOP/COEP headers mt needs, so coi-serviceworker.min.js (vendored above) shims cross-origin
# isolation client-side via a one-time reload. selectBundle's mt path is `${base}/chdb.mjs`,
# i.e. no st/ nesting.
# Sanity check: the deployed page must be able to boot (worker.js/bindings.js/status.js
# are the worker-side chain — AsyncChdb spawns worker.js, which imports the others)
for f in engine/index.js engine/async.js engine/platform.js engine/worker.js \
         engine/bindings.js engine/status.js engine/chdb.mjs engine/chdb.wasm \
         coi-serviceworker.min.js; do
  test -f "$WT/$f" || { echo "engine bundle incomplete: missing $f"; exit 1; }
done
# GitHub hard-rejects files >= 100 MiB; chdb.wasm sits ~0.7 MiB under that today,
# so an engine-version bump can silently cross the line — fail loudly instead.
wasm_bytes=$(wc -c < "$WT/engine/chdb.wasm")
test "$wasm_bytes" -lt 104857600 || {
  echo "engine/chdb.wasm is $wasm_bytes bytes, over GitHub's 100 MiB push limit"; exit 1; }
touch "$WT/.nojekyll"

git -C "$WT" add -A
git -C "$WT" commit -m "deploy playground $(git rev-parse --short HEAD)" || echo "nothing to deploy"
git -C "$WT" push -u origin gh-pages
git worktree remove "$WT"
echo "enable Pages once with:"
echo "  gh api repos/{owner}/{repo}/pages -X POST -f 'source[branch]=gh-pages' -f 'source[path]=/'"
