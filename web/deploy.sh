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

rm -rf "$WT"/{index.html,app.js,bench.js,data.js,learn.js,replayhouse.js,engine}
cp web/index.html web/app.js web/bench.js web/data.js web/learn.js web/replayhouse.js "$WT/"
mkdir -p "$WT/engine"
cp -R web/node_modules/chdb-wasm/dist/st/. "$WT/engine/"
cp web/node_modules/chdb-wasm/dist/index.js "$WT/engine/" 2>/dev/null || true
touch "$WT/.nojekyll"

git -C "$WT" add -A
git -C "$WT" commit -m "deploy playground $(git rev-parse --short HEAD)" || echo "nothing to deploy"
git -C "$WT" push -u origin gh-pages
git worktree remove "$WT"
echo "enable Pages once with:"
echo "  gh api repos/{owner}/{repo}/pages -X POST -f 'source[branch]=gh-pages' -f 'source[path]=/'"
