const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const test = require("node:test");
const source = fs.readFileSync("docs/javascripts/api-reference.js", "utf8");
const bookmarks = JSON.parse(fs.readFileSync("docs/reference/api/legacy-bookmarks.json", "utf8")).bookmarks;

function run(href, options = {}) {
  const replacements = [];
  const listeners = {};
  const url = new URL(href);
  const nodes = {};
  for (const [id, target] of Object.entries(bookmarks)) {
    nodes[id] = {
      hasAttribute: name => name === "data-api-legacy",
      getAttribute: () => options.badTarget || target,
    };
  }
  const directory = { contains: node => Object.values(nodes).includes(node) };
  const location = { href, hash: url.hash, replace: value => replacements.push(value) };
  const document = {
    readyState: "complete",
    getElementById: id => id === "api-legacy-bookmarks" ? (options.otherPage ? null : directory) : nodes[id],
    addEventListener: (event, fn) => { listeners[event] = fn; },
  };
  vm.runInNewContext(source, {
    URL, decodeURIComponent, document,
    window: { location, addEventListener: (event, fn) => { listeners[event] = fn; } },
  });
  return { replacements, listeners, location };
}

for (const base of [
  "https://vamos-optimization.org/reference/api_reference/",
  "https://vamos-optimization.org/docs/1.0.0/reference/api_reference/",
  "https://pr-33-vamos-docs-preview.vamos-optimization.workers.dev/reference/api_reference/",
  "http://127.0.0.1:8001/reference/api_reference/",
]) {
  test(`all frozen bookmarks keep their origin and route context: ${base}`, () => {
    for (const [old, target] of Object.entries(bookmarks)) {
      const expected = new URL(target, base);
      expected.search = "?source=bookmark&value=1";
      assert.deepEqual(run(`${base}?source=bookmark&value=1#${encodeURIComponent(old)}`).replacements, [expected.href]);
    }
  });
}

test("unknown, malformed, empty, and ordinary heading fragments stay on the index", () => {
  for (const hash of ["", "#missing-symbol", "#%", "#api-reference", "#core-optimization"]) {
    assert.deepEqual(run(`https://example.org/reference/api_reference/${hash}`).replacements, []);
  }
});
test("unrelated pages and unsafe targets cannot redirect", () => {
  const base = "https://example.org/reference/api_reference/#unified-api";
  assert.deepEqual(run(base, { otherPage: true }).replacements, []);
  for (const badTarget of ["https://evil.invalid/", "/docs/future/api/", "javascript:alert(1)"]) {
    assert.deepEqual(run(base, { badTarget }).replacements, []);
  }
});
test("index.html and slashless local entry points resolve the same way", () => {
  const base = "https://example.org/docs/1.0.0/reference/api_reference";
  const expected = "https://example.org/docs/1.0.0/reference/api/optimization/";
  for (const suffix of ["", "/", "/index.html"]) {
    assert.deepEqual(run(base + suffix + "#unified-api").replacements, [expected]);
  }
});
test("a hash change after initial load is handled", () => {
  const result = run("https://example.org/reference/api_reference/");
  result.location.hash = "#unified-api";
  result.location.href += "#unified-api";
  result.listeners.hashchange();
  assert.deepEqual(result.replacements, ["https://example.org/reference/api/optimization/"]);
});
