const CANONICAL_HOST = "vamos-optimization.org";
const REDIRECT_HOSTS = new Set([
  "www.vamos-optimization.org",
  "vamos-optimization.dev",
  "www.vamos-optimization.dev",
]);
const LEGACY_VERSION = /^\/(\d+\.\d+\.\d+)(\/.*)?$/;

function canonicalPath(pathname) {
  if (pathname === "/" || pathname === "/docs" || pathname === "/docs/") {
    return "/docs/stable/";
  }
  if (pathname === "/latest" || pathname === "/latest/") {
    return "/docs/stable/";
  }
  if (pathname.startsWith("/latest/")) {
    return `/docs/stable/${pathname.slice("/latest/".length)}`;
  }
  const match = LEGACY_VERSION.exec(pathname);
  if (match) {
    return `/docs/${match[1]}${match[2] ?? "/"}`;
  }
  return pathname;
}

function redirectTarget(request) {
  const source = new URL(request.url);
  const target = new URL(source);

  if (REDIRECT_HOSTS.has(target.hostname)) {
    target.hostname = CANONICAL_HOST;
  }
  target.protocol = "https:";
  target.port = "";
  target.pathname = canonicalPath(target.pathname);

  if (
    target.protocol !== source.protocol ||
    target.hostname !== source.hostname ||
    target.port !== source.port ||
    target.pathname !== source.pathname
  ) {
    return target;
  }
  return null;
}

export default {
  async fetch(request, env) {
    const target = redirectTarget(request);
    if (target !== null) {
      return Response.redirect(target.toString(), 308);
    }
    return env.ASSETS.fetch(request);
  },
};
