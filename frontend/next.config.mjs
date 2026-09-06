/** @type {import('next').NextConfig} */

// The frontend proxies /api/* to the backend so the browser never needs
// cross-origin access and no secrets ever reach the client bundle.
const BACKEND_URL = (process.env.BACKEND_URL || "http://localhost:8000").replace(/\/+$/, "");

// `standalone` produces the self-contained server the Docker image runs, but
// `next start` refuses to serve it — so plain local/LAN serving opts out, and
// Vercel builds its own server anyway.
const standalone =
  !process.env.VERCEL && process.env.NEXT_DISABLE_STANDALONE !== "1";

const nextConfig = {
  output: standalone ? "standalone" : undefined,
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${BACKEND_URL}/api/:path*`,
      },
    ];
  },
};

export default nextConfig;
