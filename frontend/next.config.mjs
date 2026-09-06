/** @type {import('next').NextConfig} */

// The frontend proxies /api/* to the backend so the browser never needs
// cross-origin access and no secrets ever reach the client bundle.
const BACKEND_URL = (process.env.BACKEND_URL || "http://localhost:8000").replace(/\/+$/, "");

const nextConfig = {
  // `standalone` produces the self-contained server the Docker image runs.
  // Vercel builds and hosts the server itself, so leave its own output alone.
  output: process.env.VERCEL ? undefined : "standalone",
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
