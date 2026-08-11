import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // The dashboard is the control plane (PLAN.md §1.5): it uploads, calls Modal,
  // and polls. It must never touch audio, so no large-body config is needed.
  reactStrictMode: true,
};

export default nextConfig;
