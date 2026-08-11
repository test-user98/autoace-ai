import type { Metadata } from "next";

import "./globals.css";

export const metadata: Metadata = {
  title: "AutoAce Voice Trial",
  description: "Batch emotional-tone and background-noise analysis for call audio",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
