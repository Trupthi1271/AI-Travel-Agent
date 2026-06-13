import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "TripWeaver AI — Travel Concierge",
  description: "AI-powered travel concierge for Indian travelers.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" style={{ height: "100%" }}>
      <body style={{ height: "100%", margin: 0, overflow: "hidden" }}>{children}</body>
    </html>
  );
}
