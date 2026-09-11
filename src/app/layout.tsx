import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "SatQuery AI — ISRO Agentic Remote-Sensing Intelligence (SIH26167)",
  description: "Autonomous agentic remote-sensing VQA and multimodal spatial grounding dashboard for ISRO Earth Observation scientists.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className="bg-[#070b14] text-slate-100 antialiased min-h-screen overflow-hidden">
        {children}
      </body>
    </html>
  );
}
