import type { Metadata } from "next";
import { Inter } from "next/font/google";
import { ThemeProvider } from "@/components/ThemeProvider";
import "./globals.css";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
  fallback: ["system-ui"],
});

export const metadata: Metadata = {
  title: "GPT Studio",
  description: "Autoregressive GPT-2 Playground",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${inter.variable} h-full antialiased`}
      suppressHydrationWarning
    >
      <body className="min-h-full flex flex-col bg-background text-foreground transition-colors duration-500">
        <ThemeProvider>
          {children}
        </ThemeProvider>
      </body>
    </html>
  );
}
