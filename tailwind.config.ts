import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        charcoal: {
          950: "#090a0d",
          900: "#0d0f12",
          850: "#111419",
          800: "#15181f",
          700: "#1c2028",
          600: "#252b36",
          500: "#323a48",
          400: "#4b5565",
          300: "#748094",
          200: "#a2acbd",
          100: "#d0d7e2",
        },
        coral: {
          accent: "#e07856",
          hover: "#e98767",
          subtle: "rgba(224, 120, 86, 0.12)",
          border: "rgba(224, 120, 86, 0.35)",
        },
        status: {
          amber: "#d99b32",
          red: "#d45353",
          green: "#48b27e",
          blue: "#5294e2",
        }
      },
      fontFamily: {
        mono: [
          "JetBrains Mono",
          "ui-monospace",
          "SFMono-Regular",
          "Menlo",
          "Monaco",
          "Consolas",
          "Liberation Mono",
          "Courier New",
          "monospace",
        ],
        sans: [
          "-apple-system",
          "BlinkMacSystemFont",
          "Segoe UI",
          "Roboto",
          "Helvetica Neue",
          "Arial",
          "sans-serif",
        ],
      },
    },
  },
  plugins: [],
};
export default config;
