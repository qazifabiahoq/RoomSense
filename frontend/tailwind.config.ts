import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}", "./lib/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        brand: {
          50: "#FBF3EC",
          100: "#F5E1CE",
          200: "#EAC09C",
          300: "#DE9E69",
          400: "#D07D42",
          500: "#BC6329",
          600: "#9C501F",
          700: "#7C3F1A",
          800: "#5F3016",
          900: "#492611",
        },
      },
      fontFamily: {
        display: ["var(--font-space-grotesk)", "sans-serif"],
        sans: ["var(--font-inter)", "sans-serif"],
      },
      keyframes: {
        shimmer: {
          "0%, 100%": { opacity: "1" },
          "50%": { opacity: "0.6" },
        },
        fadeUp: {
          "0%": { opacity: "0", transform: "translateY(8px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
      },
      animation: {
        shimmer: "shimmer 1.6s ease-in-out infinite",
        fadeUp: "fadeUp 0.4s ease-out",
      },
    },
  },
  plugins: [],
};

export default config;
