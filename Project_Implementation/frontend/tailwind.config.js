/** @type {import('tailwindcss').Config} */
export default {
    darkMode: "class",
    content: [
      "./index.html",
      "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            "colors": {
                "surface-bright": "#3b383e",
                "primary": "#cfbcff",
                "secondary-container": "#4d4465",
                "inverse-surface": "#e6e0e9",
                "surface-container-low": "#1d1b20",
                "on-primary-fixed": "#22005d",
                "on-background": "#e6e0e9",
                "on-primary-container": "#e0d2ff",
                "surface-container-lowest": "#0f0d13",
                "background": "#141218",
                "on-tertiary-fixed-variant": "#594400",
                "primary-fixed-dim": "#cfbcff",
                "error": "#ffb4ab",
                "on-error": "#690005",
                "outline": "#948e9c",
                "on-primary-fixed-variant": "#4f378a",
                "primary-fixed": "#e9ddff",
                "surface-container-high": "#2b292f",
                "on-secondary": "#342b4b",
                "surface-container-highest": "#36343a",
                "on-primary": "#381e72",
                "on-secondary-fixed-variant": "#4b4263",
                "secondary-fixed-dim": "#cdc0e9",
                "surface-variant": "#36343a",
                "primary-container": "#6750a4",
                "surface-tint": "#cfbcff",
                "inverse-on-surface": "#322f35",
                "on-secondary-fixed": "#1f1635",
                "surface": "#141218",
                "inverse-primary": "#6750a4",
                "on-secondary-container": "#bfb2da",
                "error-container": "#93000a",
                "outline-variant": "#494551",
                "tertiary": "#e7c365",
                "tertiary-fixed": "#ffdf93",
                "secondary": "#cdc0e9",
                "tertiary-container": "#c9a74d",
                "on-surface": "#e6e0e9",
                "on-tertiary": "#3e2e00",
                "on-surface-variant": "#cbc4d2",
                "surface-container": "#211f24",
                "tertiary-fixed-dim": "#e7c365",
                "surface-dim": "#141218",
                "on-tertiary-container": "#503d00",
                "secondary-fixed": "#e9ddff",
                "on-tertiary-fixed": "#241a00",
                "on-error-container": "#ffdad6"
            },
            "borderRadius": {
                "DEFAULT": "0.125rem",
                "lg": "0.25rem",
                "xl": "0.5rem",
                "full": "0.75rem"
            },
            "spacing": {
                "container-padding": "24px",
                "unit": "4px",
                "gutter": "16px",
                "stack-sm": "8px",
                "stack-lg": "32px",
                "stack-md": "16px"
            },
            "fontFamily": {
                "display": ["Inter", "sans-serif"],
                "body-lg": ["Inter", "sans-serif"],
                "body-sm": ["Inter", "sans-serif"],
                "h2": ["Inter", "sans-serif"],
                "mono-data": ["Space Grotesk", "monospace"],
                "h1": ["Inter", "sans-serif"],
                "label-caps": ["Inter", "sans-serif"]
            },
            "fontSize": {
                "display": ["48px", { "lineHeight": "1.1", "letterSpacing": "-0.04em", "fontWeight": "600" }],
                "body-lg": ["16px", { "lineHeight": "1.5", "letterSpacing": "-0.01em", "fontWeight": "400" }],
                "body-sm": ["14px", { "lineHeight": "1.5", "letterSpacing": "-0.01em", "fontWeight": "400" }],
                "h2": ["24px", { "lineHeight": "1.3", "letterSpacing": "-0.02em", "fontWeight": "500" }],
                "mono-data": ["13px", { "lineHeight": "1", "letterSpacing": "-0.02em", "fontWeight": "500" }],
                "h1": ["32px", { "lineHeight": "1.2", "letterSpacing": "-0.03em", "fontWeight": "600" }],
                "label-caps": ["11px", { "lineHeight": "1", "letterSpacing": "0.05em", "fontWeight": "600" }]
            }
        }
    },
    plugins: [],
  }
