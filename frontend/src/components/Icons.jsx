/* Minimal, dependency-free line-icon set (24x24, stroke-based) — keeps the
   frontend bundle self-contained instead of pulling in an icon library. */
const base = {
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.8,
  strokeLinecap: "round",
  strokeLinejoin: "round",
};

function Svg({ children, size = 18, className }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" className={className} {...base}>
      {children}
    </svg>
  );
}

export const IconDashboard = (p) => (
  <Svg {...p}><rect x="3" y="3" width="7" height="9" rx="1.5" /><rect x="14" y="3" width="7" height="5" rx="1.5" /><rect x="14" y="12" width="7" height="9" rx="1.5" /><rect x="3" y="16" width="7" height="5" rx="1.5" /></Svg>
);
export const IconUsers = (p) => (
  <Svg {...p}><circle cx="9" cy="8" r="3.2" /><path d="M3 20c0-3.3 2.7-6 6-6s6 2.7 6 6" /><circle cx="17.5" cy="9" r="2.5" /><path d="M15.5 14.2c2.6.4 4.5 2.6 4.5 5.3" /></Svg>
);
export const IconTrendDown = (p) => (
  <Svg {...p}><path d="M3 7l7 7 4-4 7 7" /><path d="M21 12v5h-5" /></Svg>
);
export const IconLayers = (p) => (
  <Svg {...p}><path d="M12 3l9 5-9 5-9-5 9-5z" /><path d="M3 13l9 5 9-5" /></Svg>
);
export const IconTarget = (p) => (
  <Svg {...p}><circle cx="12" cy="12" r="8.5" /><circle cx="12" cy="12" r="4.5" /><circle cx="12" cy="12" r="0.8" fill="currentColor" /></Svg>
);
export const IconWallet = (p) => (
  <Svg {...p}><rect x="3" y="6" width="18" height="13" rx="2" /><path d="M3 10h18" /><circle cx="16.5" cy="14" r="1" fill="currentColor" /></Svg>
);
export const IconBarChart = (p) => (
  <Svg {...p}><path d="M4 20V10" /><path d="M12 20V4" /><path d="M20 20v-7" /></Svg>
);
export const IconActivity = (p) => (
  <Svg {...p}><path d="M3 12h4l2.5-7L14 19l2.5-7H21" /></Svg>
);
export const IconSettings = (p) => (
  <Svg {...p}><circle cx="12" cy="12" r="3" /><path d="M19.4 15a1.7 1.7 0 00.3 1.9l.1.1a2 2 0 11-2.8 2.8l-.1-.1a1.7 1.7 0 00-1.9-.3 1.7 1.7 0 00-1 1.5V21a2 2 0 11-4 0v-.1a1.7 1.7 0 00-1-1.6 1.7 1.7 0 00-1.9.3l-.1.1a2 2 0 11-2.8-2.8l.1-.1a1.7 1.7 0 00.3-1.9 1.7 1.7 0 00-1.5-1H3a2 2 0 110-4h.1a1.7 1.7 0 001.5-1 1.7 1.7 0 00-.3-1.9l-.1-.1a2 2 0 112.8-2.8l.1.1a1.7 1.7 0 001.9.3H9a1.7 1.7 0 001-1.5V3a2 2 0 114 0v.1a1.7 1.7 0 001 1.5 1.7 1.7 0 001.9-.3l.1-.1a2 2 0 112.8 2.8l-.1.1a1.7 1.7 0 00-.3 1.9V9a1.7 1.7 0 001.5 1H21a2 2 0 110 4h-.1a1.7 1.7 0 00-1.5 1z" /></Svg>
);
export const IconLogOut = (p) => (
  <Svg {...p}><path d="M9 21H5a2 2 0 01-2-2V5a2 2 0 012-2h4" /><path d="M16 17l5-5-5-5" /><path d="M21 12H9" /></Svg>
);
export const IconSearch = (p) => (
  <Svg {...p}><circle cx="11" cy="11" r="7" /><path d="M21 21l-4.3-4.3" /></Svg>
);
export const IconFilter = (p) => (
  <Svg {...p}><path d="M4 5h16l-6.5 8v6l-3-1.5v-4.5z" /></Svg>
);
export const IconChevronDown = (p) => (
  <Svg {...p}><path d="M6 9l6 6 6-6" /></Svg>
);
export const IconChevronLeft = (p) => (
  <Svg {...p}><path d="M15 18l-6-6 6-6" /></Svg>
);
export const IconChevronRight = (p) => (
  <Svg {...p}><path d="M9 18l6-6-6-6" /></Svg>
);
export const IconAlertTriangle = (p) => (
  <Svg {...p}><path d="M10.3 3.8L2.5 18a1.5 1.5 0 001.3 2.2h16.4a1.5 1.5 0 001.3-2.2L13.7 3.8a1.5 1.5 0 00-3.4 0z" /><path d="M12 9v5" /><circle cx="12" cy="17" r="0.8" fill="currentColor" /></Svg>
);
export const IconCheckCircle = (p) => (
  <Svg {...p}><circle cx="12" cy="12" r="9" /><path d="M8.5 12.5l2.5 2.5 5-5.5" /></Svg>
);
export const IconXCircle = (p) => (
  <Svg {...p}><circle cx="12" cy="12" r="9" /><path d="M9 9l6 6M15 9l-6 6" /></Svg>
);
export const IconRefresh = (p) => (
  <Svg {...p}><path d="M3 12a9 9 0 0115.3-6.4L21 8" /><path d="M21 3v5h-5" /><path d="M21 12a9 9 0 01-15.3 6.4L3 16" /><path d="M3 21v-5h5" /></Svg>
);
export const IconMenu = (p) => (
  <Svg {...p}><path d="M4 7h16M4 12h16M4 17h16" /></Svg>
);
export const IconDatabase = (p) => (
  <Svg {...p}><ellipse cx="12" cy="5.5" rx="8" ry="3" /><path d="M4 5.5V18c0 1.7 3.6 3 8 3s8-1.3 8-3V5.5" /><path d="M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3" /></Svg>
);
export const IconInfo = (p) => (
  <Svg {...p}><circle cx="12" cy="12" r="9" /><path d="M12 11v6" /><circle cx="12" cy="7.5" r="0.9" fill="currentColor" /></Svg>
);
export const IconLock = (p) => (
  <Svg {...p}><rect x="4" y="10" width="16" height="10" rx="2" /><path d="M8 10V7a4 4 0 018 0v3" /></Svg>
);
export const IconArrowUp = (p) => (
  <Svg {...p}><path d="M12 19V5M5 12l7-7 7 7" /></Svg>
);
export const IconArrowDown = (p) => (
  <Svg {...p}><path d="M12 5v14M19 12l-7 7-7-7" /></Svg>
);
export const IconX = (p) => (
  <Svg {...p}><path d="M18 6L6 18M6 6l12 12" /></Svg>
);
export const IconDownload = (p) => (
  <Svg {...p}><path d="M12 3v12" /><path d="M7 10l5 5 5-5" /><path d="M4 20h16" /></Svg>
);
