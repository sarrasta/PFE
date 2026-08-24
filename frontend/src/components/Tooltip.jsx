import { useState } from "react";
import { IconInfo } from "./Icons";

export function Tooltip({ text, children }) {
  const [visible, setVisible] = useState(false);
  return (
    <span
      className="tt-tooltip-wrap"
      onMouseEnter={() => setVisible(true)}
      onMouseLeave={() => setVisible(false)}
    >
      {children}
      {visible && <span className="tt-tooltip-bubble">{text}</span>}
    </span>
  );
}

export function InfoHint({ text }) {
  return (
    <Tooltip text={text}>
      <IconInfo size={14} style={{ color: "var(--text-muted)", cursor: "help" }} />
    </Tooltip>
  );
}
