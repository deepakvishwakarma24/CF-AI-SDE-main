"use client";

import { motion } from "framer-motion";

const points = [
  { x: 0, y: 60 },
  { x: 15, y: 55 },
  { x: 30, y: 70 },
  { x: 45, y: 68 },
  { x: 60, y: 80 },
  { x: 75, y: 78 },
  { x: 90, y: 90 },
];

export default function BackgroundTrendChart() {
  return (
    <div className="absolute inset-0 overflow-hidden pointer-events-none">
      {/* grid */}
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#ffffff08_1px,transparent_1px),linear-gradient(to_bottom,#ffffff08_1px,transparent_1px)] bg-[size:80px_80px]" />

      {/* animated trend line */}
      <svg
        viewBox="0 0 100 100"
        preserveAspectRatio="none"
        className="absolute inset-0 w-full h-full"
      >
        <motion.path
          d={`M ${points.map(p => `${p.x} ${p.y}`).join(" L ")}`}
          fill="none"
          stroke="rgba(16,185,129,0.35)"
          strokeWidth="0.8"
          initial={{ pathLength: 0 }}
          animate={{ pathLength: 1 }}
          transition={{
            duration: 3.5,
            ease: "easeInOut",
            repeat: Infinity,
            repeatType: "mirror",
          }}
        />
      </svg>
    </div>
  );
}
