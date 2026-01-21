"use client";

import { motion } from "framer-motion";

const candles = [
  { x: 0, open: 40, close: 65, high: 72, low: 32 },
  { x: 1, open: 62, close: 50, high: 70, low: 45 },
  { x: 2, open: 48, close: 78, high: 82, low: 44 },
  { x: 3, open: 76, close: 68, high: 84, low: 60 },
  { x: 4, open: 66, close: 88, high: 92, low: 62 },
  { x: 5, open: 86, close: 80, high: 94, low: 74 },
  { x: 6, open: 78, close: 96, high: 100, low: 76 },
  { x: 7, open: 94, close: 88, high: 102, low: 84 },
];

 function CandlestickInfographic() {
  return (
    <div className="absolute inset-0 z-0 pointer-events-none overflow-hidden">
      {/* GRID */}
      <div
        className="absolute inset-0 opacity-[0.06]"
        style={{
          backgroundImage:
            "linear-gradient(rgba(16,185,129,0.15) 1px, transparent 1px), linear-gradient(90deg, rgba(16,185,129,0.15) 1px, transparent 1px)",
          backgroundSize: "60px 60px",
        }}
      />

      {/* CHART LAYER */}
      <motion.div
        className="absolute inset-0 flex items-center justify-center"
        initial={{ x: 0 }}
        animate={{ x: [-40, 40, -40] }}
        transition={{
          duration: 40, // 👈 ultra slow
          repeat: Infinity,
          ease: "linear",
        }}
      >
        <svg
          viewBox="0 0 800 300"
          className="w-[220%] h-[440px] opacity-[0.85] blur-[0.4px]"
        >
          {/* BLUE CURVE (FADED) */}
          <path
            d="M50 220
               C150 180, 250 140, 350 120
               S550 140, 700 80"
            fill="none"
            stroke="rgba(59,130,246,0.22)"
            strokeWidth="3"
          />

          {/* CANDLESTICKS */}
          {candles.map((c, i) => {
            const isBullish = c.close > c.open;
            const color = isBullish
              ? "rgba(16,185,129,0.45)"
              : "rgba(239,68,68,0.4)";

            const x = 80 + i * 85;
            const bodyTop = Math.min(c.open, c.close);
            const bodyHeight = Math.abs(c.close - c.open);

            return (
              <g key={i}>
                {/* Wick */}
                <line
                  x1={x}
                  x2={x}
                  y1={300 - c.high}
                  y2={300 - c.low}
                  stroke={color}
                  strokeWidth="2"
                  opacity="0.6"
                />

                {/* Body */}
                <rect
                  x={x - 10}
                  y={260 - bodyTop}
                  width="20"
                  height={bodyHeight}
                  fill={color}
                  rx="2"
                />
              </g>
            );
          })}
        </svg>
      </motion.div>

      {/* DARK FADE MASK (TOP + BOTTOM) */}
      <div className="absolute inset-0 bg-gradient-to-b from-black/70 via-transparent to-black/70" />
    </div>
  );
}


export default CandlestickInfographic