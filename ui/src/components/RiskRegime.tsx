"use client";

import { motion } from "framer-motion";

type Regime = "RISK_ON" | "NEUTRAL" | "RISK_OFF";

export default function RiskRegime() {
  /**
   * v1: Rule-based output (mocked but realistic)
   * Later → fetched from /api/risk-regime
   */
  const regime: Regime = "RISK_ON";
  const confidence = 72; // %
  const trend = "Improving";

  const drivers = [
    "Equity momentum positive",
    "Volatility declining",
    "Crypto beta outperforming",
  ];

  const regimeConfig = {
    RISK_ON: {
      label: "RISK-ON",
      color: "emerald",
      glow: "rgba(16,185,129,0.35)",
    },
    NEUTRAL: {
      label: "NEUTRAL",
      color: "yellow",
      glow: "rgba(234,179,8,0.35)",
    },
    RISK_OFF: {
      label: "RISK-OFF",
      color: "red",
      glow: "rgba(239,68,68,0.35)",
    },
  }[regime];

  return (
    <section className="relative py-20 bg-black">
      <div className="max-w-7xl mx-auto px-6">
        {/* HEADER */}
        <div className="mb-6 flex items-center justify-between">
          <h2 className="text-xl font-semibold text-white">
            System Risk Regime
          </h2>
          <span className="text-xs text-gray-500">
            Macro environment snapshot
          </span>
        </div>

        {/* MAIN PANEL */}
        <motion.div
          initial={{ opacity: 0, y: 18 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5, ease: "easeOut" }}
          className="relative rounded-2xl border border-white/10
                     bg-gradient-to-br from-zinc-900/70 to-black/70
                     p-8 backdrop-blur"
          style={{
            boxShadow: `0 0 40px ${regimeConfig.glow}`,
          }}
        >
          {/* REGIME LABEL */}
          <div className="flex items-center gap-4">
            <div
              className={`h-3 w-3 rounded-full bg-${regimeConfig.color}-400`}
            />
            <h3
              className={`text-3xl font-bold text-${regimeConfig.color}-400`}
            >
              {regimeConfig.label}
            </h3>
          </div>

          {/* CONFIDENCE */}
          <div className="mt-6">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-gray-400">
                Signal confidence
              </span>
              <span className="text-sm text-white font-medium">
                {confidence}%
              </span>
            </div>

            <div className="h-2 rounded-full bg-white/10 overflow-hidden">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${confidence}%` }}
                transition={{ duration: 1.2, ease: "easeOut" }}
                className={`h-full bg-${regimeConfig.color}-400`}
              />
            </div>
          </div>

          {/* DRIVERS */}
          <div className="mt-6">
            <p className="text-sm text-gray-400 mb-2">
              Primary drivers
            </p>
            <ul className="space-y-1">
              {drivers.map((d, i) => (
                <li
                  key={i}
                  className="text-sm text-gray-300 flex items-center gap-2"
                >
                  <span className="text-emerald-400">•</span>
                  {d}
                </li>
              ))}
            </ul>
          </div>

          {/* TREND */}
          <div className="mt-6 text-sm text-gray-400">
            Trend (24h):{" "}
            <span className="text-white font-medium">
              {trend}
            </span>
          </div>

          {/* SUBTLE PULSE */}
          <motion.span
            className="pointer-events-none absolute inset-0 rounded-2xl"
            animate={{ opacity: [0.15, 0.25, 0.15] }}
            transition={{
              duration: 4,
              repeat: Infinity,
              ease: "easeInOut",
            }}
            style={{
              boxShadow: `inset 0 0 60px ${regimeConfig.glow}`,
            }}
          />
        </motion.div>
      </div>
    </section>
  );
}
