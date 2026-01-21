"use client";

import { motion } from "framer-motion";
import type { Variants } from "framer-motion";


const pulseDotVariants : Variants = {
  animate: {
    opacity: [0.4, 1, 0.4],
    scale: [1, 1.4, 1],
    transition: {
      duration: 1.6,
      repeat: Infinity,
      ease: "easeInOut",
    },
  },
};

const subtlePulseText : Variants = {
  animate: {
    opacity: [0.6, 1, 0.6],
    transition: {
      duration: 2.2,
      repeat: Infinity,
      ease: "easeInOut",
    },
  },
};

const logs = [
  "Signal detected: RSI oversold on ETH",
  "Order executed: BUY 0.5 BTC @ 97,234",
  "Backtest complete: +12.4% return",
  "Risk limit checks passed",
];

 function SystemActivityCard() {
  return (
    <div className="rounded-xl border border-white/10 bg-zinc-900/50 p-6">
      {/* <h3 className="text-sm tracking-widest text-emerald-400 mb-4">
        SYSTEM ACTIVITY
      </h3> */}

      <div className="flex items-center gap-2 mb-4">
  <motion.span
    variants={pulseDotVariants}
    animate="animate"
    className="h-2 w-2 rounded-full bg-emerald-400"
  />

  <h3 className="text-sm tracking-widest text-emerald-400">
    SYSTEM ACTIVITY
  </h3>
</div>


      <ul className="space-y-3 text-sm text-gray-300 font-mono">
        {logs.map((log, i) => (
          <motion.li
            key={i}
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            transition={{ delay: i * 0.2 }}
            viewport={{ once: true }}
          >
            • {log}
          </motion.li>
        ))}
      </ul>

      {/* <div className="mt-6 text-xs text-emerald-400">
        ● Processing 2,847 signals/sec
      </div> */}

      <motion.div
  variants={subtlePulseText}
  animate="animate"
  className="mt-6 text-xs text-emerald-400"
>
  ● Processing 2,847 signals/sec
</motion.div>

    </div>
  );
}

export default SystemActivityCard
