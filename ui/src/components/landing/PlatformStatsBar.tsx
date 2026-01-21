"use client";

import { motion } from "framer-motion";

function PlatformStatsBar() {
  return (
    <motion.section
      className="bg-black px-6"
      initial={{ opacity: 0, y: 24 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, amount: 0.4 }}
      transition={{ duration: 0.7, ease: "easeOut" }}
    >
      <div className="max-w-7xl mx-auto">
        <div className="flex items-center justify-between rounded-2xl border border-white/10 bg-gradient-to-r from-zinc-900/70 to-zinc-900/40 px-8 py-6 backdrop-blur-md">
          
          {/* LEFT: Metrics */}
          <div className="flex gap-10">
            <Stat label="STRATEGIES ACTIVE" value="1,247" />
            <Stat label="TOTAL VOLUME" value="$2.4B" />
            <Stat label="AVG WIN RATE" value="67.3%" />
            <Stat label="UPTIME" value="99.99%" />
          </div>

          {/* RIGHT: Researchers online */}
          <div className="flex items-center gap-3">
            <div className="flex -space-x-2">
              {["JD", "MK", "AS"].map((initials, i) => (
                <div
                  key={i}
                  className="h-8 w-8 rounded-full bg-zinc-700 text-xs font-medium text-white flex items-center justify-center border border-black"
                >
                  {initials}
                </div>
              ))}
              <div className="h-8 w-8 rounded-full bg-zinc-800 text-xs text-white flex items-center justify-center border border-black">
                +
              </div>
            </div>

            <span className="text-sm text-gray-400 ml-2">
              <span className="text-white font-medium">847</span> researchers online
            </span>
          </div>
        </div>
      </div>
    </motion.section>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-xl font-semibold text-white">{value}</div>
      <div className="mt-1 text-xs tracking-widest text-gray-400">
        {label}
      </div>
    </div>
  );
}

export default PlatformStatsBar;
