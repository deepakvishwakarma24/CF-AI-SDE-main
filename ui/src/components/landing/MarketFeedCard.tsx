"use client";

import { motion } from "framer-motion";

const markets = [
  { symbol: "BTC/USD", price: "97,234.50", change: "+2.4%" },
  { symbol: "ETH/USD", price: "3,456.78", change: "+1.8%" },
  { symbol: "SOL/USD", price: "187.23", change: "-0.5%" },
  { symbol: "AAPL", price: "198.45", change: "+0.3%" },
];

 function MarketFeedCard() {
  return (
    <div className="rounded-xl border border-white/10 bg-zinc-900/50 p-6">
      <h3 className="text-sm tracking-widest text-emerald-400 mb-4">
        MARKET FEED
      </h3>

      <ul className="space-y-3">
        {markets.map((m, i) => (
          <motion.li
            key={m.symbol}
            initial={{ opacity: 0, x: -10 }}
            whileInView={{ opacity: 1, x: 0 }}
            transition={{ delay: i * 0.1 }}
            viewport={{ once: true }}
            className="flex justify-between text-sm text-gray-300"
          >
            <span>{m.symbol}</span>
            <span>
              ${m.price}{" "}
              <span
                className={
                  m.change.startsWith("-")
                    ? "text-red-400"
                    : "text-emerald-400"
                }
              >
                {m.change}
              </span>
            </span>
          </motion.li>
        ))}
      </ul>
    </div>
  );
}

export default MarketFeedCard
