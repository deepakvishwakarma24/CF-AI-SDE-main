"use client";

import { useEffect, useState } from "react";
import Image from "next/image";
import { motion } from "framer-motion";

type Coin = {
  id: string;
  name: string;
  symbol: string;
  image: string;
  market_cap: number;
  price_change_percentage_24h: number | null;
};

function CryptoHeatmap() {
  const [coins, setCoins] = useState<Coin[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(
      "https://api.coingecko.com/api/v3/coins/markets" +
        "?vs_currency=usd" +
        "&order=market_cap_desc" +
        "&per_page=20" +
        "&page=1" +
        "&sparkline=false"
    )
      .then(res => res.json())
      .then(data => {
        setCoins(data);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  // Normalize size based on market cap
  const maxCap = Math.max(...coins.map(c => c.market_cap || 0));

  const getColor = (change: number | null) => {
    if (change === null) return "bg-zinc-800";

    if (change > 5) return "bg-emerald-500/40";
    if (change > 0) return "bg-emerald-500/25";
    if (change > -5) return "bg-red-500/25";
    return "bg-red-500/40";
  };

  return (
    <section className="py-20 bg-black">
      <div className="max-w-7xl mx-auto px-6">
        {/* HEADER */}
        <div className="mb-8 flex items-center justify-between">
          <h2 className="text-xl font-semibold text-white">
            Crypto Market Heatmap
          </h2>
          <span className="text-xs text-gray-500">
            Size = Market Cap · Color = 24h Change
          </span>
        </div>

        {loading ? (
          <p className="text-gray-500 animate-pulse">
            Loading market heatmap…
          </p>
        ) : (
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3 auto-rows-[140px]">
            {coins.map((coin, i) => {
              const sizeFactor =
                coin.market_cap && maxCap
                  ? 1 + coin.market_cap / maxCap
                  : 1;

              return (
                <motion.div
                  key={coin.id}
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: i * 0.03 }}
                  whileHover={{ scale: 1.05 }}
                  className={`
                    relative rounded-xl p-4 border border-white/10
                    ${getColor(coin.price_change_percentage_24h)}
                  `}
                  style={{
                    gridRow: `span ${sizeFactor > 1.6 ? 2 : 1}`,
                    gridColumn: `span ${sizeFactor > 1.8 ? 2 : 1}`,
                  }}
                >
                  {/* COIN */}
                  <div className="flex items-center gap-2">
                    <Image
                      src={coin.image}
                      alt={coin.name}
                      width={22}
                      height={22}
                    />
                    <div>
                      <p className="text-sm font-medium text-white">
                        {coin.name}
                      </p>
                      <p className="text-xs uppercase text-gray-300">
                        {coin.symbol}
                      </p>
                    </div>
                  </div>

                  {/* CHANGE */}
                  <div className="absolute bottom-3 left-4">
                    {coin.price_change_percentage_24h !== null ? (
                      <p
                        className={`text-sm font-semibold ${
                          coin.price_change_percentage_24h >= 0
                            ? "text-emerald-300"
                            : "text-red-300"
                        }`}
                      >
                        {coin.price_change_percentage_24h >= 0 ? "+" : ""}
                        {coin.price_change_percentage_24h.toFixed(2)}%
                      </p>
                    ) : (
                      <p className="text-xs text-gray-400">—</p>
                    )}
                  </div>
                </motion.div>
              );
            })}
          </div>
        )}
      </div>
    </section>
  );
}


export default CryptoHeatmap
