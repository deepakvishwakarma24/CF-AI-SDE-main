// "use client";

// import { useEffect, useState } from "react";
// import { motion } from "framer-motion";
// import Image from "next/image";

// type Coin = {
//   id: string;
//   name: string;
//   symbol: string;
//   price: number;
//   change: number;
//   image: string;
//   marketCapRank: number;
// };

// export default function CryptoMarketLeaders() {
//   const [coins, setCoins] = useState<Coin[]>([]);
//   const [loading, setLoading] = useState(true);

//   useEffect(() => {
//     fetch("/api/crypto-market")
//       .then(res => res.json())
//       .then(data => {
//         setCoins(data);
//         setLoading(false);
//       })
//       .catch(() => setLoading(false));
//   }, []);

//   return (
//     <section className="py-20 bg-black">
//       <div className="max-w-7xl mx-auto px-6">
//         {/* HEADER */}
//         <div className="mb-8 flex items-center justify-between">
//           <h2 className="text-xl font-semibold text-white">
//             Crypto Market Leaders
//           </h2>
//           <span className="text-xs text-gray-500">
//             Top 10 by Market Cap
//           </span>
//         </div>

//         {/* LOADING */}
//         {loading ? (
//           <div className="text-gray-500 animate-pulse">
//             Fetching crypto markets…
//           </div>
//         ) : (
//           <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-5">
//             {coins.map((coin, i) => {
//               const positive = coin.change >= 0;

//               return (
//                 <motion.div
//                   key={coin.id}
//                   initial={{ opacity: 0, y: 16 }}
//                   animate={{ opacity: 1, y: 0 }}
//                   transition={{ delay: i * 0.06, duration: 0.4 }}
//                   whileHover={{
//                     y: -6,
//                     boxShadow: "0 0 0 1px rgba(59,130,246,0.35)",
//                   }}
//                   className="
//                     relative rounded-xl border border-white/10
//                     bg-gradient-to-br from-zinc-900/70 to-black/70
//                     p-4 backdrop-blur
//                   "
//                 >
//                   {/* RANK */}
//                   <span className="absolute top-3 right-3 text-xs text-gray-500">
//                     #{coin.marketCapRank}
//                   </span>

//                   {/* COIN HEADER */}
//                   <div className="flex items-center gap-3">
//                     <Image
//                       src={coin.image}
//                       alt={coin.name}
//                       width={28}
//                       height={28}
//                       className="rounded-full"
//                     />
//                     <div>
//                       <p className="text-sm font-medium text-white">
//                         {coin.name}
//                       </p>
//                       <p className="text-xs text-gray-400">
//                         {coin.symbol}
//                       </p>
//                     </div>
//                   </div>

//                   {/* PRICE */}
//                   <p className="mt-4 text-lg font-semibold text-white">
//                     ${coin.price.toLocaleString()}
//                   </p>

//                   {/* CHANGE */}
//                   <p
//                     className={`mt-1 text-sm font-medium ${
//                       positive ? "text-emerald-400" : "text-red-400"
//                     }`}
//                   >
//                     {positive ? "▲" : "▼"}{" "}
//                     {Math.abs(coin.change).toFixed(2)}%
//                   </p>

//                   {/* SUBTLE LIVE GLOW */}
//                   <span className="pointer-events-none absolute inset-0 rounded-xl opacity-0 hover:opacity-100 transition">
//                     <span className="absolute inset-0 rounded-xl bg-blue-500/5 blur-xl" />
//                   </span>
//                 </motion.div>
//               );
//             })}
//           </div>
//         )}
//       </div>
//     </section>
//   );
// }



// "use client";

// import { useEffect, useState } from "react";
// import Image from "next/image";
// import { motion } from "framer-motion";

// type Coin = {
//   id: string;
//   name: string;
//   symbol: string;
//   image: string;
//   current_price: number | null;
//   price_change_percentage_24h: number | null;
//   market_cap_rank: number;
// };

// export default function CryptoMarketLeaders() {
//   const [coins, setCoins] = useState<Coin[]>([]);
//   const [loading, setLoading] = useState(true);

//   useEffect(() => {
//     fetch(
//       "https://api.coingecko.com/api/v3/coins/markets" +
//         "?vs_currency=usd" +
//         "&order=market_cap_desc" +
//         "&per_page=10" +
//         "&page=1" +
//         "&sparkline=false"
//     )
//       .then(res => res.json())
//       .then(data => {
//         setCoins(data);
//         setLoading(false);
//       })
//       .catch(() => setLoading(false));
//   }, []);

//   return (
//     <section className="py-16 bg-black">
//       <div className="max-w-7xl mx-auto px-6">
//         {/* HEADER */}
//         <div className="mb-6 flex items-center justify-between">
//           <h2 className="text-xl font-semibold text-white">
//             Crypto Market Leaders
//           </h2>
//           <span className="text-xs text-gray-500">
//             Top 10 by Market Cap
//           </span>
//         </div>

//         {/* STRIP */}
//         {loading ? (
//           <div className="text-gray-500 animate-pulse">
//             Fetching live crypto data…
//           </div>
//         ) : (
//           <div className="relative overflow-x-auto">
//             <div className="flex gap-4 min-w-max pb-2">
//               {coins.map((coin, i) => {
//                 const price =
//                   typeof coin.current_price === "number"
//                     ? coin.current_price
//                     : null;

//                 const change =
//                   typeof coin.price_change_percentage_24h === "number"
//                     ? coin.price_change_percentage_24h
//                     : null;

//                 const positive = change !== null && change >= 0;

//                 return (
//                   <motion.div
//                     key={coin.id}
//                     initial={{ opacity: 0, y: 10 }}
//                     animate={{ opacity: 1, y: 0 }}
//                     transition={{ delay: i * 0.05 }}
//                     whileHover={{ y: -4 }}
//                     className="
//                       w-[220px] shrink-0 rounded-xl
//                       border border-white/10
//                       bg-gradient-to-br from-zinc-900/70 to-black/70
//                       p-4 backdrop-blur
//                     "
//                   >
//                     {/* TOP */}
//                     <div className="flex items-center justify-between">
//                       <div className="flex items-center gap-2">
//                         <Image
//                           src={coin.image}
//                           alt={coin.name}
//                           width={24}
//                           height={24}
//                         />
//                         <div>
//                           <p className="text-sm font-medium text-white">
//                             {coin.name}
//                           </p>
//                           <p className="text-xs uppercase text-gray-500">
//                             {coin.symbol}
//                           </p>
//                         </div>
//                       </div>

//                       <span className="text-xs text-gray-500">
//                         #{coin.market_cap_rank}
//                       </span>
//                     </div>

//                     {/* PRICE */}
//                     <div className="mt-3">
//                       <p className="text-lg font-semibold text-white">
//                         {price !== null ? (
//                           `$${price.toLocaleString()}`
//                         ) : (
//                           <span className="text-gray-500 animate-pulse">
//                             Loading
//                           </span>
//                         )}
//                       </p>

//                       {change !== null ? (
//                         <p
//                           className={`mt-1 text-xs font-medium ${
//                             positive
//                               ? "text-emerald-400"
//                               : "text-red-400"
//                           }`}
//                         >
//                           {positive ? "▲" : "▼"}{" "}
//                           {change.toFixed(2)}%
//                         </p>
//                       ) : (
//                         <p className="mt-1 text-xs text-gray-500 animate-pulse">
//                           Updating
//                         </p>
//                       )}
//                     </div>
//                   </motion.div>
//                 );
//               })}
//             </div>
//           </div>
//         )}
//       </div>
//     </section>
//   );
// }





"use client";

import { useEffect, useState } from "react";
import Image from "next/image";
import { motion } from "framer-motion";

type Coin = {
  id: string;
  name: string;
  symbol: string;
  image: string;
  current_price: number | null;
  price_change_percentage_24h: number | null;
  market_cap_rank: number;
};

export default function CryptoMarketLeaders() {
  const [coins, setCoins] = useState<Coin[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(
      "https://api.coingecko.com/api/v3/coins/markets" +
        "?vs_currency=usd" +
        "&order=market_cap_desc" +
        "&per_page=10" +
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

  // duplicate list for infinite loop illusion
  const loopCoins = [...coins, ...coins];

  return (
    <section className="py-16 bg-black overflow-hidden">
      <div className="max-w-7xl mx-auto px-6">
        {/* HEADER */}
        <div className="mb-6 flex items-center justify-between">
          <h2 className="text-xl font-semibold text-white">
            Crypto Market Leaders
          </h2>
          <span className="text-xs text-gray-500">
            Live · Top 10 by Market Cap
          </span>
        </div>

        {loading ? (
          <p className="text-gray-500 animate-pulse">
            Fetching live crypto data…
          </p>
        ) : (
          <div className="relative">
            {/* FADE EDGES */}
            <div className="pointer-events-none absolute left-0 top-0 h-full w-20 bg-gradient-to-r from-black to-transparent z-10" />
            <div className="pointer-events-none absolute right-0 top-0 h-full w-20 bg-gradient-to-l from-black to-transparent z-10" />

            {/* SLIDING STRIP */}
            <motion.div
              className="flex gap-4 w-max"
              animate={{ x: ["0%", "-50%"] }}
              transition={{
                duration: 40,
                ease: "linear",
                repeat: Infinity,
              }}
              whileHover={{ animationPlayState: "paused" }}
            >
              {loopCoins.map((coin, i) => {
                const price =
                  typeof coin.current_price === "number"
                    ? coin.current_price
                    : null;

                const change =
                  typeof coin.price_change_percentage_24h === "number"
                    ? coin.price_change_percentage_24h
                    : null;

                const positive = change !== null && change >= 0;

                return (
                  <div
                    key={`${coin.id}-${i}`}
                    className="
                      w-[220px] shrink-0 rounded-xl
                      border border-white/10
                      bg-gradient-to-br from-zinc-900/70 to-black/70
                      p-4 backdrop-blur
                    "
                  >
                    {/* TOP */}
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <Image
                          src={coin.image}
                          alt={coin.name}
                          width={24}
                          height={24}
                        />
                        <div>
                          <p className="text-sm font-medium text-white">
                            {coin.name}
                          </p>
                          <p className="text-xs uppercase text-gray-500">
                            {coin.symbol}
                          </p>
                        </div>
                      </div>

                      <span className="text-xs text-gray-500">
                        #{coin.market_cap_rank}
                      </span>
                    </div>

                    {/* PRICE */}
                    <div className="mt-3">
                      <p className="text-lg font-semibold text-white">
                        {price !== null ? (
                          `$${price.toLocaleString()}`
                        ) : (
                          <span className="text-gray-500 animate-pulse">
                            —
                          </span>
                        )}
                      </p>

                      {change !== null ? (
                        <p
                          className={`mt-1 text-xs font-medium ${
                            positive
                              ? "text-emerald-400"
                              : "text-red-400"
                          }`}
                        >
                          {positive ? "▲" : "▼"} {change.toFixed(2)}%
                        </p>
                      ) : (
                        <p className="mt-1 text-xs text-gray-500 animate-pulse">
                          Updating
                        </p>
                      )}
                    </div>
                  </div>
                );
              })}
            </motion.div>
          </div>
        )}
      </div>
    </section>
  );
}
