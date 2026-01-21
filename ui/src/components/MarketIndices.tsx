// "use client";

// import { useEffect, useState } from "react";
// import { motion } from "framer-motion";

// type Index = {
//   name: string;
//   price: number;
//   change: number;
//   percent: number;
// };

// export default function MarketIndices() {
//   const [indices, setIndices] = useState<Index[]>([]);
//   const [loading, setLoading] = useState(true);

//   useEffect(() => {
//     fetch("/api/market-indices")
//       .then(res => res.json())
//       .then(data => {
//         setIndices(data);
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
//             Market Indices
//           </h2>
//           <span className="text-xs text-gray-500">
//             Live snapshot
//           </span>
//         </div>

//         {/* LOADING */}
//         {loading ? (
//           <div className="text-gray-500 animate-pulse">
//             Fetching market data…
//           </div>
//         ) : (
//           <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
//             {indices.map((idx, i) => {
//               const positive = idx.change >= 0;

//               return (
//                 <motion.div
//                   key={idx.name}
//                   initial={{ opacity: 0, y: 14 }}
//                   animate={{ opacity: 1, y: 0 }}
//                   transition={{
//                     duration: 0.4,
//                     ease: "easeOut",
//                     delay: i * 0.08,
//                   }}
//                   whileHover={{
//                     y: -4,
//                     boxShadow: "0 0 0 1px rgba(16,185,129,0.25)",
//                   }}
//                   className="
//                     relative rounded-xl border border-blue-50
//                     bg-gradient-to-br from-zinc-900/60 to-black/60
//                     p-5 backdrop-blur
//                   "
//                 >
//                   {/* INDEX NAME */}
//                   <p className="text-xs tracking-widest text-gray-400">
//                     {idx.name}
//                   </p>

//                   {/* PRICE */}
//                   <p className="mt-2 text-2xl font-semibold text-white">
//                     {idx.price?.toLocaleString()}
//                   </p>

//                   {/* CHANGE */}
//                   <div
//                     className={`mt-1 text-sm font-medium ${
//                       positive ? "text-emerald-400" : "text-red-400"
//                     }`}
//                   >
//                     {positive ? "▲" : "▼"}{" "}
//                     {idx.change?.toFixed(2)} (
//                     {idx.percent?.toFixed(2)}%)
//                   </div>

//                   {/* SUBTLE GLOW */}
//                   <span className="pointer-events-none absolute inset-0 rounded-xl opacity-0 hover:opacity-100 transition">
//                     <span className="absolute inset-0 rounded-xl bg-emerald-500/5 blur-xl" />
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



"use client";

import { useEffect, useState } from "react";
import { motion } from "framer-motion";

type Index = {
  name: string;
  price: number | string;
  change: number | string;
  percent: number | string;
};

export default function MarketIndices() {
  const [indices, setIndices] = useState<Index[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("/api/market-indices")
      .then(res => res.json())
      .then(data => {
        setIndices(data);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  return (
    <section className="py-20 bg-black">
      <div className="max-w-7xl mx-auto px-6">
        {/* HEADER */}
        <div className="mb-8 flex items-center justify-between">
          <h2 className="text-xl font-semibold text-white">
            Market Indices
          </h2>
          <span className="text-xs text-gray-500">
            Live snapshot
          </span>
        </div>

        {/* LOADING */}
        {loading ? (
          <div className="text-gray-500 animate-pulse">
            Fetching market data…
          </div>
        ) : (
          <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
            {indices.map((idx, i) => {
              // ✅ NORMALIZE API VALUES (CRITICAL FIX)
              const price = Number(idx.price) || 0;
              const change = Number(idx.change) || 0;
              const percent = Number(
                typeof idx.percent === "string"
                  ? idx.percent.replace("%", "")
                  : idx.percent
              ) || 0;

              const positive = change >= 0;

              return (
                <motion.div
                  key={idx.name}
                  initial={{ opacity: 0, y: 14 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{
                    duration: 0.4,
                    ease: "easeOut",
                    delay: i * 0.08,
                  }}
                  whileHover={{
                    y: -4,
                    boxShadow: "0 0 0 1px rgba(59,130,246,0.35)",
                  }}
                  className="
                    relative rounded-xl
                    border border-blue-400/30
                    bg-gradient-to-br from-zinc-900/60 to-black/60
                    p-5 backdrop-blur
                  "
                >
                  {/* INDEX NAME */}
                  <p className="text-xs tracking-widest text-gray-400">
                    {idx.name}
                  </p>

                  {/* PRICE */}
                  <p className="mt-2 text-2xl font-semibold text-white">
                    {price.toLocaleString()}
                  </p>

                  {/* CHANGE */}
                  <div
                    className={`mt-1 text-sm font-medium ${
                      positive ? "text-emerald-400" : "text-red-400"
                    }`}
                  >
                    {positive ? "▲" : "▼"}{" "}
                    {change.toFixed(2)} ({percent.toFixed(2)}%)
                  </div>

                  {/* SUBTLE GLOW */}
                  <span className="pointer-events-none absolute inset-0 rounded-xl opacity-0 hover:opacity-100 transition">
                    <span className="absolute inset-0 rounded-xl bg-blue-500/5 blur-xl" />
                  </span>
                </motion.div>
              );
            })}
          </div>
        )}
      </div>
    </section>
  );
}

