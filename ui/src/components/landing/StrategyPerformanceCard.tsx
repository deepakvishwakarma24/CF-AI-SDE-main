// "use client";

// import { motion } from "framer-motion";

// const bars = [40, 55, 48, 70, 65, 80, 72];

//  function StrategyPerformanceCard() {
//   return (
//     <div className="rounded-xl border border-white/10 bg-zinc-900/50 p-6">
//       <h3 className="text-sm tracking-widest text-emerald-400 mb-6">
//         STRATEGY PERFORMANCE
//       </h3>

//       <div className="flex items-end gap-2 h-40">
//         {bars.map((h, i) => (
//           <motion.div
//             key={i}
//             initial={{ height: 0 }}
//             whileInView={{ height: `${h}%` }}
//             transition={{ delay: i * 0.08, duration: 0.6 }}
//             viewport={{ once: true }}
//             className="w-full rounded bg-emerald-500/70"
//           />
//         ))}
//       </div>

//       <div className="mt-6 flex justify-between text-sm text-gray-400">
//         <span>Total Return</span>
//         <span className="text-emerald-400 font-medium">+127.4%</span>
//       </div>
//     </div>
//   );
// }

// export default StrategyPerformanceCard



"use client";

import { motion } from "framer-motion";

const bars = [40, 55, 48, 70, 65, 80, 72];

const barBreath = {
  animate: {
    scaleY: [0.96, 1.04, 0.96],
    transition: {
      duration: 3.6,
      repeat: Infinity,
      ease: "easeInOut",
    },
  },
};

function StrategyPerformanceCard() {
  return (
    <div className="rounded-xl border border-white/10 bg-zinc-900/50 p-6">
      <h3 className="text-sm tracking-widest text-emerald-400 mb-6">
        STRATEGY PERFORMANCE
      </h3>

      <div className="flex items-end gap-2 h-40">
        {bars.map((h, i) => (
          <motion.div
            key={i}
            initial={{ height: 0 }}
            whileInView={{ height: `${h}%` }}
            transition={{ delay: i * 0.08, duration: 0.6 }}
            viewport={{ once: true }}
            variants={barBreath}
            animate="animate"
            style={{ transformOrigin: "bottom" }}
            className="w-full rounded bg-emerald-500/70"
          />
        ))}
      </div>

      <div className="mt-6 flex justify-between text-sm text-gray-400">
        <span>Total Return</span>
        <span className="text-emerald-400 font-medium">+127.4%</span>
      </div>
    </div>
  );
}

export default StrategyPerformanceCard;
