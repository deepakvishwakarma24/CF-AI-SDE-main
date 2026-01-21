// "use client";

// import { motion } from "framer-motion";
// // import MarketFeedCard from "./MarketFeedCard";
// // import StrategyPerformanceCard from "./StrategyPerformanceCard";
// // import SystemActivityCard from "./SystemActivityCard";


// import MarketFeedCard from "./MarketFeedCard";
// import StrategyPerformanceCard from "./StrategyPerformanceCard";
// import SystemActivityCard from "./SystemActivityCard";



// const containerVariants = {
//     hidden: { opacity: 0 },
//     visible: {
//         opacity: 1,
//         transition: {
//             staggerChildren: 0.18,
//             delayChildren: 0.15,
//         },
//     },
// };

// const cardVariants = {
//     hidden: {
//         opacity: 0,
//         y: 30,
//         scale: 0.98,
//     },
//     visible: {
//         opacity: 1,
//         y: 0,
//         scale: 1,
//         transition: {
//             duration: 0.6,
//             ease: "easeOut",
//         },
//     },
// };


// function LiveSystemSnapshot() {
//     return (

//         <motion.section
//             className="bg-black py-28 px-6"
//             initial={{ opacity: 0, y: 60 }}
//             whileInView={{ opacity: 1, y: 0 }}
//             viewport={{ once: true, amount: 0.3 }}
//             transition={{ duration: 0.9, ease: "easeOut" }}
//         >
//             <motion.div
//                 initial={{ opacity: 0, y: 40 }}
//                 whileInView={{ opacity: 1, y: 0 }}
//                 viewport={{ once: true }}
//                 transition={{ duration: 0.8, ease: "easeOut" }}
//                 className="max-w-7xl mx-auto"
//             >
//                 <h2 className="text-center text-3xl md:text-4xl font-semibold text-white">
//                     Live Research Environment
//                 </h2>

//                 <p className="mt-4 text-center text-gray-400 max-w-2xl mx-auto">
//                     A real-time snapshot of strategy signals, system activity, and
//                     performance — simulated for demonstration.
//                 </p>

//                 <motion.div
//                     variants={containerVariants}
//                     initial="hidden"
//                     whileInView="visible"
//                     viewport={{ once: true, amount: 0.3 }}
//                     className="mt-16 grid grid-cols-1 md:grid-cols-3 gap-6"
//                 >


//                     <motion.div
//                         variants={cardVariants}
//                         whileHover={{
//                             y: -6,
//                             scale: 1.015,
//                             boxShadow: "0 20px 40px rgba(16, 185, 129, 0.08)",
//                         }}
//                         transition={{ type: "spring", stiffness: 220, damping: 18 }}
//                         className="rounded-xl bg-zinc-900 p-6"
//                     >
//                         <MarketFeedCard />
//                     </motion.div>

//                     <motion.div
//                         variants={cardVariants}
//                         whileHover={{
//                             y: -6,
//                             scale: 1.015,
//                             boxShadow: "0 20px 40px rgba(16, 185, 129, 0.08)",
//                         }}
//                         transition={{ type: "spring", stiffness: 220, damping: 18 }}
//                         className="rounded-xl bg-zinc-900 p-6"
//                     >
//                         <StrategyPerformanceCard />
//                     </motion.div>

//                     <motion.div
//                         variants={cardVariants}
//                         whileHover={{
//                             y: -6,
//                             scale: 1.015,
//                             boxShadow: "0 20px 40px rgba(16, 185, 129, 0.08)",
//                         }}
//                         transition={{ type: "spring", stiffness: 220, damping: 18 }}
//                         className="rounded-xl bg-zinc-900 p-6"
//                     >

//                         <SystemActivityCard />
//                     </motion.div>
//                 </motion.div>
//             </motion.div>
//         </motion.section>
//     );
// }



// export default LiveSystemSnapshot



"use client";

import { motion, Variants } from "framer-motion";
import MarketFeedCard from "./MarketFeedCard";
import StrategyPerformanceCard from "./StrategyPerformanceCard";
import SystemActivityCard from "./SystemActivityCard";

/* ============================
   Animation Variants (TYPED)
   ============================ */

const containerVariants: Variants = {
  hidden: {},
  visible: {
    transition: {
      staggerChildren: 0.18,
      delayChildren: 0.12,
    },
  },
};

const cardVariants: Variants = {
  hidden: {
    opacity: 0,
    y: 24,
    scale: 0.98,
  },
  visible: {
    opacity: 1,
    y: 0,
    scale: 1,
    transition: {
      duration: 0.55,
      ease: "easeOut",
    },
  },
};

/* ============================
   Component
   ============================ */

 function LiveSystemSnapshot() {
  return (
    <section className="bg-black py-28 px-6">
      {/* Section entrance */}
      <motion.div
        initial={{ opacity: 0, y: 40 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, amount: 0.35 }}
        transition={{ duration: 0.8, ease: "easeOut" }}
        className="max-w-7xl mx-auto"
      >
        <h2 className="text-center text-3xl md:text-4xl font-semibold text-white">
          Live Research Environment
        </h2>

        <p className="mt-4 text-center text-gray-400 max-w-2xl mx-auto">
          A real-time snapshot of strategy signals, system activity, and
          performance — simulated for demonstration.
        </p>

        {/* Cards container (stagger controller) */}
        <motion.div
          variants={containerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, amount: 0.35 }}
          className="mt-16 grid grid-cols-1 md:grid-cols-3 gap-6"
        >
          {/* Market Feed */}
          <motion.div
            variants={cardVariants}
            whileHover={{
              y: -6,
              scale: 1.015,
              boxShadow: "0 20px 40px rgba(16,185,129,0.08)",
            }}
            transition={{ type: "spring", stiffness: 220, damping: 18 }}
            className="rounded-xl bg-zinc-900 p-6"
          >
            <MarketFeedCard />
          </motion.div>

          {/* Strategy Performance */}
          <motion.div
            variants={cardVariants}
            whileHover={{
              y: -6,
              scale: 1.015,
              boxShadow: "0 20px 40px rgba(16,185,129,0.08)",
            }}
            transition={{ type: "spring", stiffness: 220, damping: 18 }}
            className="rounded-xl bg-zinc-900 p-6"
          >
            <StrategyPerformanceCard />
          </motion.div>

          {/* System Activity */}
          <motion.div
            variants={cardVariants}
            whileHover={{
              y: -6,
              scale: 1.015,
              boxShadow: "0 20px 40px rgba(16,185,129,0.08)",
            }}
            transition={{ type: "spring", stiffness: 220, damping: 18 }}
            className="rounded-xl bg-zinc-900 p-6"
          >
            <SystemActivityCard />
          </motion.div>
        </motion.div>
      </motion.div>
    </section>
  );
}
export default LiveSystemSnapshot
