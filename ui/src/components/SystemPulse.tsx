// "use client";

// import { motion } from "framer-motion";

//  function SystemPulse() {
//   return (
//     <section className="relative h-[260px] overflow-hidden bg-black">
//       {/* VERY SUBTLE GRID */}
//       <div
//         className="absolute inset-0 opacity-[0.05]"
//         style={{
//           backgroundImage:
//             "linear-gradient(rgba(16,185,129,0.15) 1px, transparent 1px), linear-gradient(90deg, rgba(16,185,129,0.15) 1px, transparent 1px)",
//           backgroundSize: "80px 80px",
//         }}
//       />

//       {/* ANIMATED SYSTEM LINES */}
//       <motion.svg
//         viewBox="0 0 1200 260"
//          className="absolute inset-0 w-full h-full"
//         // className="absolute inset-0 flex items-center justify-center -translate-y-6"
//         initial={{ x: 0 }}
//         animate={{ x: [-40, 40, -40] }}
//         transition={{
//           duration: 45,
//           repeat: Infinity,
//           ease: "linear",
//         }}
//       >
//         {/* Primary Pulse Line */}
//         <path
//           d="M0 160 C 200 140, 400 180, 600 150 C 800 120, 1000 160, 1200 140"
//           fill="none"
//           stroke="rgba(59,130,246,0.35)"
//           strokeWidth="2.2"
//         />

//         {/* Secondary Faint Line */}
//         <path
//           d="M0 190 C 300 210, 600 170, 900 200 C 1050 215, 1150 200, 1200 210"
//           fill="none"
//           stroke="rgba(16,185,129,0.25)"
//           strokeWidth="1.5"
//         />
//       </motion.svg>

//       {/* MOVING PULSE DOTS */}
//       {[0, 1, 2].map(i => (
//         <motion.div
//           key={i}
//           className="absolute top-1/2 h-1.5 w-1.5 rounded-full bg-emerald-400/70"
//           initial={{ x: "-10%" }}
//           animate={{ x: "110%" }}
//           transition={{
//             duration: 18 + i * 6,
//             repeat: Infinity,
//             ease: "linear",
//             delay: i * 3,
//           }}
//           style={{
//             marginTop: `${i * 14 - 14}px`,
//           }}
//         />
//       ))}

//       {/* TOP & BOTTOM FADE */}
//       <div className="pointer-events-none absolute inset-0 bg-gradient-to-b from-black via-transparent to-black" />
//     </section>
//   );
// }


// export default SystemPulse



// "use client";

// import { motion } from "framer-motion";

// export default function SystemPulse() {
//   return (
//     <section className="relative h-[280px] bg-black overflow-hidden">
//       {/* SUBTLE GRID */}
//       {/* SUBTLE GRID */}
//         <div
//   className="absolute inset-0 z-[1] pointer-events-none"
//   style={{
//     backgroundImage: `
//       linear-gradient(to right, rgba(16,185,129,0.35) 1px, transparent 1px),
//       linear-gradient(to bottom, rgba(16,185,129,0.35) 1px, transparent 1px)
//     `,
//     backgroundSize: "72px 72px",
//     opacity: 0.12,
//   }}
// />

//       {/* SIGNAL LINES */}
//       <motion.svg
//         viewBox="0 0 1000 120"
//         className="relative z-[3] mx-auto w-full max-w-6xl opacity-40"
//         initial={{ opacity: 0 }}
//         whileInView={{ opacity: 1 }}
//         viewport={{ once: true }}
//       >
//         {/* BLUE SIGNAL */}
//         <motion.path
//           d="M0 70 C150 60, 300 80, 450 65 S700 55, 1000 60"
//           fill="none"
//           stroke="rgba(59,130,246,0.35)"
//           strokeWidth="2"
//           animate={{ opacity: [0.4, 0.7, 0.4] }}
//           transition={{ duration: 6, repeat: Infinity, ease: "easeInOut" }}
//         />

//         {/* GREEN SIGNAL */}
//         <motion.path
//           d="M0 85 C200 90, 400 70, 600 80 S800 95, 1000 85"
//           fill="none"
//           stroke="rgba(16,185,129,0.35)"
//           strokeWidth="2"
//           animate={{ opacity: [0.35, 0.6, 0.35] }}
//           transition={{ duration: 8, repeat: Infinity, ease: "easeInOut" }}
//         />

//         {/* SYSTEM NODES */}
// {[120, 300, 520, 760].map((x, i) => (
//   <motion.circle
//     key={i}
//     cx={x}
//     cy={65}
//     r={3}
//     fill="rgb(16,185,129)"
//     opacity="0.9"
//     initial={{ opacity: 0.3 }}
//     animate={{ opacity: [0.3, 0.9, 0.3] }}
//     transition={{
//       duration: 2.5,
//       repeat: Infinity,
//       delay: i * 0.4,
//       ease: "easeInOut",
//     }}
//   />
// ))}

//       </motion.svg>
//     </section>
//   );
// }


"use client";

import { motion } from "framer-motion";

export default function SystemPulse() {
  return (
    <section className="relative h-[280px] bg-black overflow-hidden">
      
      {/* SVG LAYER (GRID + SIGNALS) */}
      <motion.svg
        viewBox="0 0 1200 280"
        className="absolute inset-0 w-full h-full"
      >
        {/* SVG GRID (THIS CANNOT DISAPPEAR) */}
        <defs>
          <pattern
            id="grid"
            width="72"
            height="72"
            patternUnits="userSpaceOnUse"
          >
            <path
              d="M 72 0 L 0 0 0 72"
              fill="none"
              stroke="rgba(16,185,129,0.25)"
              strokeWidth="1"
            />
          </pattern>
        </defs>

        <rect
          width="100%"
          height="100%"
          fill="url(#grid)"
          opacity="0.18"
        />

        {/* BLUE SIGNAL */}
        <motion.path
          d="M0 150 C200 130, 400 170, 600 145 S900 135, 1200 145"
          fill="none"
          stroke="rgba(59,130,246,0.4)"
          strokeWidth="2"
          animate={{ opacity: [0.5, 0.8, 0.6] }}
          transition={{ duration: 6, repeat: Infinity, ease: "easeInOut" }}
        />

        {/* GREEN SIGNAL */}
        <motion.path
          d="M0 170 C300 190, 600 150, 900 180 S1100 190, 1200 175"
          fill="none"
          stroke="rgba(16,185,129,0.55)"
          strokeWidth="2"
          animate={{ opacity: [0.5, 0.8, 0.5] }}
          transition={{ duration: 8, repeat: Infinity, ease: "easeInOut" }}
        />

        {/* PULSE DOTS */}
        {[180, 420, 700, 980].map((x, i) => (
          <motion.circle
            key={i}
            cx={x}
            cy={150}
            r={3}
            fill="rgb(16,185,129)"
            animate={{ opacity: [0.5, 1, 0.6] }}
            transition={{
              duration: 2.2,
              repeat: Infinity,
              delay: i * 0.5,
              ease: "easeInOut",
            }}
          />
        ))}
      </motion.svg>

      {/* DARK FADE (SAFE NOW) */}
      <div className="absolute inset-0 bg-gradient-to-b from-black via-transparent to-black" />
    </section>
  );
}
