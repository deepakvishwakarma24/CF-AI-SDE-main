//  import Link from "next/link";

//  import EquityCurveBackground from "@/components/landing/EquityCurveBackground";

//  function Hero() {
//   return (
//      <section className="min-h-screen pt-32 flex items-center justify-center bg-black relative overflow-hidden">
// {/* //  <section className="min-h-screen w-screen bg-black relative overflow-hidden">  */}
//       <EquityCurveBackground />
//       {/* subtle grid */}
//       <div className="absolute inset-0 bg-[linear-gradient(to_right,#ffffff08_1px,transparent_1px),linear-gradient(to_bottom,#ffffff08_1px,transparent_1px)] bg-[size:60px_60px]" />
      

//       <div className="relative z-10 text-center max-w-4xl px-6">
//       {/* <div className="relative z-10 max-w-5xl mx-auto px-6 text-center">  */}

//         <span className="inline-block mb-6 px-4 py-1 text-xs tracking-widest text-emerald-400 border border-emerald-400/30 rounded-full">
//           ENTERPRISE RESEARCH ENVIRONMENT v1.0
//         </span>

//         <h1 className="text-5xl md:text-6xl font-bold text-white leading-tight">
//           Design, Test, and Understand
//           <br />
//           <span className="text-emerald-400">Trading Strategies</span>
//           <br />
//           Systematically.
//         </h1>

//         <p className="mt-6 text-gray-400 text-lg">
//           A high-performance research ecosystem for backtesting,
//           risk management, and AI-driven alpha discovery.
//         </p>

//         <div className="mt-10 flex justify-center gap-4">
//          <Link href="/market">
//   <button className="bg-white text-black px-6 py-3 rounded-lg font-semibold hover:opacity-90 transition">
//     Launch Platform →
//   </button>
// </Link>

//           <button className="px-6 py-3 border border-white/20 text-white rounded-md">
//             View Architecture
//           </button>
//         </div>
//       </div>
//     </section>
//   );
// }


// export default Hero


import Link from "next/link";
import EquityCurveBackground from "@/components/landing/EquityCurveBackground";

function Hero() {
  return (
    <section className="relative h-screen bg-black overflow-hidden">
      {/* BACKGROUND CHART */}
      <div className="absolute inset-0 z-0">
        <EquityCurveBackground />
      </div>

     


      {/* GRID OVERLAY */}
      <div className="absolute inset-0 z-[1] bg-[linear-gradient(to_right,#ffffff08_1px,transparent_1px),linear-gradient(to_bottom,#ffffff08_1px,transparent_1px)] bg-[size:60px_60px]" />

      {/* HERO CONTENT */}
      <div className="relative z-10 flex h-full items-center justify-center text-center">
        <div className="max-w-4xl px-6">
          <span className="inline-block mb-6 px-4 py-1 text-xs tracking-widest text-emerald-400 border border-emerald-400/30 rounded-full">
            ENTERPRISE RESEARCH ENVIRONMENT v1.0
          </span>

          <h1 className="text-5xl md:text-6xl font-bold text-white leading-tight">
            Design, Test, and Understand
            <br />
            <span className="text-emerald-400">Trading Strategies</span>
            <br />
            Systematically.
          </h1>

          <p className="mt-6 text-gray-400 text-lg">
            A high-performance research ecosystem for backtesting,
            risk management, and AI-driven alpha discovery.
          </p>

          <div className="mt-10 flex justify-center gap-4">
            <Link href="/market">
              <button className="bg-white text-black px-6 py-3 rounded-lg font-semibold hover:opacity-90 transition">
                Launch Platform →
              </button>
            </Link>

            <button className="px-6 py-3 border border-white/20 text-white rounded-md">
              View Architecture
            </button>
          </div>
        </div>
      </div>
    </section>
  );
}

export default Hero;
