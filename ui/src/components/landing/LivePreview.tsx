//  function LivePreview() {
//   return (
//     <section className="bg-black py-24">
//       <div className="max-w-7xl mx-auto px-6 grid md:grid-cols-3 gap-6">
        
//         {/* Market Feed */}
//         <div className="rounded-xl border border-white/10 p-6 bg-black">
//           <div className="flex justify-between mb-4">
//             <h3 className="text-white font-medium">Market Feed</h3>
//             <span className="text-emerald-400 text-xs">● LIVE</span>
//           </div>
//           <ul className="space-y-3 text-sm">
//             <li className="flex justify-between text-gray-300">BTC/USD <span className="text-emerald-400">+2.4%</span></li>
//             <li className="flex justify-between text-gray-300">ETH/USD <span className="text-emerald-400">+1.8%</span></li>
//             <li className="flex justify-between text-gray-300">SOL/USD <span className="text-red-400">-0.5%</span></li>
//             <li className="flex justify-between text-gray-300">AAPL <span className="text-emerald-400">+0.3%</span></li>
//           </ul>
//         </div>

//         {/* Strategy Performance */}
//         <div className="rounded-xl border border-white/10 p-6 bg-black">
//           <h3 className="text-white font-medium mb-4">Strategy Performance</h3>
//           <div className="h-40 bg-gradient-to-t from-emerald-500/30 to-transparent rounded-md" />
//           <div className="mt-4 flex justify-between text-sm">
//             <span className="text-emerald-400">+127.4%</span>
//             <span className="text-gray-400">Sharpe 2.34</span>
//           </div>
//         </div>

//         {/* System Activity */}
//         <div className="rounded-xl border border-white/10 p-6 bg-black">
//           <h3 className="text-white font-medium mb-4">System Activity</h3>
//           <ul className="text-xs space-y-2 text-gray-400">
//             <li>Signal detected: RSI oversold on ETH</li>
//             <li className="text-emerald-400">Order executed: BUY 0.5 BTC</li>
//             <li>Backtest complete: +12.4% return</li>
//             <li>Risk limit check passed</li>
//           </ul>
//         </div>

//       </div>
//     </section>
//   );
// }

// export default LivePreview


import LandingCandlestickChart from "@/components/charts/LandingCandlestickChart";

 function LivePreview() {
  return (
    <section className="bg-black py-24">
      <div className="max-w-7xl mx-auto px-6 grid md:grid-cols-3 gap-6">

        {/* Market Feed */}
        <div className="rounded-xl border border-white/10 p-6 bg-black">
          <h3 className="text-white mb-4">Market Feed</h3>
          <ul className="text-sm space-y-2 text-gray-400">
            <li>BTC/USD <span className="float-right text-emerald-400">+2.4%</span></li>
            <li>ETH/USD <span className="float-right text-emerald-400">+1.8%</span></li>
            <li>SOL/USD <span className="float-right text-red-400">-0.5%</span></li>
          </ul>
        </div>

        {/* Candlestick Chart */}
        <div className="rounded-xl border border-white/10 p-4 bg-black">
          <LandingCandlestickChart />
        </div>

        {/* System Activity */}
        <div className="rounded-xl border border-white/10 p-6 bg-black text-xs text-gray-400">
          <p>Signal detected: RSI oversold on ETH</p>
          <p className="text-emerald-400">Order executed: BUY 0.5 BTC</p>
          <p>Backtest complete: +12.4% return</p>
          <p>Risk checks passed</p>
        </div>

      </div>
    </section>
  );
}

export default LivePreview

