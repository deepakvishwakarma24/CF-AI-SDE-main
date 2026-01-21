// import CandlestickChart from '@/components/CandlestickChart';
// import React from 'react'

//  function MarketPage() {
//   return (
//     <div>
//       <h1 className="text-2xl font-semibold mb-2">Market Data</h1>
//       <p className="text-gray-400">
//         Explore historical and real-time market data.
//         <CandlestickChart/>
//       </p>
//     </div>
//   );
// }


// export default MarketPage


"use client";

import { useState } from "react";
import CandlestickChart from "@/components/CandlestickChart";
import ClientShell from "@/components/ClientShell";

export default function MarketPage() {
  const [showSMA, setShowSMA] = useState(false);

  return (
    <ClientShell>
    <div>
      <div className="flex items-center justify-between mb-4">
        <h1 className="text-2xl font-semibold">Market Data</h1>

        <label className="flex items-center gap-2 text-sm text-gray-300">
          <input
            type="checkbox"
            checked={showSMA}
            onChange={(e) => setShowSMA(e.target.checked)}
          />
          Show SMA
        </label>
      </div>

      <div className="bg-gray-800 border border-gray-700 rounded-lg p-4">
        <CandlestickChart showSMA={showSMA} />
      </div>
    </div>
    </ClientShell>
  );
}


