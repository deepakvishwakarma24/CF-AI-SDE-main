// import React from 'react'

//  function BacktestPage() {
//   return (
//     <div>
//       <h1 className="text-2xl font-semibold mb-2">Backtesting</h1>
//       <p className="text-gray-400">
//         Simulate strategy performance on historical data.
//       </p>
//     </div>
//   );
// }

// export default BacktestPage


"use client";

import { useState } from "react";
import { mockBacktestResults } from "@/data/backtestResults";
import ClientShell from "@/components/ClientShell";

 function BacktestPage() {
  const [isRunning, setIsRunning] = useState(false);
  const [results, setResults] = useState<any>(null);

  const runBacktest = () => {
    setIsRunning(true);

    setTimeout(() => {
      setResults(mockBacktestResults);
      setIsRunning(false);
    }, 2000);
  };

  return (
    <ClientShell>
    <div className="space-y-6">
      <h1 className="text-2xl font-semibold">Backtesting</h1>

      {!results && (
        <button
          onClick={runBacktest}
          disabled={isRunning}
          className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded"
        >
          {isRunning ? "Running Backtest..." : "Run Backtest"}
        </button>
      )}

      {results && (
        <>
          {/* Metrics */}
          <div className="grid grid-cols-4 gap-4">
            <Metric label="Total Return" value={`${results.metrics.totalReturn}%`} />
            <Metric label="Sharpe Ratio" value={results.metrics.sharpeRatio} />
            <Metric label="Max Drawdown" value={`${results.metrics.maxDrawdown}%`} />
            <Metric label="Win Rate" value={`${results.metrics.winRate}%`} />
          </div>

          {/* Equity Curve */}
          <div className="bg-gray-800 border border-gray-700 rounded-lg p-4">
            <h2 className="text-lg font-semibold mb-2">Equity Curve</h2>
            <div className="h-48 flex items-center justify-center text-gray-400">
              Equity curve chart (mock)
            </div>
          </div>
        </>
      )}
    </div>
    </ClientShell>
  );
}

function Metric({ label, value }: { label: string; value: any }) {
  return (
    <ClientShell>
    <div className="bg-gray-800 border border-gray-700 rounded-lg p-4 text-center">
      <div className="text-sm text-gray-400">{label}</div>
      <div className="text-xl font-semibold mt-1">{value}</div>
    </div>
    </ClientShell>
  );
}


export default BacktestPage

