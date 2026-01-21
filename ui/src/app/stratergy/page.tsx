// import React from 'react'

// function StrategyPage() {
//   return (
//     <div>
//       <h1 className="text-2xl font-semibold mb-2">Strategy Builder</h1>
//       <p className="text-gray-400">
//         Define trading strategies using visual rules.
//       </p>
//     </div>
//   );
// }
// export default StrategyPage



"use client";

import ClientShell from "@/components/ClientShell";
import { useState } from "react";

type EntryRule = {
  indicator: string;
  operator: string;
  value: number;
};

export default function StrategyPage() {
  const [entryRule, setEntryRule] = useState<EntryRule>({
    indicator: "RSI",
    operator: "<",
    value: 30,
  });

  const [exitRule, setExitRule] = useState({
    stopLoss: 5,
    takeProfit: 10,
  });

  const strategyJSON = {
    entry: [entryRule],
    exit: exitRule,
  };

  return (
    <ClientShell>
    <div className="grid grid-cols-3 gap-4">
      {/* ENTRY RULES */}
      <div className="bg-gray-800 border border-gray-700 rounded-lg p-4">
        <h2 className="text-lg font-semibold mb-4">Entry Rule</h2>

        <div className="space-y-3 text-sm">
          <div>
            <label className="block mb-1 text-gray-400">Indicator</label>
            <select
              value={entryRule.indicator}
              onChange={(e) =>
                setEntryRule({ ...entryRule, indicator: e.target.value })
              }
              className="w-full bg-gray-900 border border-gray-600 rounded px-2 py-1"
            >
              <option>RSI</option>
              <option>SMA</option>
              <option>EMA</option>
            </select>
          </div>

          <div>
            <label className="block mb-1 text-gray-400">Condition</label>
            <select
              value={entryRule.operator}
              onChange={(e) =>
                setEntryRule({ ...entryRule, operator: e.target.value })
              }
              className="w-full bg-gray-900 border border-gray-600 rounded px-2 py-1"
            >
              <option value="<">&lt;</option>
              <option value=">">&gt;</option>
            </select>
          </div>

          <div>
            <label className="block mb-1 text-gray-400">Value</label>
            <input
              type="number"
              value={entryRule.value}
              onChange={(e) =>
                setEntryRule({
                  ...entryRule,
                  value: Number(e.target.value),
                })
              }
              className="w-full bg-gray-900 border border-gray-600 rounded px-2 py-1"
            />
          </div>
        </div>
      </div>

      {/* EXIT RULES */}
      <div className="bg-gray-800 border border-gray-700 rounded-lg p-4">
        <h2 className="text-lg font-semibold mb-4">Exit Rules</h2>

        <div className="space-y-3 text-sm">
          <div>
            <label className="block mb-1 text-gray-400">
              Stop Loss (%)
            </label>
            <input
              type="number"
              value={exitRule.stopLoss}
              onChange={(e) =>
                setExitRule({
                  ...exitRule,
                  stopLoss: Number(e.target.value),
                })
              }
              className="w-full bg-gray-900 border border-gray-600 rounded px-2 py-1"
            />
          </div>

          <div>
            <label className="block mb-1 text-gray-400">
              Take Profit (%)
            </label>
            <input
              type="number"
              value={exitRule.takeProfit}
              onChange={(e) =>
                setExitRule({
                  ...exitRule,
                  takeProfit: Number(e.target.value),
                })
              }
              className="w-full bg-gray-900 border border-gray-600 rounded px-2 py-1"
            />
          </div>
        </div>
      </div>

      {/* STRATEGY SUMMARY */}
      <div className="bg-gray-900 border border-gray-700 rounded-lg p-4">
        <h2 className="text-lg font-semibold mb-4">Strategy JSON</h2>

        <pre className="text-xs text-green-400 overflow-auto">
{JSON.stringify(strategyJSON, null, 2)}
        </pre>
      </div>
    </div>
    </ClientShell>
  );
}

