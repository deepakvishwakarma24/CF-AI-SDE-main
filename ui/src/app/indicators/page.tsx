// import React from 'react'

//  function IndicatorsPage() {
//   return (
//     <div>
//       <h1 className="text-2xl font-semibold mb-2">Indicator Lab</h1>
//       <p className="text-gray-400">
//         Build and visualize technical indicators.
//       </p>
//     </div>
//   );
// }

// export default IndicatorsPage



"use client";

import ClientShell from "@/components/ClientShell";
import { useState } from "react";

 function IndicatorsPage() {
  const [selectedIndicator, setSelectedIndicator] = useState<string | null>(
    "RSI"
  );

  return (
    <ClientShell>
    <div className="h-full flex gap-4">
      {/* Left Panel - Indicator Library */}
      <div className="w-64 bg-gray-800 border border-gray-700 rounded-lg p-4">
        <h2 className="text-lg font-semibold mb-4">Indicator Library</h2>

        <ul className="space-y-2 text-sm">
          {["RSI", "MACD", "SMA", "EMA"].map((indicator) => (
            <li
              key={indicator}
              className={`px-3 py-2 rounded cursor-pointer ${
                selectedIndicator === indicator
                  ? "bg-gray-700 text-white"
                  : "text-gray-400 hover:bg-gray-700"
              }`}
              onClick={() => setSelectedIndicator(indicator)}
            >
              {indicator}
            </li>
          ))}
        </ul>
      </div>

      {/* Center Panel - Chart Area */}
      <div className="flex-1 bg-gray-800 border border-gray-700 rounded-lg p-4">
        <h2 className="text-lg font-semibold mb-2">Indicator Preview</h2>
        <p className="text-sm text-gray-400 mb-4">
          Preview how the selected indicator behaves.
        </p>

        <div className="h-64 flex items-center justify-center border border-dashed border-gray-600 rounded">
          <span className="text-gray-500">
            {selectedIndicator} chart preview (mock)
          </span>
        </div>
      </div>

      {/* Right Panel - Settings */}
      <div className="w-72 bg-gray-800 border border-gray-700 rounded-lg p-4">
        <h2 className="text-lg font-semibold mb-4">Settings</h2>

        <div className="space-y-4 text-sm">
          <div>
            <label className="block text-gray-400 mb-1">
              Period
            </label>
            <input
              type="number"
              defaultValue={14}
              className="w-full bg-gray-900 border border-gray-600 rounded px-2 py-1"
            />
          </div>

          <div>
            <label className="block text-gray-400 mb-1">
              Color
            </label>
            <input
              type="color"
              defaultValue="#3b82f6"
              className="w-full h-8"
            />
          </div>
        </div>
      </div>
    </div>
    </ClientShell>
  );
}

export default IndicatorsPage
