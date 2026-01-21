"use client";

import { createChart } from "lightweight-charts";
import { useEffect, useRef } from "react";

 function LandingCandlestickChart() {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!ref.current) return;

    const chart = createChart(ref.current, {
      height: 220,
      layout: {
        background: { color: "#000000" },
        textColor: "#9ca3af",
      },
      grid: {
        vertLines: { color: "#1f2937" },
        horzLines: { color: "#1f2937" },
      },
      timeScale: { borderColor: "#1f2937" },
      rightPriceScale: { borderColor: "#1f2937" },
      crosshair: { mode: 0 },
    });

    const series = chart.addCandlestickSeries({
      upColor: "#10b981",
      downColor: "#ef4444",
      borderVisible: false,
      wickUpColor: "#10b981",
      wickDownColor: "#ef4444",
    });

    series.setData([
      { time: "2024-01-01", open: 100, high: 108, low: 98, close: 105 },
      { time: "2024-01-02", open: 105, high: 112, low: 102, close: 110 },
      { time: "2024-01-03", open: 110, high: 115, low: 107, close: 109 },
      { time: "2024-01-04", open: 109, high: 113, low: 101, close: 103 },
      { time: "2024-01-05", open: 103, high: 106, low: 97, close: 100 },
    ]);

    chart.timeScale().fitContent();
    return () => chart.remove();
  }, []);

  return <div ref={ref} className="w-full h-full" />;
}

export default LandingCandlestickChart
