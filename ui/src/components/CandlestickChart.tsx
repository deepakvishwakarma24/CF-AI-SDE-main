// "use client";
// // import React from 'react'
// import { createChart } from "lightweight-charts";
// import { useEffect, useRef } from "react";
// import  mockCandles ,{mockSMA}  from "@/data/marketData";

// type Props = {
//   showSMA: boolean;
// };


//  function CandlestickChart() {
//   const chartContainerRef = useRef<HTMLDivElement>(null);

//   // eslint-disable-next-line react-hooks/exhaustive-deps

//   useEffect(() => {
//     if (!chartContainerRef.current) return;

//     const chart = createChart(chartContainerRef.current, {
//       width: chartContainerRef.current.clientWidth,
//       height: 400,
//       layout: {
//         background: { color: "#111827" },
//         textColor: "#d1d5db",
//       },
//       grid: {
//         vertLines: { color: "#1f2937" },
//         horzLines: { color: "#1f2937" },
//       },
//     });

//     const candleSeries = chart.addCandlestickSeries({
//       upColor: "#22c55e",
//       downColor: "#ef4444",
//       borderVisible: false,
//       wickUpColor: "#22c55e",
//       wickDownColor: "#ef4444",
//     });

//     candleSeries.setData(mockCandles);

//     chart.timeScale().fitContent();

//     return () => chart.remove();
//   }, []);

//   return <div ref={chartContainerRef} className="w-full" />;
// }


// export default CandlestickChart


"use client";

import { createChart } from "lightweight-charts";
import { useEffect, useRef } from "react";
import {candles, sma } from "@/data/marketData";

type Props = {
  showSMA: boolean;
};

 function CandlestickChart({ showSMA }: Props) {
  const chartContainerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!chartContainerRef.current) return;

    const chart = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height: 400,
      layout: {
        background: { color: "#111827" },
        textColor: "#d1d5db",
      },
      grid: {
        vertLines: { color: "#1f2937" },
        horzLines: { color: "#1f2937" },
      },
    });

    const candleSeries = chart.addCandlestickSeries({
      upColor: "#22c55e",
      downColor: "#ef4444",
      borderVisible: false,
      wickUpColor: "#22c55e",
      wickDownColor: "#ef4444",
    });

    candleSeries.setData(candles);

    let smaSeries: any = null;

    if (showSMA) {
      smaSeries = chart.addLineSeries({
        color: "#3b82f6",
        lineWidth: 2,
      });
      smaSeries.setData(sma);
    }

    chart.timeScale().fitContent();

    return () => chart.remove();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [showSMA]);

  return <div ref={chartContainerRef} className="w-full" />;
}


export default CandlestickChart
