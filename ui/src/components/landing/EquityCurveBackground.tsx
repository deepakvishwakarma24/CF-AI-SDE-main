// "use client";

// import { useEffect, useRef } from "react";
// import { createChart, LineStyle } from "lightweight-charts";

//  function EquityCurveBackground() {
//   const chartRef = useRef<HTMLDivElement>(null);

//   useEffect(() => {
//     if (!chartRef.current) return;

//     const chart = createChart(chartRef.current, {
//       width: chartRef.current.clientWidth,
//       height: chartRef.current.clientHeight,
//       layout: {
//         background: { color: "transparent" },
//         textColor: "transparent",
//       },
//       grid: {
//         vertLines: { color: "transparent" },
//         horzLines: { color: "transparent" },
//       },
//       timeScale: {
//         visible: false,
//         borderVisible: false,
//       },
//       rightPriceScale: {
//         visible: false,
//         borderVisible: false,
//       },
//       crosshair: {
//         vertLine: { visible: false },
//         horzLine: { visible: false },
//       },
//       handleScroll: false,
//       handleScale: false,
//     });

//     const lineSeries = chart.addLineSeries({
//       color: "rgba(16, 185, 129, 0.35)", // emerald-500-ish
//       lineWidth: 2,
//       lineStyle: LineStyle.Solid,
//     });

//     // Mock equity curve data (smooth & realistic)
//     lineSeries.setData([
//   { time: "2020-01-01", value: 100 },
//   { time: "2020-02-01", value: 104 },
//   { time: "2020-03-01", value: 96 },
//   { time: "2020-04-01", value: 108 },
//   { time: "2020-05-01", value: 115 },
//   { time: "2020-06-01", value: 122 },
//   { time: "2020-07-01", value: 118 },
//   { time: "2020-08-01", value: 130 },
//   { time: "2020-09-01", value: 138 },
// ]);

//     chart.timeScale().fitContent();

//     const handleResize = () => {
//       chart.applyOptions({
//         width: chartRef.current!.clientWidth,
//         height: chartRef.current!.clientHeight,
//       });
//     };

//     window.addEventListener("resize", handleResize);

//     return () => {
//       window.removeEventListener("resize", handleResize);
//       chart.remove();
//     };
//   }, []);

//   return (
//     // <div className="absolute inset-0 z-0 opacity-[0.07] pointer-events-none">
//     //   <div ref={chartRef} className="w-full h-full" />
//     // </div>


//     <div
//     ref={chartRef}
//     className="absolute inset-0 z-[1] pointer-events-none opacity-20"
//   />
//   );
// }



// export default EquityCurveBackground


"use client";

import { useEffect, useRef } from "react";
import { createChart } from "lightweight-charts";

function EquityCurveBackground() {
  const chartRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!chartRef.current) return;

    const width = chartRef.current.clientWidth;
    const height = chartRef.current.clientHeight;

    if (!width || !height) return;

    const chart = createChart(chartRef.current, {
      width,
      height,
      layout: {
        background: { color: "transparent" },
        textColor: "transparent",
      },
      grid: {
        vertLines: { color: "transparent" },
        horzLines: { color: "transparent" },
      },
      timeScale: {
        visible: false,
        borderVisible: false,
      },
      rightPriceScale: {
        visible: false,
        borderVisible: false,
      },
      crosshair: {
        vertLine: { visible: false },
        horzLine: { visible: false },
      },
      handleScroll: false,
      handleScale: false,
    });

    const lineSeries = chart.addLineSeries({
      color: "rgba(16, 185, 129, 0.45)", // emerald glow
        lineWidth: 2,
      priceLineVisible: false
    });

    lineSeries.setData([
      { time: "2020-01-01", value: 100 },
      { time: "2020-02-01", value: 104 },
      { time: "2020-03-01", value: 96 },
      { time: "2020-04-01", value: 108 },
      { time: "2020-05-01", value: 115 },
      { time: "2020-06-01", value: 122 },
      { time: "2020-07-01", value: 118 },
      { time: "2020-08-01", value: 130 },
      { time: "2020-09-01", value: 138 },
    ]);

    chart.timeScale().fitContent();

    const resize = () => {
      if (!chartRef.current) return;
      chart.applyOptions({
        width: chartRef.current.clientWidth,
        height: chartRef.current.clientHeight,
      });
    };

    window.addEventListener("resize", resize);

    return () => {
      window.removeEventListener("resize", resize);
      chart.remove();
    };
  }, []);

  return (
    <div
      ref={chartRef}
      className="absolute inset-0 opacity-70 pointer-events-none"
    />
  );
}

export default EquityCurveBackground;
