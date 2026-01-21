
"use client";
import Link from "next/link";

 function Sidebar() {
  return (
    <aside className="w-64 bg-[#0b1220] text-white p-4">
      <h2 className="text-lg font-semibold mb-6">Quant Lab</h2>

      <nav className="space-y-3">
        <Link href="/market" className="block text-gray-300 hover:text-white">
          Market Data
        </Link>
        <Link href="/indicators" className="block text-gray-300 hover:text-white">
          Indicator Lab
        </Link>
        <Link href="/strategy" className="block text-gray-300 hover:text-white">
          Strategy Builder
        </Link>
        <Link href="/backtest" className="block text-gray-300 hover:text-white">
          Backtesting
        </Link>
        <Link href="/mentor" className="block text-gray-300 hover:text-white">
          Trading Mentor
        </Link>
      </nav>
    </aside>
  );
}

export default Sidebar
