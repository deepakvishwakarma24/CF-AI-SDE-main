import { NextResponse } from "next/server";

export async function GET() {
  try {
    const res = await fetch(
      "https://api.coingecko.com/api/v3/coins/markets" +
        "?vs_currency=usd" +
        "&order=market_cap_desc" +
        "&per_page=10" +
        "&page=1" +
        "&sparkline=false" +
        "&price_change_percentage=24h",
      { next: { revalidate: 60 } } // refresh every minute
    );

    if (!res.ok) {
      throw new Error("Failed to fetch crypto data");
    }

    const data = await res.json();

    const formatted = data.map((coin: any) => ({
      id: coin.id,
      name: coin.name,
      symbol: coin.symbol.toUpperCase(),
      price: coin.current_price,
      change: coin.price_change_percentage_24h,
      image: coin.image,
      marketCapRank: coin.market_cap_rank,
    }));

    return NextResponse.json(formatted);
  } catch (err) {
    return NextResponse.json(
      { error: "Crypto market fetch failed" },
      { status: 500 }
    );
  }
}
