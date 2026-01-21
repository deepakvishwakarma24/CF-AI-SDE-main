// // 


// import { NextResponse } from "next/server";

// const API_KEY = "UK3JG0RS21M59B2D"; // OK to hardcode for now

// const INDICES = [
//   { name: "S&P 500", symbol: "SPX" },
//   { name: "NASDAQ", symbol: "IXIC" },
//   { name: "Dow Jones", symbol: "DJI" },
// ];

// export async function GET() {
//   try {
//     const results = await Promise.all(
//       INDICES.map(async index => {
//         const res = await fetch(
//           `https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=${index.symbol}&apikey=${API_KEY}`,
//           { next: { revalidate: 60 } }
//         );

//         const data = await res.json();
//         const quote = data["Global Quote"];

//         if (!quote) {
//           return {
//             name: index.name,
//             price: null,
//             change: null,
//             percent: null,
//           };
//         }

//         return {
//           name: index.name,
//           price: Number(quote["05. price"]),
//           change: Number(quote["09. change"]),
//           percent: Number(quote["10. change percent"]?.replace("%", "")),
//         };
//       })
//     );

//     return NextResponse.json(results);
//   } catch (err) {
//     return NextResponse.json(
//       { error: "Failed to fetch market indices" },
//       { status: 500 }
//     );
//   }
// }


import { NextResponse } from "next/server";

const API_KEY = "UK3JG0RS21M59B2D"; // hardcode for now (OK for dev)

const INDICES = [
  { name: "S&P 500", symbol: "SPY" },
  { name: "NASDAQ", symbol: "QQQ" },
  { name: "Dow Jones", symbol: "DIA" },
];


export async function GET() {
  try {
    const results = await Promise.all(
      INDICES.map(async index => {
        const res = await fetch(
          `https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=${index.symbol}&apikey=${API_KEY}`
        );

        const data = await res.json();
        const series = data["Time Series (Daily)"];

        if (!series) {
          return {
            name: index.name,
            price: null,
            change: null,
            percent: null,
          };
        }

        const dates = Object.keys(series);
        const latest = series[dates[0]];
        const prev = series[dates[1]];

        const price = parseFloat(latest["4. close"]);
        const prevPrice = parseFloat(prev["4. close"]);
        const change = price - prevPrice;
        const percent = (change / prevPrice) * 100;

        return {
          name: index.name,
          price: price.toFixed(2),
          change: change.toFixed(2),
          percent: percent.toFixed(2),
        };
      })
    );

    return NextResponse.json(results);
  } catch (err) {
    return NextResponse.json(
      { error: "Failed to fetch market indices" },
      { status: 500 }
    );
  }
}

