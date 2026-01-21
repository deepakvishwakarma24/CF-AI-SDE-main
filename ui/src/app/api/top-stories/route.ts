import { NextResponse } from "next/server";

const NEWS_API_KEY = "d4e5521d7f254bc3a06e2e56d3edcb45"; // ⚠️ TEMP ONLY

export async function GET() {
  const res = await fetch(
    `https://newsapi.org/v2/top-headlines?category=business&language=en&pageSize=6&apiKey=${NEWS_API_KEY}`,
    { next: { revalidate: 300 } }
  );

  if (!res.ok) {
    return NextResponse.json(
      { error: "Failed to fetch news" },
      { status: 500 }
    );
  }

  const data = await res.json();
  return NextResponse.json(data.articles);
}
