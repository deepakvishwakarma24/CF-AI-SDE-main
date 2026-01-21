import { NextResponse } from "next/server";

/**
 * Risk Regime API Route
 * Reads latest regime data from MongoDB agent_outputs collection.
 * Falls back to mock data if DB unavailable.
 */

// MongoDB connection for API route
import { MongoClient } from "mongodb";

const MONGODB_URI = process.env.MONGODB_URI || "mongodb://localhost:27017";
const MONGODB_DATABASE = process.env.MONGODB_DATABASE || "cf_ai_sde";

// Connection cache for serverless functions
let cachedClient: MongoClient | null = null;

async function getMongoClient(): Promise<MongoClient | null> {
  try {
    if (cachedClient) {
      return cachedClient;
    }

    const client = new MongoClient(MONGODB_URI, {
      connectTimeoutMS: 5000,
      serverSelectionTimeoutMS: 5000,
    });

    await client.connect();
    cachedClient = client;
    return client;
  } catch (error) {
    console.error("MongoDB connection failed:", error);
    return null;
  }
}

// Mock data fallback (matches current UI expectations)
const MOCK_REGIME_DATA = {
  regime: "RISK_ON" as const,
  confidence: 72,
  trend: "Improving",
  drivers: [
    "Equity momentum positive",
    "Volatility declining",
    "Crypto beta outperforming",
  ],
  timestamp: new Date().toISOString(),
  source: "mock",
};

type RegimeType = "RISK_ON" | "NEUTRAL" | "RISK_OFF";

interface RiskRegimeResponse {
  regime: RegimeType;
  confidence: number;
  trend: string;
  drivers: string[];
  timestamp: string;
  source: string;
  agent_name?: string;
  run_id?: string;
}

function mapRecommendationToRegime(recommendation: string): RegimeType {
  const rec = recommendation?.toUpperCase() || "";

  if (
    rec.includes("RISK_ON") ||
    rec.includes("BUY") ||
    rec.includes("BULLISH")
  ) {
    return "RISK_ON";
  }
  if (
    rec.includes("RISK_OFF") ||
    rec.includes("SELL") ||
    rec.includes("REDUCE") ||
    rec.includes("BEARISH")
  ) {
    return "RISK_OFF";
  }
  return "NEUTRAL";
}

function extractDrivers(data: Record<string, unknown>): string[] {
  const drivers: string[] = [];

  // Try to extract meaningful drivers from agent response
  const response = data.response as Record<string, unknown> | undefined;
  if (!response) return drivers;

  const structuredData = response.structured_data as
    | Record<string, unknown>
    | undefined;
  const summary = response.summary as string | undefined;

  // Extract from structured data
  if (structuredData) {
    if (structuredData.current_regime) {
      drivers.push(`Regime: ${structuredData.current_regime}`);
    }
    if (structuredData.regime_probability !== undefined) {
      drivers.push(
        `Regime probability: ${(Number(structuredData.regime_probability) * 100).toFixed(1)}%`,
      );
    }
    if (structuredData.volatility_forecast) {
      drivers.push(
        `Volatility forecast: ${structuredData.volatility_forecast}`,
      );
    }
    if (structuredData.var !== undefined) {
      drivers.push(`VaR: ${(Number(structuredData.var) * 100).toFixed(2)}%`);
    }
    if (structuredData.drawdown !== undefined) {
      drivers.push(
        `Drawdown: ${(Number(structuredData.drawdown) * 100).toFixed(2)}%`,
      );
    }
  }

  // Extract from summary if no structured data drivers
  if (drivers.length === 0 && summary) {
    // Split summary into bullet points
    const sentences = summary.split(/[.!]/).filter((s) => s.trim().length > 10);
    drivers.push(...sentences.slice(0, 3).map((s) => s.trim()));
  }

  // Fallback drivers
  if (drivers.length === 0) {
    drivers.push("Agent analysis complete");
  }

  return drivers.slice(0, 5);
}

function determineTrend(data: Record<string, unknown>): string {
  const response = data.response as Record<string, unknown> | undefined;
  if (!response) return "Stable";

  const structuredData = response.structured_data as
    | Record<string, unknown>
    | undefined;
  const recommendation = response.recommendation as string | undefined;

  if (structuredData?.trend) {
    return String(structuredData.trend);
  }

  if (recommendation) {
    const rec = recommendation.toUpperCase();
    if (rec.includes("BUY") || rec.includes("BULLISH")) return "Improving";
    if (rec.includes("SELL") || rec.includes("BEARISH")) return "Deteriorating";
  }

  return "Stable";
}

export async function GET() {
  try {
    const client = await getMongoClient();

    if (!client) {
      // MongoDB unavailable - return mock data
      console.log("MongoDB unavailable, returning mock data");
      return NextResponse.json(MOCK_REGIME_DATA);
    }

    const db = client.db(MONGODB_DATABASE);
    const collection = db.collection("agent_outputs");

    // Try to get RegimeDetectionAgent output first
    let record = await collection.findOne(
      { agent_name: "RegimeDetectionAgent" },
      { sort: { created_at: -1 } },
    );

    // Fallback to RiskMonitoringAgent
    if (!record) {
      record = await collection.findOne(
        { agent_name: "RiskMonitoringAgent" },
        { sort: { created_at: -1 } },
      );
    }

    // Fallback to SignalAggregatorAgent
    if (!record) {
      record = await collection.findOne(
        { agent_name: "SignalAggregatorAgent" },
        { sort: { created_at: -1 } },
      );
    }

    // If no agent data found, return mock
    if (!record) {
      console.log("No agent outputs found, returning mock data");
      return NextResponse.json(MOCK_REGIME_DATA);
    }

    // Transform agent output to UI format
    const response = record.response as Record<string, unknown> | undefined;
    const recommendation = (response?.recommendation as string) || "NEUTRAL";
    const confidenceScore = (response?.confidence_score as number) || 0.5;

    const regimeData: RiskRegimeResponse = {
      regime: mapRecommendationToRegime(recommendation),
      confidence: Math.round(confidenceScore * 100),
      trend: determineTrend(record),
      drivers: extractDrivers(record),
      timestamp: record.created_at?.toISOString() || new Date().toISOString(),
      source: "mongodb",
      agent_name: record.agent_name,
      run_id: record.run_id,
    };

    return NextResponse.json(regimeData);
  } catch (error) {
    console.error("Error fetching risk regime:", error);
    // Return mock data on error
    return NextResponse.json(MOCK_REGIME_DATA);
  }
}
