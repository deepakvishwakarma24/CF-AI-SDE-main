# Database Integration Documentation – CF-AI-SDE Project

---

## 1. Overview of Database Role in the Project

### Why MongoDB Was Introduced

Before the database integration work, the CF-AI-SDE project consisted of three independent components built by different team members, each working in complete isolation:

1. **Data Pipeline (Data-inges-fe/)** — Fetched market data from Yahoo Finance, validated it, cleaned it, and generated technical indicators. The output was only CSV files stored locally. There was no persistent database storage.

2. **AI Agents (AI_Agents/)** — A multi-agent system that performed market analysis including regime detection, volatility forecasting, and risk monitoring. All results existed only in memory and were lost when the process terminated.

3. **User Interface (ui/)** — A Next.js frontend designed for visualization. The API routes existed but returned hardcoded mock data. There was no connection to the actual backend computations.

### What Problem Existed Before Database Integration

The fundamental problem was **component isolation**. Each part of the system worked independently, but there was no way to share data between them:

- The data pipeline could not share processed features with the AI agents.
- The AI agents could not share their analysis results with the UI.
- The UI could not display real analysis — it showed fake data.
- Every time the pipeline ran, all data was recomputed from scratch because nothing was persisted.
- There was no historical data storage for agents to reference past analyses.

MongoDB was introduced to serve as the **central data hub** that connects all three components. It provides persistent storage where the pipeline writes data, agents read and write analysis results, and the API reads those results to serve the UI.

---

## 2. MongoDB Database Design

### Database Name

The database is named **`cf_ai_sde`** (configurable via the `MONGODB_DATABASE` environment variable).

### All Collections Used

Nine collections were created to handle different types of data:

| Collection Name | Purpose |
|-----------------|---------|
| `market_data_raw` | Stores raw OHLCV data directly from Yahoo Finance |
| `market_data_validated` | Stores data after validation checks have passed |
| `market_data_clean` | Stores final cleaned data ready for ML consumption |
| `market_features` | Stores computed technical indicators (SMA, RSI, MACD, etc.) |
| `normalization_params` | Stores mean/std values used for feature normalization |
| `validation_log` | Stores records of validation issues found during processing |
| `agent_outputs` | Stores analysis results from each AI agent |
| `agent_memory` | Stores conversation/context memory for agents |
| `positions_and_risk` | Stores portfolio positions and risk metrics |

### Purpose of Each Collection

**market_data_raw**: The original market data from Yahoo Finance is stored here exactly as received. This preserves the source data for reproducibility and auditing. Each document contains the symbol, timestamp, timeframe, and OHLCV values (open, high, low, close, volume).

**market_data_validated**: After the validation module checks the raw data for issues (missing values, outliers, corrupt records), the records that pass validation are stored here. This separates "checked" data from raw ingestion.

**market_data_clean**: This collection contains the final cleaned data where outliers have been handled and gaps filled. This is what the ML models and agents actually consume for analysis.

**market_features**: Technical indicators like moving averages, RSI, MACD, Bollinger Bands, and other features are expensive to compute. Storing them avoids recalculating them every time an agent needs them. Each record includes version numbers so different feature sets can coexist.

**normalization_params**: When features are normalized (scaled to have zero mean and unit variance), the mean and standard deviation values must be saved. The same normalization must be applied during inference to maintain consistency.

**validation_log**: Every validation issue found (missing data, suspicious values, outliers) is logged here. This creates an audit trail for data quality monitoring and debugging.

**agent_outputs**: This is the most important collection for the UI. When any AI agent completes its analysis, the response is stored here with the agent's name, a unique run ID, and the full response data. The API reads from this collection to serve real data to the frontend.

**agent_memory**: Agents may need to remember past interactions or context. This collection stores memory entries that agents can reference for coherent multi-turn analysis.

**positions_and_risk**: Risk monitoring agents can store and retrieve portfolio positions and associated risk metrics from this collection.

### Why MongoDB Was Suitable

MongoDB was chosen for several reasons:

1. **Schema Flexibility** — Agent responses and feature sets have varying structures. MongoDB's document model handles this naturally without rigid schema migrations.

2. **JSON-Like Documents** — Python dictionaries convert directly to MongoDB documents, making the integration code simple.

3. **Indexing for Time-Series** — MongoDB's compound indexes on (symbol, timestamp, timeframe) enable fast lookups for specific stocks and date ranges.

4. **Easy Deployment** — MongoDB runs locally for development and scales to Atlas for production without code changes.

5. **Python Driver Maturity** — The `pymongo` driver is well-maintained and handles connection pooling automatically.

---

## 3. Database Connection Layer (db/ folder)

The `db/` folder contains three Python files that handle all database interactions:

### Purpose of connection.py

The file `db/connection.py` is responsible for establishing and managing the MongoDB connection. It ensures that only one connection is created and reused throughout the application's lifetime.

### Singleton Pattern Usage

The `MongoDBConnection` class implements the **Singleton design pattern**:

```python
class MongoDBConnection:
    _instance: Optional["MongoDBConnection"] = None
    _client: Optional[MongoClient] = None
    _database: Optional[Database] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
```

The first time `MongoDBConnection()` is called, it creates a new instance and establishes the connection. All subsequent calls return the same instance. This avoids the overhead of repeatedly connecting to MongoDB and ensures connection pooling works correctly.

Module-level convenience functions are provided so other modules do not need to manage the singleton directly:

- `get_connection()` — Returns the singleton instance
- `get_database()` — Returns the database object
- `get_collection(name)` — Returns a specific collection
- `is_mongodb_available()` — Checks if MongoDB is connected

### Environment Variable Handling

The connection reads configuration from environment variables with sensible defaults:

- `MONGODB_URI` — Connection string (default: `mongodb://localhost:27017`)
- `MONGODB_DATABASE` — Database name (default: `cf_ai_sde`)

This allows the same code to run locally (using localhost) and in production (using a cloud MongoDB URI) by simply setting environment variables.

### Index Creation

When the connection is first established, the `_create_indexes()` method runs automatically. It creates compound indexes on all collections to ensure fast query performance. For example:

- `market_data_raw`: Index on (symbol, timestamp, timeframe)
- `agent_outputs`: Index on (agent_name, created_at) and (run_id)
- `market_features`: Index on (symbol, timestamp, timeframe, version)

These indexes are essential for the read queries that filter by symbol and sort by timestamp.

### Safety Checks When MongoDB Is Unavailable

The connection layer implements **graceful degradation**. If MongoDB is not running or the connection fails:

1. The `_connect()` method catches `ConnectionFailure` and `ServerSelectionTimeoutError` exceptions.
2. It sets `_client` and `_database` to `None` and logs a warning.
3. The `is_mongodb_available()` function returns `False`.
4. All writer and reader methods check `is_mongodb_available()` before attempting operations.
5. The pipeline continues to work using CSV files only — nothing crashes.

This design ensures the system remains functional even if MongoDB is temporarily down.

---

## 4. Database Write Flow (writers.py)

The file `db/writers.py` contains classes that handle writing data to MongoDB collections.

### How Data Flows FROM Pipeline → MongoDB

When the data pipeline runs, it processes market data through three stages: ingestion, validation, and feature engineering. At each stage, the corresponding writer class sends data to MongoDB:

1. **Ingestion Stage**: `MarketDataWriter.write_raw()` saves raw OHLCV data to `market_data_raw`
2. **Validation Stage**: `MarketDataWriter.write_validated()` and `MarketDataWriter.write_clean()` save validated and cleaned data
3. **Feature Stage**: `FeatureWriter.write_features()` saves computed indicators to `market_features`

Each write method follows the same pattern:
- Check if MongoDB is available (if not, return False and skip)
- Get the collection reference
- Convert DataFrame rows to MongoDB documents
- Add metadata (timestamps, version numbers)
- Insert documents using `insert_many()`
- Log success or failure

### How Agent Outputs Are Persisted

The `AgentOutputWriter` class handles agent response storage:

```python
AgentOutputWriter.write_output(
    agent_name="RegimeDetectionAgent",
    response_data=response.model_dump(),
    run_id="run_20260201_143052_abc123",
    context={"symbol": "AAPL", "timeframe": "1d"}
)
```

Each agent output document contains:
- `agent_name` — Which agent produced this output
- `run_id` — A unique identifier for this pipeline run
- `created_at` — UTC timestamp when stored
- `response` — The full agent response (nested document)
- `context` — The input context that was given to the agent

### Append-Only Design

All writers use **append-only semantics**. They call `insert_many()` or `insert_one()` rather than `update()` or `replace()`. This means:

- Historical data is never overwritten
- Every pipeline run creates new documents
- The database grows over time with full history
- Queries use timestamps to get the latest records

This design is intentional for auditability — you can always trace back what data existed at any point in time.

### UTC Timestamps

All timestamps are stored in UTC to avoid timezone confusion:

```python
def _get_utc_now() -> datetime:
    return datetime.now(timezone.utc)
```

Every document includes a timestamp field (`ingested_at`, `validated_at`, `created_at`) that records exactly when it was written to MongoDB.

### Error Handling and Graceful Degradation

Every write method is wrapped in try-except blocks:

- `BulkWriteError` — Some documents may have been inserted even if the batch failed. The code logs a warning but returns True (partial success is acceptable).
- `PyMongoError` — General database errors are logged, and the method returns False.
- If `is_mongodb_available()` returns False, the method immediately returns False without attempting any database operation.

This ensures the pipeline never crashes due to database issues. CSV outputs continue to work regardless of MongoDB status.

---

## 5. Database Read Flow (readers.py)

The file `db/readers.py` contains classes that handle reading data from MongoDB collections.

### How AI Agents Read Data

Agents use the `FeatureReader` class to access market data and computed features:

```python
df = FeatureReader.read_features(symbol="AAPL", timeframe="1d", limit=100)
context = FeatureReader.get_returns_and_indicators(symbol="AAPL", timeframe="1d")
```

The `get_returns_and_indicators()` method is particularly useful for agents. It queries the `market_features` collection and extracts key metrics (returns, volatility, momentum, trend indicators) into a dictionary that agents can directly consume.

### How API Reads Agent Outputs

The `AgentOutputReader` class provides methods for the API to retrieve agent results:

- `get_latest_output(agent_name)` — Gets the most recent output for a specific agent
- `get_latest_run_outputs(run_id)` — Gets all outputs from a specific pipeline run
- `get_risk_regime_output()` — Gets formatted risk regime data for the UI
- `get_aggregated_signals()` — Gets signals from the SignalAggregatorAgent

The API route uses `get_risk_regime_output()` which handles multiple fallbacks:
1. First tries to find `RegimeDetectionAgent` output
2. Falls back to `RiskMonitoringAgent` output
3. Falls back to `SignalAggregatorAgent` output
4. If nothing found, returns None (API then uses mock data)

### How Data Is Converted Back to DataFrames / JSON

When reading market data, the reader methods:

1. Query MongoDB with filters (symbol, timeframe, date range)
2. Convert the cursor to a list of dictionaries
3. Create a pandas DataFrame from the list
4. Remove the MongoDB `_id` field (not needed by application code)
5. Convert the `timestamp` field to pandas datetime
6. Set timestamp as the DataFrame index and sort

```python
df = pd.DataFrame(records)
if "_id" in df.columns:
    df = df.drop("_id", axis=1)
if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
```

For agent outputs, the reader returns raw dictionaries (JSON-serializable) since the API needs to send JSON to the frontend.

### How Latest Results Are Fetched

To get the latest result, queries use MongoDB's sort capability:

```python
record = collection.find_one(
    {"agent_name": agent_name},
    sort=[("created_at", DESCENDING)]
)
```

This returns the single most recent document for that agent. The `created_at` field is indexed for fast sorting.

---

## 6. Data Pipeline → MongoDB Integration

### Role of Data-inges-fe/src/integration/

The folder `Data-inges-fe/src/integration/` contains the MongoDB integration layer for the data pipeline.

The file `mongodb_writer.py` provides three wrapper classes:

1. **IngestionDBWriter** — Wraps `MarketDataWriter.write_raw()` and adds summary statistics
2. **ValidationDBWriter** — Wraps `MarketDataWriter.write_validated()`, `write_clean()`, and `ValidationWriter.write_log()`
3. **FeatureDBWriter** — Wraps `FeatureWriter.write_features()` and `write_normalization_params()`

These wrappers accept the exact output format from each pipeline stage and translate it to the format expected by the db writers.

### How CSV Pipeline Was Preserved

The original pipeline code in `Data-inges-fe/main.py` was NOT modified. The MongoDB integration was added through a wrapper file called `pipeline_with_mongodb.py`.

This wrapper:
1. Imports and calls the original `run_full_pipeline()` function
2. Receives the results dictionary containing all pipeline outputs
3. Passes those outputs to the MongoDB writers
4. Returns the combined results

```python
# Run original pipeline (unchanged behavior)
results = run_full_pipeline(symbols=symbols, timeframes=timeframes, save_to_file=save_to_file)

# Add MongoDB writes in parallel
if enable_mongodb:
    mongodb_results["ingestion"] = IngestionDBWriter.write_ingested_data(results["ingestion"])
    mongodb_results["validation"] = ValidationDBWriter.write_validation_results(...)
    mongodb_results["features"] = FeatureDBWriter.write_feature_data(results["features"])
```

### How MongoDB Writes Were Added Without Breaking Existing Code

The key principle is **non-invasive integration**:

- No existing pipeline functions were modified
- MongoDB writes happen AFTER the original pipeline completes
- If MongoDB is unavailable, the wrapper catches the error and continues
- CSV outputs are always generated regardless of MongoDB status
- The wrapper adds `enable_mongodb=True` parameter, defaulting to enabled

This means a teammate who pulls the latest code sees the same pipeline behavior as before. The CSV files still appear in the same locations. MongoDB persistence is an additional benefit, not a breaking change.

---

## 7. AI Agents → MongoDB Integration

### How Agent Logic Was NOT Modified

The agent implementations in `AI_Agents/agents.py` were kept unchanged. Each agent inherits from `BaseAgent` and implements its `analyze()` method. The agent logic (how analysis is performed) was not touched.

### How Persistence Was Added Externally

A new file `AI_Agents/persistence.py` was created to handle database operations. It provides wrapper classes that can be used AROUND agent execution:

**AgentResponsePersistence** — Persists agent responses after execution:

```python
persistence = AgentResponsePersistence()
response = my_agent.analyze(context)
persistence.persist_response(agent_name=my_agent.name, response=response, context=context)
```

**AgentContextAssembler** — Assembles context FROM MongoDB data:

```python
context = AgentContextAssembler.get_market_context(symbol="AAPL", timeframe="1d")
# context now contains returns, volatility, momentum, trend data from market_features
```

**PersistentAgentOrchestrator** — Wraps the entire orchestration with persistence:

```python
orchestrator = PersistentAgentOrchestrator()
results = orchestrator.execute_with_persistence(agent_orchestrator, initial_context)
```

### Role of run_id

Every pipeline execution generates a unique `run_id`:

```python
def _generate_run_id() -> str:
    return f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
```

This produces IDs like `run_20260201_143052_a1b2c3d4`. The run_id:

- Groups all agent outputs from the same pipeline run
- Allows querying "all outputs from run X"
- Provides traceability for debugging
- Is stored with every agent output document

### How Agent Outputs Are Stored and Later Reused

When an agent response is persisted:

1. The `AgentResponse` Pydantic model is converted to a dictionary using `model_dump()`
2. The dictionary is wrapped with metadata (agent_name, run_id, created_at, context)
3. The document is inserted into `agent_outputs` collection
4. The response is also kept in memory (in case MongoDB is unavailable)

Later, the API or other agents can retrieve outputs:

```python
latest = AgentOutputReader.get_latest_output("RegimeDetectionAgent")
all_from_run = AgentOutputReader.get_latest_run_outputs(run_id="run_20260201_143052_a1b2c3d4")
```

---

## 8. API → MongoDB → UI Flow (High Level)

### How Backend API Reads MongoDB

The Next.js API routes in `ui/src/app/api/` connect directly to MongoDB. The file `ui/src/app/api/risk-regime/route.ts` demonstrates the pattern:

1. The route creates a MongoDB client using the same connection string
2. It queries the `agent_outputs` collection for the latest regime data
3. It transforms the agent response format to match what the UI expects
4. It returns JSON to the frontend

```typescript
const client = await getMongoClient();
const db = client.db(MONGODB_DATABASE);
const collection = db.collection("agent_outputs");

let record = await collection.findOne(
    { agent_name: "RegimeDetectionAgent" },
    { sort: { created_at: -1 } }
);
```

### How UI Receives Real Data

The UI components call the API routes using standard fetch/axios. When the API successfully reads from MongoDB:

- The response includes `source: "mongodb"` to indicate real data
- The response includes `agent_name` and `run_id` for traceability
- The data is formatted to match UI component expectations

### READ-ONLY Flow

The UI only READS from MongoDB through the API. It never writes. The data flow is strictly:

```
Pipeline → writes → MongoDB → reads → API → JSON → UI
```

This separation ensures the UI cannot accidentally corrupt the data. The UI is a pure consumer of the analysis results.

---

## 9. End-to-End Execution Flow

### What Happens When `python run_e2e_pipeline.py` Is Executed

The file `run_e2e_pipeline.py` orchestrates the entire flow:

**STEP 1: Check MongoDB Connection**
```python
check_mongodb_connection()
```
- Imports `get_connection` and `is_mongodb_available` from db/connection.py
- Calls `is_mongodb_available()` to test if MongoDB is running
- Logs success or warning (pipeline continues either way)

**STEP 2: Run Data Pipeline**
```python
run_data_pipeline(symbols=["AAPL"], timeframes=["1d"])
```
- Imports `run_pipeline_with_mongodb` from Data-inges-fe/pipeline_with_mongodb.py
- This function calls the original `run_full_pipeline()` internally
- Ingestion: Fetches OHLCV data from Yahoo Finance → CSV + MongoDB (`market_data_raw`)
- Validation: Checks for issues, cleans data → CSV + MongoDB (`market_data_validated`, `market_data_clean`, `validation_log`)
- Features: Computes technical indicators → CSV + MongoDB (`market_features`, `normalization_params`)

**STEP 3: Run AI Agent Analysis**
```python
run_agent_analysis(symbol="AAPL")
```
- Creates an `AgentResponsePersistence` instance with a new run_id
- Calls `AgentContextAssembler.get_market_context()` to load feature data from MongoDB
- Runs `MarketDataAgent.analyze()` with the context
- Persists the response using `persistence.persist_response()` → MongoDB (`agent_outputs`)
- Runs `RegimeDetectionAgent.analyze()` with the market agent's output
- Persists the regime response → MongoDB (`agent_outputs`)
- Returns in-memory outputs for verification

**STEP 4: Verify API Data Availability**
```python
verify_api_data()
```
- Imports `AgentOutputReader` from db/readers.py
- Calls `get_risk_regime_output()` to verify the API can read agent results
- Logs success if data is found

**The Complete Data Flow:**
```
Yahoo Finance API
       ↓
   Ingestion
       ↓
  market_data_raw (MongoDB) + raw CSV
       ↓
   Validation
       ↓
  market_data_validated + market_data_clean (MongoDB) + validated CSV
       ↓
Feature Engineering
       ↓
  market_features (MongoDB) + features CSV
       ↓
  AI Agent Analysis (reads features from MongoDB)
       ↓
  agent_outputs (MongoDB)
       ↓
  API Route (reads agent_outputs)
       ↓
  UI displays real data
```

---

## 10. Key Engineering Decisions

### Singleton Connection

The singleton pattern for database connections was chosen because:
- Only one connection pool is needed per process
- Avoids the overhead of reconnecting for every operation
- pymongo's connection pooling works best with a single client instance
- Simplifies the API (just call `get_collection()` anywhere)

### Parallel CSV + MongoDB

The decision to keep CSV outputs alongside MongoDB was intentional:
- CSV files provide a backup if MongoDB is unavailable
- Teammates who don't have MongoDB can still run the pipeline
- CSV outputs are easy to inspect and debug
- No breaking changes to existing workflows

### Wrapper-Based Integration

Instead of modifying existing code, wrapper modules were created:
- `pipeline_with_mongodb.py` wraps the original pipeline
- `persistence.py` wraps agent execution
- This approach minimizes merge conflicts with teammate code
- Makes it easy to disable MongoDB by changing one flag

### Isolation from Teammate Code

The database integration was designed to avoid touching files written by other team members:
- Agent logic in `agents.py` was not modified
- Pipeline logic in `main.py` was not modified
- UI components were not modified
- Only new files were added or existing integration points were used

---

## 11. Summary (What Works Now)

### What Was Impossible Before

Before database integration:
- Pipeline results were only in CSV files (no central storage)
- Agent outputs were lost after execution (no persistence)
- UI showed mock/fake data (no real backend connection)
- Re-running analysis meant re-computing everything (no caching)
- No way to query historical data (no time-series storage)
- Components could not share data (complete isolation)

### What Works After Database Integration

After database integration:
- All market data is persisted in MongoDB with full history
- All agent outputs are stored and queryable by agent name or run ID
- The UI displays real analysis results from the agents
- The API reads actual data from MongoDB (with mock fallback)
- Historical queries are fast due to proper indexing
- The pipeline is idempotent — results can be retrieved without re-running
- Components communicate through the database as a shared data layer
- Graceful degradation ensures nothing crashes if MongoDB is down
- Full audit trail of every data transformation is preserved

The database layer now serves as the **central nervous system** of the CF-AI-SDE platform, connecting data ingestion, AI analysis, and user interface into a cohesive, production-ready application.
