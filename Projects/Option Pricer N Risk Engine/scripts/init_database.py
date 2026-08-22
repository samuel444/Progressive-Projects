
from options_risk_engine.config import DATABASE_PATH, SCHEMA_PATH, ensure_project_directories
from options_risk_engine.database import SQLiteStore

ensure_project_directories()
SQLiteStore(DATABASE_PATH, SCHEMA_PATH).initialise()
print(f"Initialised {DATABASE_PATH}")
