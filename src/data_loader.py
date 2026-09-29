"""Vireo Support Intelligence — Data Loader.

Loads all CSV files with consistent types and basic validation.
"""

import pandas as pd
import os
from . import config


def _find_data_dir():
    """Find the data directory — either data/ subdir or project root."""
    if os.path.isdir(config.DATA_DIR) and os.path.exists(os.path.join(config.DATA_DIR, "tickets.csv")):
        return config.DATA_DIR
    # Fallback: files at project root
    root = os.path.dirname(os.path.dirname(__file__))
    if os.path.exists(os.path.join(root, "tickets.csv")):
        return root
    raise FileNotFoundError("Cannot find tickets.csv in data/ or project root")


def load_tickets() -> pd.DataFrame:
    """Load tickets.csv with appropriate dtypes."""
    data_dir = _find_data_dir()
    df = pd.read_csv(
        os.path.join(data_dir, "tickets.csv"),
        dtype={
            "ticket_id": str,
            "customer_id": str,
            "order_id": str,
            "product_sku": str,
            "agent_id": str,
            "category": str,
            "channel": str,
            "status": str,
            "priority": str,
            "assigned_team": str,
            "replacement_issued": str,
            "refund_reason_code": str,
            "customer_message": str,
            "agent_notes": str,
            "source_system": str,
        },
        parse_dates=["created_at", "first_response_at", "resolved_at"],
    )
    # Normalise NaN for string columns that might have empty strings
    for col in ["order_id", "refund_reason_code"]:
        df[col] = df[col].replace("", pd.NA)
    return df


def load_agents() -> pd.DataFrame:
    """Load agents.csv with date parsing."""
    data_dir = _find_data_dir()
    df = pd.read_csv(
        os.path.join(data_dir, "agents.csv"),
        dtype={"agent_id": str, "name": str, "site": str, "team": str, "shift": str, "tier": int},
        parse_dates=["from_date"],
    )
    df["to_date"] = pd.to_datetime(df["to_date"], errors="coerce")
    return df


def load_orders() -> pd.DataFrame:
    """Load orders.csv."""
    data_dir = _find_data_dir()
    df = pd.read_csv(
        os.path.join(data_dir, "orders.csv"),
        dtype={
            "order_id": str,
            "customer_id": str,
            "sku": str,
            "channel": str,
            "lot_code": str,
        },
        parse_dates=["order_date"],
    )
    return df


def load_customers() -> pd.DataFrame:
    """Load customers.csv."""
    data_dir = _find_data_dir()
    df = pd.read_csv(
        os.path.join(data_dir, "customers.csv"),
        dtype={"customer_id": str, "name": str, "city": str, "state": str, "care_plus": str},
        parse_dates=["signup_date"],
    )
    return df


def load_products() -> pd.DataFrame:
    """Load products.csv."""
    data_dir = _find_data_dir()
    df = pd.read_csv(
        os.path.join(data_dir, "products.csv"),
        dtype={"sku": str, "product_name": str, "family": str},
        parse_dates=["launch_date"],
    )
    return df


def load_all():
    """Load all datasets and return as a dict."""
    return {
        "tickets": load_tickets(),
        "agents": load_agents(),
        "orders": load_orders(),
        "customers": load_customers(),
        "products": load_products(),
    }
