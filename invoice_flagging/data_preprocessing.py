
import sqlite3
import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Resolve DB path relative to this file — works from any working directory
_THIS_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = _THIS_DIR.parent / "data" / "inventory.db"


# Load invoice data
def load_invoice_data(db_path: str | Path = DEFAULT_DB_PATH):

    conn = sqlite3.connect(str(db_path))

    query = """
    WITH purchase_agg AS (
        SELECT
            p.PONumber,
            COUNT(DISTINCT p.Brand) AS total_brands,
            SUM(p.Quantity) AS total_item_quantity,
            SUM(p.Dollars) AS total_item_dollars,

            AVG(
                julianday(p.ReceivingDate) - julianday(p.PODate)
            ) AS avg_receiving_delay

        FROM purchases p
        GROUP BY p.PONumber
    )

    SELECT
        vi.PONumber,
        vi.Quantity AS invoice_quantity,
        vi.Dollars AS invoice_dollars,
        vi.Freight,

        (julianday(vi.InvoiceDate) - julianday(vi.PODate))
            AS days_po_to_invoice,

        (julianday(vi.PayDate) - julianday(vi.InvoiceDate))
            AS days_to_pay,

        pa.total_brands,
        pa.total_item_quantity,
        pa.total_item_dollars,
        pa.avg_receiving_delay

    FROM vendor_invoice vi

    LEFT JOIN purchase_agg pa
        ON vi.PONumber = pa.PONumber
    """

    try:
        df = pd.read_sql_query(query, conn)
    finally:
        conn.close()

    return df


# Create invoice risk labels
def create_invoice_risk_label(row):

    invoice_amount = row["invoice_dollars"]
    purchase_amount = row["total_item_dollars"]
    receiving_delay = row["avg_receiving_delay"]

    if (
        pd.isna(invoice_amount)
        or pd.isna(purchase_amount)
        or pd.isna(receiving_delay)
    ):
        return 1

    if abs(invoice_amount - purchase_amount) > 5:
        return 1

    if receiving_delay > 10:
        return 1

    return 0


# Apply labels
def apply_labels(df):

    df = df.copy()

    df["flag_invoice"] = df.apply(
        create_invoice_risk_label,
        axis=1
    )

    return df


# Split data
def split_data(
    df,
    features,
    target,
    test_size=0.2,
    random_state=42
):

    # Select features
    X = df[features].copy()

    # Select target
    y = df[target].copy()

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )


# Scale features
def scale_features(X_train, X_test, scaler_path):

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    # Save scaler
    scaler_path = Path(scaler_path)

    scaler_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(scaler, scaler_path)

    return X_train_scaled, X_test_scaled
