#make_target.py
from pathlib import Path
import pandas as pd

DATA = Path("data")
ORDERS_CSV = DATA / "orders_full.csv"
OUT_CSV = DATA / "dataset_churn.csv"

# 1) Cargar pedidos
orders = pd.read_csv(ORDERS_CSV)

# Tipos
orders["order_timestamp"] = pd.to_datetime(orders["order_timestamp"], errors="coerce")
orders["order_price"] = pd.to_numeric(orders.get("order_price", 0), errors="coerce")

# 2) Definir ventana de historia (H) y predicción (P)
t0 = orders["order_timestamp"].max().normalize()        # fecha de corte (último día con datos)
H_start = t0 - pd.Timedelta(days=179)                   # 180 días de historia
H_end   = t0                                            # hasta el corte
P_start = t0 + pd.Timedelta(days=1)                     # día siguiente al corte
P_end   = P_start + pd.Timedelta(days=90)               # 90 días de predicción

print(f"H: {H_start.date()} → {H_end.date()} | P: {P_start.date()} → {P_end.date()}")

# 3) Métricas RFM en la historia (H)
h = orders[(orders["order_timestamp"] >= H_start) & (orders["order_timestamp"] <= H_end)].copy()

rfm = (
    h.groupby("client_id", as_index=False)
     .agg(
         freq_180d=("order_id", "count"),
         monetary_180d=("order_price", "sum"),
         last_order=("order_timestamp", "max"),
     )
)
rfm["last_order"] = pd.to_datetime(rfm["last_order"], errors="coerce")
rfm["recency_days"] = (t0 - rfm["last_order"]).dt.days
rfm["recency_days"] = rfm["recency_days"].fillna(10_000).astype("Int64")
rfm = rfm.drop(columns=["last_order"])


# 4) Target principal: churn_label
#    1 = churn (NO compra en P), 0 = no churn (sí compra en P)
p = orders[(orders["order_timestamp"] >= P_start) & (orders["order_timestamp"] <= P_end)].copy()
compran_p = p[["client_id"]].drop_duplicates().assign(buy_in_P=1)

base = rfm.merge(compran_p, on="client_id", how="left")
base["churn_label"] = base["buy_in_P"].fillna(0).apply(lambda x: 0 if x == 1 else 1)
base = base.drop(columns=["buy_in_P"])


# 5) Guardar dataset final
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
base.to_csv(OUT_CSV, index=False)

print("✅ dataset listo →", OUT_CSV)
print("shape:", base.shape)
print("balance churn:", base["churn_label"].value_counts(dropna=False).to_dict())
print("balance inactive_90d:", base["inactive_90d"].value_counts(dropna=False).to_dict())
print("balance downgrade_label:", base["downgrade_label"].value_counts(dropna=False).to_dict())

print("balance inactive_90d:", base["inactive_90d"].value_counts(dropna=False).to_dict())
print("balance downgrade_label:", base["downgrade_label"].value_counts(dropna=False).to_dict())
