from pathlib import Path
import pandas as pd

# RUTAS
DATA = Path("data")
ORDERS_CSV = DATA / "orders_full.csv"
OUT_CSV = DATA / "dataset_churn.csv"


# 1) CARGA DE PEDIDOS
orders = pd.read_csv(ORDERS_CSV)

# Tipos de datos mínimos
orders["order_timestamp"] = pd.to_datetime(orders["order_timestamp"], errors="coerce")

# Si no existe order_price, lo creo en 0; si existe, lo convierto a número.
if "order_price" in orders.columns:
    orders["order_price"] = pd.to_numeric(orders["order_price"], errors="coerce").fillna(0.0)
else:
    orders["order_price"] = 0.0


# 2) DEFINIR VENTANAS H (historia) y P (predicción)
#    H = 180 días antes del corte; P = 90 días después del corte

t0 = orders["order_timestamp"].max().normalize()  # fecha de corte
H_start = t0 - pd.Timedelta(days=179)            # ventana historia 180d (incluye t0)
H_end   = t0
P_start = t0 + pd.Timedelta(days=1)
P_end   = P_start + pd.Timedelta(days=90)

print(f"H: {H_start.date()} → {H_end.date()} | P: {P_start.date()} → {P_end.date()}")


# 3) MÉTRICAS RFM EN HISTORIA (H)
h = orders[(orders["order_timestamp"] >= H_start) & (orders["order_timestamp"] <= H_end)].copy()

rfm = (
    h.groupby("client_id", as_index=False)
     .agg(
         freq_180d=("order_id", "count"),              # cuántas compras en 180d
         monetary_180d=("order_price", "sum"),         # gasto total 180d
         last_order=("order_timestamp", "max"),        # última compra
     )
)

# Recency: días desde la última compra hasta t0
rfm["last_order"] = pd.to_datetime(rfm["last_order"], errors="coerce")
rfm["recency_days"] = (t0 - rfm["last_order"]).dt.days
# Si alguien no tiene 'last_order' (NaT), le pongo un número grande para considerarlo "muy inactivo"
rfm["recency_days"] = rfm["recency_days"].fillna(10_000).astype("Int64")
rfm = rfm.drop(columns=["last_order"])

# inactividad en los últimos 90 días (respecto a t0)
rfm["inactive_90d"] = (rfm["recency_days"] > 90).astype("Int64")


# 4) TARGET PRINCIPAL: churn_label
#    1 = churn (NO compra en P), 0 = no churn (sí compra en P)
p = orders[(orders["order_timestamp"] >= P_start) & (orders["order_timestamp"] <= P_end)].copy()
compran_p = p[["client_id"]].drop_duplicates().assign(buy_in_P=1)

base = rfm.merge(compran_p, on="client_id", how="left")
base["churn_label"] = base["buy_in_P"].fillna(0).apply(lambda x: 0 if x == 1 else 1)
base = base.drop(columns=["buy_in_P"])


# 5) GUARDAR Y RESUMEN
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
base.to_csv(OUT_CSV, index=False)

print("✅ dataset listo →", OUT_CSV)
print("shape:", base.shape)

# balances del target y del flag auxiliar
print("balance churn_label:", base["churn_label"].value_counts(dropna=False).to_dict())
print("balance inactive_90d:", base["inactive_90d"].value_counts(dropna=False).to_dict())
