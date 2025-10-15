import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy import stats
import statsmodels.stats.api as sms

st.set_page_config(page_title="Marketing Campaign Optimization", page_icon="📈", layout="wide")
st.title("📈 Marketing Campaign Optimization (Segmentation + A/B Testing)")

# ------------------------- Utilities -------------------------
def robust_read_transactions(upload):
    """
    Accepts CSV or Excel.
    Tries to produce a uniform tx table with columns: customer_id, invoice_date (datetime), amount (float).
    Handles common Online Retail II variants (Quantity, UnitPrice) or direct amount field.
    """
    if upload is None:
        return None, "No file provided."

    # Load
    if upload.name.lower().endswith((".xlsx", ".xls")):
        # Try both sheets if present
        try:
            df1 = pd.read_excel(upload, sheet_name=0, dtype={"Invoice": str, "InvoiceNo": str})
            try:
                df2 = pd.read_excel(upload, sheet_name=1, dtype={"Invoice": str, "InvoiceNo": str})
                raw = pd.concat([df1, df2], ignore_index=True)
            except Exception:
                raw = df1
        except Exception as e:
            return None, f"Excel read error: {e}"
    else:
        try:
            raw = pd.read_csv(upload)
        except Exception as e:
            return None, f"CSV read error: {e}"

    # Normalize columns
    raw.columns = (
        raw.columns
           .str.strip()
           .str.lower()
           .str.replace(r"[ /]", "_", regex=True)
    )

    # Try to map common names
    rename_map = {
        "customerid": "customer_id",
        "customer_id": "customer_id",
        "customer_id_": "customer_id",
        "invoice_date": "invoice_date",
        "invoicedate": "invoice_date",
        "invoice": "invoice_no",
        "invoiceno": "invoice_no",
        "unitprice": "unit_price",
        "price": "unit_price",
        "quantity": "quantity",
        "amount": "amount",
    }
    raw = raw.rename(columns={c: rename_map.get(c, c) for c in raw.columns})

    # Path A: already have amount
    if {"customer_id","invoice_date","amount"}.issubset(raw.columns):
        df = raw[["customer_id","invoice_date","amount"]].copy()
        df["invoice_date"] = pd.to_datetime(df["invoice_date"], errors="coerce")
        df = df.dropna(subset=["customer_id","invoice_date","amount"])
        df["customer_id"] = df["customer_id"].astype(str)
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
        df = df[df["amount"] > 0]
        df = df.sort_values(["customer_id","invoice_date"]).reset_index(drop=True)
        return df, None

    # Path B: compute amount = quantity * unit_price (e.g., UCI Online Retail II)
    required = {"customer_id","invoice_date","quantity","unit_price"}
    if required.issubset(raw.columns):
        df = raw.copy()
        # Remove credit notes if we have invoice_no
        if "invoice_no" in df.columns:
            bad = df["invoice_no"].astype(str).str.startswith("C", na=False)
            df = df[~bad]
        df["invoice_date"] = pd.to_datetime(df["invoice_date"], errors="coerce")
        df = df.dropna(subset=["customer_id","invoice_date","quantity","unit_price"])
        df["customer_id"] = df["customer_id"].astype(str)
        df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
        df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce")
        df = df[(df["quantity"] > 0) & (df["unit_price"] > 0)]
        df["amount"] = df["quantity"] * df["unit_price"]
        df = df.loc[:, ["customer_id","invoice_date","amount"]]
        df = df.sort_values(["customer_id","invoice_date"]).reset_index(drop=True)
        return df, None

    return None, f"File does not contain required columns. Expect either (customer_id, invoice_date, amount) OR (customer_id, invoice_date, quantity, unit_price). Got: {list(raw.columns)}"

def synthetic_transactions(n_customers=10000, start="2024-01-01", end="2024-12-31", seed=42):
    np.random.seed(seed)
    start = pd.Timestamp(start); end = pd.Timestamp(end)
    cust = pd.DataFrame({"customer_id": np.arange(1, n_customers+1).astype(str)})
    seg = np.random.choice(["Value","Loyal","Premium","Occasional"], size=n_customers, p=[0.45,0.25,0.2,0.10])
    cust["latent_segment"] = seg
    lam = cust["latent_segment"].map({"Value":3,"Loyal":8,"Premium":12,"Occasional":1}).values
    purchases = np.random.poisson(lam=lam)
    rows = []
    for cid, s, k in zip(cust.customer_id, cust.latent_segment, purchases):
        if k == 0: 
            continue
        dates = pd.to_datetime(np.random.randint(start.value//10**9, end.value//10**9, size=k), unit="s")
        base = {"Value": 30, "Loyal": 50, "Premium": 120, "Occasional": 25}[s]
        amt = np.maximum(5, np.random.normal(base, base*0.3, size=k)).round(2)
        rows.append(pd.DataFrame({"customer_id": cid, "invoice_date": dates, "amount": amt}))
    tx = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["customer_id","invoice_date","amount"])
    return tx.sort_values(["customer_id","invoice_date"]).reset_index(drop=True)

def build_rfm(tx: pd.DataFrame):
    snap = tx["invoice_date"].max() + pd.Timedelta(days=1)
    rfm = (
        tx.groupby("customer_id")
          .agg(
              recency_days=("invoice_date", lambda s: (snap - s.max()).days),
              frequency=("invoice_date","count"),
              monetary=("amount","sum")
          )
          .reset_index()
    )
    rfm["aov"] = rfm["monetary"] / rfm["frequency"]
    return rfm

def cluster_kmeans(rfm, k):
    X = rfm[["recency_days","frequency","monetary","aov"]].copy()
    X["recency_days"] = np.log1p(X["recency_days"])
    X["monetary"] = np.log1p(X["monetary"])
    X["aov"] = np.log1p(X["aov"])
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    km = KMeans(n_clusters=k, n_init=50, random_state=42)
    labels = km.fit_predict(Xs)
    sil = silhouette_score(Xs, labels) if k > 1 else np.nan
    return labels, Xs, sil

def profile_clusters(rfm):
    prof = (
        rfm.groupby("kmeans_cluster")
           .agg(customers=("customer_id","count"),
                recency_days=("recency_days","median"),
                frequency=("frequency","median"),
                monetary=("monetary","median"),
                aov=("aov","median"))
           .sort_values("customers", ascending=False)
           .reset_index()
    )
    return prof

def segment_labeler(row, prof_medians):
    # Higher frequency/monetary & lower recency => more valuable
    if row["frequency"]>prof_medians["frequency"] and row["monetary"]>prof_medians["monetary"] and row["recency_days"]<prof_medians["recency_days"]:
        return "Champions"
    elif row["frequency"]>prof_medians["frequency"] and row["recency_days"]<prof_medians["recency_days"]:
        return "Loyal"
    elif row["monetary"]>prof_medians["monetary"]:
        return "High-Value"
    elif row["recency_days"]>prof_medians["recency_days"]:
        return "At-Risk"
    else:
        return "Value"

def pca_scatter(Xs, labels):
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(Xs)
    viz = pd.DataFrame(coords, columns=["pc1","pc2"])
    viz["cluster"] = labels.astype(str)
    fig = px.scatter(viz, x="pc1", y="pc2", color="cluster", title="K-Means clusters (PCA 2D)")
    return fig

def ab_summary(df):
    g = df.groupby("variant").agg(n=("customer_id","count"), conv=("converted","sum"))
    g["cr"] = g["conv"]/g["n"]
    A, B = g.loc["A"], g.loc["B"]
    p_pool = (A.conv + B.conv) / (A.n + B.n)
    se = np.sqrt(p_pool*(1-p_pool)*(1/A.n + 1/B.n))
    diff = B.cr - A.cr
    z = diff/se if se>0 else np.nan
    pval = 2*(1-stats.norm.cdf(abs(z))) if se>0 else np.nan
    ci_low, ci_high = (diff - 1.96*se, diff + 1.96*se) if se>0 else (np.nan, np.nan)
    return pd.DataFrame({
        "A_n":[A.n], "A_CR":[A.cr],
        "B_n":[B.n], "B_CR":[B.cr],
        "Lift (B-A)":[diff], "95% CI low":[ci_low], "95% CI high":[ci_high], "p_value":[pval]
    })

def roi_block(df, cpi, profit_per_conv):
    g = df.groupby("variant").agg(n=("customer_id","count"), conv=("converted","sum"))
    g["cr"] = g["conv"]/g["n"]
    A, B = g.loc["A"], g.loc["B"]
    incr_conversions = (B.cr - A.cr) * min(A.n, B.n)
    incr_profit = incr_conversions * profit_per_conv
    incr_cost   = (B.n - A.n) * cpi  # ≈0 if 50/50 split
    roi = (incr_profit - incr_cost) / (incr_cost if incr_cost>0 else 1e-9)
    return pd.Series({"incr_conversions":incr_conversions, "incr_profit":incr_profit, "incr_cost":incr_cost, "ROI":roi})

# ------------------------- Sidebar -------------------------
st.sidebar.header("1) Data")
mode = st.sidebar.radio("Choose data source", ["Upload file", "Use synthetic"], index=1)
upload = None
if mode == "Upload file":
    upload = st.sidebar.file_uploader("CSV or Excel with transactions", type=["csv","xlsx","xls"])
else:
    st.sidebar.write("Synthetic data will be generated (12 months).")

st.sidebar.header("2) Clustering")
k = st.sidebar.slider("Number of clusters (K-Means)", 2, 10, 5)
st.sidebar.caption("Tip: try several K and compare Silhouette in the main panel.")

st.sidebar.header("3) A/B & ROI")
simulate_ab = st.sidebar.radio("A/B source", ["Simulate from segments", "Upload A/B outcomes"], index=0)
cpi = st.sidebar.number_input("Cost per impression (CPI) $", value=0.02, step=0.01, min_value=0.0)
profit_per_conv = st.sidebar.number_input("Profit per conversion $", value=20.0, step=1.0, min_value=0.0)

# ------------------------- Main flow -------------------------
# Load transactions
if mode == "Upload file":
    tx, err = robust_read_transactions(upload)
    if err: st.warning(err)
else:
    tx = synthetic_transactions()

if tx is None or tx.empty:
    st.info("➡️ Load or generate data to continue.")
    st.stop()

with st.expander("Preview transactions", expanded=False):
    st.dataframe(tx.head(20), use_container_width=True)
    st.caption(f"Rows: {len(tx):,} | Date range: {tx['invoice_date'].min().date()} → {tx['invoice_date'].max().date()}")

# RFM
rfm = build_rfm(tx)
st.subheader("RFM Features")
st.dataframe(rfm.describe().T, use_container_width=True)

# K-Means
labels, Xs, sil = cluster_kmeans(rfm, k=k)
rfm["kmeans_cluster"] = labels
st.metric("Silhouette score", f"{sil:.3f}")
prof = profile_clusters(rfm)
st.subheader("Cluster profile (median stats)")
st.dataframe(prof, use_container_width=True)

# Segment labels
meds = {
    "recency_days": rfm["recency_days"].median(),
    "frequency": rfm["frequency"].median(),
    "monetary": rfm["monetary"].median()
}
rfm["segment"] = rfm.apply(lambda r: segment_labeler(r, meds), axis=1)
st.write("Segment distribution")
st.bar_chart(rfm["segment"].value_counts())

# PCA scatter
st.subheader("PCA 2D visualization")
fig = pca_scatter(Xs, labels)
st.plotly_chart(fig, use_container_width=True)

# ------------------------- A/B Testing -------------------------
st.header("A/B Testing & ROI")

if simulate_ab == "Upload A/B outcomes":
    """
    Expect a CSV with at least:
    - customer_id
    - variant (A or B)
    - converted (0/1)
    We'll left-join it to the segmentation table.
    """
    ab_file = st.file_uploader("Upload A/B outcomes CSV", type=["csv"])
    if ab_file is not None:
        ab_raw = pd.read_csv(ab_file)
        # normalize
        cols = {c.lower(): c for c in ab_raw.columns}
        get = lambda x: cols.get(x, None)
        need = [get("customer_id"), get("variant"), get("converted")]
        if None in need:
            st.error("A/B CSV must include customer_id, variant, converted.")
            st.stop()
        ab = ab_raw.rename(columns={get("customer_id"):"customer_id", get("variant"):"variant", get("converted"):"converted"})
        ab["customer_id"] = ab["customer_id"].astype(str)
        ab["variant"] = ab["variant"].astype(str).str.upper().str.strip()
        ab["converted"] = ab["converted"].astype(int)
    else:
        st.info("Upload A/B CSV to continue or switch to 'Simulate from segments'.")
        st.stop()
else:
    # Simulate per-segment propensities
    base_conv = {"Champions":0.15, "Loyal":0.12, "High-Value":0.10, "Value":0.05, "At-Risk":0.02}
    uplift    = {"Champions":0.02, "Loyal":0.02, "High-Value":0.03, "Value":0.02, "At-Risk":0.01}
    ab = rfm[["customer_id","segment"]].copy()
    ab["variant"] = np.where(np.random.rand(len(ab)) < 0.5, "A", "B")
    p = ab.apply(lambda r: base_conv.get(r["segment"],0.05) + (uplift.get(r["segment"],0.0) if r["variant"]=="B" else 0.0), axis=1)
    ab["converted"] = (np.random.rand(len(ab)) < p).astype(int)

# Join segments (if sourced externally, ensure we still have segment)
if "segment" not in ab.columns:
    ab = ab.merge(rfm[["customer_id","segment"]], on="customer_id", how="left")

# Overall A/B
overall = ab_summary(ab)
st.subheader("Overall A/B results")
st.dataframe(overall, use_container_width=True)

# Segment-level A/B + BH-FDR
seg_rows = []
for seg, g in ab.groupby("segment"):
    row = ab_summary(g).assign(segment=seg)
    seg_rows.append(row)
seg_tbl = pd.concat(seg_rows, ignore_index=True).sort_values("p_value")
m = len(seg_tbl)
seg_tbl["BH_crit"] = [((i+1)/m)*0.05 for i in range(m)]
seg_tbl["BH_significant"] = seg_tbl["p_value"] <= seg_tbl["BH_crit"]

st.subheader("A/B by segment (Benjamini–Hochberg FDR)")
st.dataframe(seg_tbl, use_container_width=True)

# Uplift plot
seg_plot = seg_tbl.copy()
seg_plot["Lift_mid"] = seg_plot["Lift (B-A)"]
fig_lift = px.scatter(seg_plot, x="segment", y="Lift_mid", color="BH_significant",
                      error_y=seg_plot["95% CI high"]-seg_plot["Lift_mid"],
                      error_y_minus=seg_plot["Lift_mid"]-seg_plot["95% CI low"],
                      title="Segment uplift (B-A) with 95% CI")
fig_lift.update_layout(xaxis_title="", yaxis_title="Lift")
st.plotly_chart(fig_lift, use_container_width=True)

# ROI
roi_overall = roi_block(ab, cpi=cpi, profit_per_conv=profit_per_conv)
roi_by_seg = ab.groupby("segment").apply(lambda d: roi_block(d, cpi, profit_per_conv)).reset_index()

st.subheader("ROI")
col1, col2 = st.columns(2)
with col1:
    st.write("Overall ROI")
    st.dataframe(pd.DataFrame([roi_overall]), use_container_width=True)
with col2:
    st.write("ROI by segment")
    st.dataframe(roi_by_seg, use_container_width=True)

# Power / Sample size
baseline_cr = overall["A_CR"].iloc[0]
detectable_lift = st.slider("Detectable lift for power calc (absolute, e.g., 0.02 = +2pp)", 0.005, 0.05, 0.02, 0.005)
effect = detectable_lift
power_analysis = sms.NormalIndPower()
try:
    required_n_per_arm = power_analysis.solve_power(
        effect_size = sms.proportion_effectsize(baseline_cr, baseline_cr + effect),
        power=0.8, alpha=0.05, ratio=1.0, alternative='two-sided'
    )
    st.metric("Required sample size per arm (80% power, α=0.05)", f"{int(np.ceil(required_n_per_arm)):,}")
except Exception:
    st.info("Power calc skipped (edge case with baseline CR).")

# Downloads
st.subheader("Download tables")
colA, colB, colC = st.columns(3)
with colA:
    st.download_button("Download Overall A/B CSV", data=overall.to_csv(index=False), file_name="ab_overall.csv", mime="text/csv")
with colB:
    st.download_button("Download A/B by Segment CSV", data=seg_tbl.to_csv(index=False), file_name="ab_by_segment.csv", mime="text/csv")
with colC:
    st.download_button("Download ROI by Segment CSV", data=roi_by_seg.to_csv(index=False), file_name="roi_by_segment.csv", mime="text/csv")

st.caption("Built with Streamlit • K-Means on RFM • A/B with z-test, BH-FDR, ROI, and power analysis.")
