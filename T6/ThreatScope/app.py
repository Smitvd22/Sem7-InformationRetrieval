import pandas as pd
import streamlit as st

from database import init_db, fetch_all_reports
from embeddings import search, load_index
from ranking import rank_results
from chatbot import answer as chatbot_answer
from clustering import reduce_dim

st.set_page_config(page_title="ThreatScope AI", layout="wide")


@st.cache_data(ttl=60)
def load_reports():
    return fetch_all_reports()


def get_by_id(reports):
    return {r["id"]: r for r in reports}


def search_tab():
    st.header("Semantic search")
    q = st.text_input("Query", "latest ransomware targeting healthcare")
    top_k = st.slider("Results", 5, 30, 10)

    if st.button("Search", key="search_btn"):
        results = search(q, top_k=top_k)
        reports = load_reports()
        by_id = get_by_id(reports)
        ranked = rank_results(q, results, by_id)
        if not ranked:
            st.warning("No index found. Run `python setup_demo.py` or `python build_index.py`.")
            return
        for r in ranked:
            title = f"[{r.get('source')}] {r.get('title')} — {r.get('severity')} (score {r['rank_score']:.2f})"
            with st.expander(title):
                st.write((r.get("text") or "")[:1200])
                st.caption(f"URL: {r.get('url','')}")
                st.write(
                    f"Threat type: **{r.get('threat_type')}** | CVSS: {r.get('cvss')} | "
                    f"CVEs: {r.get('cve_ids')}"
                )
                st.write(
                    f"ATT&CK: {r.get('attack_techniques')} | "
                    f"Cluster: {r.get('cluster_id')} | "
                    f"Anomaly: {r.get('anomaly_score')}"
                )


def explore_tab():
    st.header("Clusters & topics")
    reports = load_reports()
    if not reports:
        st.info("No data. Run `python setup_demo.py`.")
        return

    df = pd.DataFrame(reports)
    st.write(f"Total reports: {len(df)}")

    if df["cluster_id"].notna().any():
        counts = df["cluster_id"].value_counts().sort_index()
        st.bar_chart(counts)
        pick = st.selectbox("Cluster", counts.index.tolist())
        sub = df[df["cluster_id"] == pick]
        st.dataframe(sub[["title", "source", "severity", "threat_type", "published"]])

    if st.button("Show 2D projection"):
        emb, ids, _ = load_index()
        if emb is not None and len(emb) > 1:
            coords = reduce_dim(emb, method="pca")
            plot_df = pd.DataFrame({
                "x": coords[:, 0],
                "y": coords[:, 1],
                "cluster": df["cluster_id"].values[: len(coords)],
            })
            st.scatter_chart(plot_df, x="x", y="y", color="cluster")
        else:
            st.warning("No index found.")


def anomalies_tab():
    st.header("Anomaly review")
    reports = load_reports()
    if not reports:
        st.info("No data.")
        return
    df = pd.DataFrame(reports).sort_values("anomaly_score", ascending=False)
    st.dataframe(df[["title", "source", "severity", "threat_type", "anomaly_score"]].head(30))


def dashboard_tab():
    st.header("Evaluation dashboard")
    reports = load_reports()
    if not reports:
        st.info("No data.")
        return
    df = pd.DataFrame(reports)

    c1, c2, c3 = st.columns(3)
    c1.metric("Reports", len(df))
    c2.metric("Sources", df["source"].nunique())
    c3.metric(
        "Clusters",
        int(df["cluster_id"].nunique()) if df["cluster_id"].notna().any() else 0,
    )

    st.subheader("Severity distribution")
    st.bar_chart(df["severity"].value_counts())

    st.subheader("Threat type distribution")
    st.bar_chart(df["threat_type"].value_counts())

    if "confidence" in df.columns and df["confidence"].notna().any():
        st.subheader("Average confidence by threat type")
        st.bar_chart(df.groupby("threat_type")["confidence"].mean())


def chatbot_tab():
    st.header("Cited chatbot")
    q = st.text_input("Ask about the threat landscape", "Summarize recent ransomware activity.")
    if st.button("Ask", key="ask_btn"):
        results = search(q, top_k=8)
        reports = load_reports()
        by_id = get_by_id(reports)
        retrieved = [by_id[r["id"]] for r in results if r["id"] in by_id]
        ans = chatbot_answer(q, retrieved)
        st.markdown("**Answer**")
        st.write(ans["answer"])
        st.markdown("**Citations**")
        for c in ans["citations"]:
            st.write("- " + c)


def main():
    init_db()
    st.title("ThreatScope AI — Cyber Threat Intelligence Retrieval")
    tabs = st.tabs(["Search", "Explore", "Anomalies", "Dashboard", "Chatbot"])
    with tabs[0]:
        search_tab()
    with tabs[1]:
        explore_tab()
    with tabs[2]:
        anomalies_tab()
    with tabs[3]:
        dashboard_tab()
    with tabs[4]:
        chatbot_tab()


if __name__ == "__main__":
    main()