import streamlit as st
import requests
import pandas as pd

# FastAPI endpoint
API_URL = "https://shl-fastapi-uf7r.onrender.com/recommend"

# Page setup
st.set_page_config(page_title=" SHL Assessment Recommender", layout="wide")
st.markdown("##  SHL Assessment Recommender")

# Input area
user_query = st.text_area("📝 Enter Job Description or Query:", height=200, placeholder="e.g. We need a data analyst with strong numerical reasoning and Python skills...")

# Recommend button
if st.button("Recommend Tests"):
    if user_query.strip() == "":
        st.warning("Please enter a valid query.")
    else:
        with st.spinner("Generating smart recommendations..."):
            try:
                response = requests.post(API_URL, json={"query": user_query})
                if response.status_code == 200:
                    data = response.json()
                    recommendations = data.get("recommendations", [])

                    if recommendations:
                        st.success(f"✅ Found {len(recommendations)} suitable assessments")

                        # Convert to DataFrame
                        df = pd.DataFrame(recommendations)

                        # Enhance columns
                        df["Test Name"] = df.apply(lambda row: f"[{row['name']}]({row['url']})", axis=1)
                        df["Remote"] = df["remote_testing"].apply(lambda x: "✅ Yes" if str(x).lower() == "yes" else "❌ No")
                        df["Adaptive"] = df["adaptive"].apply(lambda x: "✅ Yes" if str(x).lower() == "yes" else "❌ No")

                        # Reorder and rename
                        df = df[["Test Name", "test_type", "duration", "Remote", "Adaptive"]]
                        df.columns = ["Test Name", "Type", "⏱️ Duration", "Remote", "Adaptive"]

                        # Display with markdown rendering
                        st.markdown(df.to_markdown(index=False), unsafe_allow_html=True)

                    else:
                        st.info("No matching assessments found.")
                else:
                    st.error("API returned an error.")
            except Exception as e:
                st.error(f"🚨 Request failed: {e}")
