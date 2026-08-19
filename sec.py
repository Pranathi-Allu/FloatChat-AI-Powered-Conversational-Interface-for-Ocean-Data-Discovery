import os
import re
import json
import pickle
import requests
import pandas as pd
import streamlit as st
import altair as alt
import pydeck as pdk
import geopandas as gpd
from shapely.geometry import MultiPoint, Point
from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect
from urllib.parse import quote_plus
from typing import Optional, List, Mapping, Any

# --- Langchain Imports ---
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from PyPDF2 import PdfReader
from langchain_core.language_models.llms import LLM
from pydantic import BaseModel, Field


# ------------------- Page Config -------------------
st.set_page_config(
    page_title="Argo AI SQL Assistant",
    page_icon="🌊",
    layout="wide"
)
st.title("🌊 Argo AI SQL Assistant")
st.caption("Ask a question about the Argo float data, and the AI will generate, run, and visualize SQL queries.")


# ------------------- File & DB Paths -------------------
pdf_metadata_path = "floaat_merged.pdf"
faiss_file_path = "faiss_metadata.pkl"
table_name = "argo_data"


# ------------------- LLaMA LLM Wrapper -------------------
class OllamaLLM(LLM, BaseModel):
    model: str = Field(...)
    base_url: str = Field(default="http://localhost:11434")
    temperature: float = Field(default=0.0)

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "options": {"temperature": self.temperature},
        }
        response = requests.post(f"{self.base_url}/api/generate", json=payload)
        response.raise_for_status()
        output = ""
        for line in response.text.splitlines():
            if line.strip():
                try:
                    j = json.loads(line.strip())
                    if "response" in j:
                        output += j["response"]
                except json.JSONDecodeError:
                    continue
        return output

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model": self.model}

    @property
    def _llm_type(self) -> str:
        return "ollama_llm"


# ------------------- DB Connection -------------------
@st.cache_resource
def get_db_engine():
    """Creates and caches a SQLAlchemy engine."""
    try:
        pg_user = os.getenv("POSTGRES_USER", "postgres")
        pg_pass = os.getenv("POSTGRES_PASSWORD", "mahhenuwu123321@")
        pg_db = os.getenv("POSTGRES_DB", "postgres")
        pg_host = os.getenv("POSTGRES_HOST", "localhost")
        pg_port = os.getenv("POSTGRES_PORT", "5432")
        encoded_password = quote_plus(pg_pass)
        postgres_uri = f"postgresql+psycopg2://{pg_user}:{encoded_password}@{pg_host}:{pg_port}/{pg_db}"
        engine = create_engine(postgres_uri)
        return engine
    except Exception as e:
        st.error(f"❌ Failed to connect to PostgreSQL: {e}")
        return None


# ------------------- FAISS Loader -------------------
@st.cache_resource
def load_faiss_index():
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    if os.path.exists(faiss_file_path):
        with open(faiss_file_path, "rb") as f:
            vectorstore = pickle.load(f)
        st.sidebar.success("✅ FAISS index loaded from file.")
        return vectorstore
    else:
        if os.path.exists(pdf_metadata_path):
            st.sidebar.info("Creating FAISS index from metadata PDF...")
            reader = PdfReader(pdf_metadata_path)
            raw_text = "".join(page.extract_text() for page in reader.pages if page.extract_text())
            docs = [Document(page_content=chunk) for chunk in raw_text.split("\n\n") if chunk.strip()]
            vectorstore = FAISS.from_documents(docs, embedding_model)
            with open(faiss_file_path, "wb") as f:
                pickle.dump(vectorstore, f)
            st.sidebar.success("✅ New FAISS index created and saved.")
            return vectorstore
        else:
            st.sidebar.error("❌ Metadata PDF not found!")
            return None


# ------------------- Visualization Helpers -------------------
def df_to_geodf(df, lat_col="latitude", lon_col="longitude"):
    try:
        if lat_col in df.columns and lon_col in df.columns:
            df_clean = df.dropna(subset=[lat_col, lon_col])
            if df_clean.empty:
                return None
            gdf = gpd.GeoDataFrame(
                df_clean.copy(),
                geometry=[Point(xy) for xy in zip(df_clean[lon_col], df_clean[lat_col])],
                crs="EPSG:4326"
            )
            return gdf
    except Exception as e:
        st.warning(f"GeoDataFrame creation failed: {e}")
    return None

def compute_mcp(gdf, buffer_km=0.0):
    try:
        if gdf is None or gdf.empty:
            return None
        multipoint = MultiPoint(list(gdf.geometry))
        polygon = multipoint.convex_hull
        if buffer_km > 0:
            polygon = polygon.buffer(buffer_km / 111.0)
        return polygon
    except Exception:
        return None

def show_pydeck_map(gdf, polygon=None, point_radius=40):
    if gdf is None or gdf.empty:
        st.warning("No geographic data available to map.")
        return
    layers = []
    if 'longitude' in gdf.columns and 'latitude' in gdf.columns:
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=gdf,
                get_position="[longitude, latitude]",
                get_radius=point_radius,
                get_color=[0, 128, 255],
                pickable=True,
            )
        )
    if polygon is not None:
        poly_gdf = gpd.GeoDataFrame(geometry=[polygon], crs="EPSG:4326")
        poly_json = poly_gdf.__geo_interface__["features"]
        layers.append(
            pdk.Layer(
                "GeoJsonLayer",
                data={"type": "FeatureCollection", "features": poly_json},
                stroked=True,
                filled=False,
                get_line_color=[255, 0, 0],
                line_width_min_pixels=2,
            )
        )
    view_state = pdk.ViewState(
        latitude=gdf.geometry.y.mean(),
        longitude=gdf.geometry.x.mean(),
        zoom=3
    )
    st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view_state, map_style="mapbox://styles/mapbox/light-v9"))

def plot_time_series_if_possible(df, date_col="juld"):
    try:
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                y_col = numeric_cols[0]
                chart = alt.Chart(df).mark_line().encode(
                    x=alt.X(f"{date_col}:T", title="Date"),
                    y=alt.Y(f"{y_col}:Q", title=y_col),
                    tooltip=[date_col, y_col]
                ).properties(height=300)
                st.altair_chart(chart, use_container_width=True)
    except Exception as e:
        st.warning(f"Time-series plot failed: {e}")


# ------------------- Initialize -------------------
load_dotenv()
engine = get_db_engine()
vectorstore = load_faiss_index()

if engine and vectorstore:
    inspector = inspect(engine)
    try:
        columns = [col["name"] for col in inspector.get_columns(table_name)]
        st.sidebar.subheader("Available Table Columns")
        st.sidebar.markdown(f"`{', '.join(columns)}`")
    except Exception as e:
        st.error(f"Could not inspect table '{table_name}': {e}")
        st.stop()

    # --- LLM Setup ---
    llm = OllamaLLM(model="mannix/llama3.1-8b-abliterated", base_url="http://localhost:11434", temperature=0)
    sql_template = f"""
You are a machine that only writes PostgreSQL queries. Generate a single valid `SELECT` query using the schema below:

**DATABASE SCHEMA**
- Table: {table_name}
- Columns: {columns}

**RULES**
1. If mixing regular columns & aggregates, add GROUP BY for all regulars.
2. Use temp_adjusted and psal_adjusted for temperature and salinity.
3. Use juld (datetime) for date filtering.
4. Wrap text values in quotes.

**USER QUESTION**
{{question}}

**PostgreSQL Query (RAW SQL ONLY):**
"""
    sql_prompt = PromptTemplate(input_variables=["question"], template=sql_template)


    # --- User Input ---
    user_query = st.text_input("Enter your question:", placeholder="e.g., Show average temperature in Arabian Sea in 2020")

    if st.button("Generate & Run Query"):
        if not user_query:
            st.warning("Please enter a question.")
        else:
            with st.spinner("Processing..."):
                try:
                    # Generate SQL
                    sql_query = llm._call(sql_prompt.format(question=user_query))
                    sql_query = re.sub(r"^```[a-zA-Z]*", "", sql_query).replace("```", "").strip()
                    if sql_query.lower().startswith("sql"):
                        sql_query = sql_query[3:].strip()

                    st.code(sql_query, language="sql")

                    # Execute SQL
                    result_df = pd.read_sql_query(sql_query, engine)

                    if result_df.empty:
                        st.warning("Query returned no results.")
                    else:
                        st.success(f"✅ Query returned {len(result_df)} row(s).")
                        st.dataframe(result_df)

                        # --- Visualization Section ---
                        st.markdown("---")
                        st.subheader("📊 Visualizations")

                        viz_col1, viz_col2 = st.columns([1, 1])

                        with viz_col1:
                            st.write("**🗺️ Map Options**")
                            lat_col_options = [c for c in result_df.columns if 'lat' in c.lower()] or ['latitude']
                            lon_col_options = [c for c in result_df.columns if 'lon' in c.lower()] or ['longitude']
                            lat_col = st.selectbox("Latitude column", options=lat_col_options, index=0)
                            lon_col = st.selectbox("Longitude column", options=lon_col_options, index=0)
                            show_mcp = st.checkbox("Show Convex Hull (MCP)", value=True)
                            buffer_km = st.slider("Buffer MCP (km)", 0.0, 200.0, 0.0, 1.0)
                            point_radius = st.slider("Point radius (m)", 1, 200, 40, 1)

                        with viz_col2:
                            st.write("**📈 Chart Options**")
                            numeric_columns = result_df.select_dtypes(include=['number']).columns.tolist()
                            categorical_columns = result_df.select_dtypes(include=['object', 'category']).columns.tolist()

                            if "juld" in result_df.columns and len(numeric_columns) > 0:
                                default_chart = "Time-Series"
                            elif len(categorical_columns) > 0 and len(numeric_columns) > 0:
                                default_chart = "Bar Chart"
                            elif len(numeric_columns) >= 2:
                                default_chart = "Scatter Plot"
                            else:
                                default_chart = "Histogram"

                            chart_options = ["Histogram", "Bar Chart", "Scatter Plot", "Time-Series"]
                            chart_type = st.radio("Chart type", options=chart_options, index=chart_options.index(default_chart))

                            if chart_type != "Time-Series":
                                x_axis = st.selectbox("X-axis", options=result_df.columns)
                            if chart_type in ["Bar Chart", "Scatter Plot"]:
                                y_axis = st.selectbox("Y-axis", options=numeric_columns)

                        # Render
                        gdf = df_to_geodf(result_df, lat_col, lon_col)
                        if gdf is not None:
                            polygon = compute_mcp(gdf, buffer_km) if show_mcp else None
                            show_pydeck_map(gdf, polygon, point_radius)

                        if chart_type == "Time-Series":
                            plot_time_series_if_possible(result_df, date_col="juld")
                        elif chart_type == "Histogram" and x_axis in numeric_columns:
                            st.altair_chart(alt.Chart(result_df).mark_bar().encode(
                                x=alt.X(x_axis, bin=alt.Bin(maxbins=40)),
                                y='count()'
                            ), use_container_width=True)
                        elif chart_type == "Bar Chart" and x_axis in categorical_columns and 'y_axis' in locals() and y_axis in numeric_columns:
                            st.altair_chart(alt.Chart(result_df).mark_bar().encode(
                                x=alt.X(x_axis, sort='-y'),
                                y=y_axis,
                                tooltip=[x_axis, y_axis]
                            ), use_container_width=True)
                        elif chart_type == "Scatter Plot" and x_axis in numeric_columns and 'y_axis' in locals() and y_axis in numeric_columns:
                            st.altair_chart(alt.Chart(result_df).mark_circle(size=60).encode(
                                x=x_axis,
                                y=y_axis,
                                tooltip=[x_axis, y_axis]
                            ).interactive(), use_container_width=True)

                except Exception as e:
                    st.error(f"An error occurred: {e}")
else:
    st.error("Could not initialize database or FAISS index.")
