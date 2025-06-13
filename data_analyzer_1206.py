import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import anthropic
import os
from dotenv import load_dotenv
import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

# Load environment variables
load_dotenv()

# Set up Claude API client
try:
    client = anthropic.Anthropic(api_key=os.environ.get("CLAUDE_API_KEY"))
except Exception as e:
    st.error(f"Error initializing Claude API: {str(e)}")
    client = None

# Set page configuration
st.set_page_config(page_title="Intelligent Data Analyzer", layout="wide", initial_sidebar_state="expanded")

@st.cache_data
def intelligent_data_cleaning(df):
    """Smart data cleaning with type detection and conversion"""
    cleaned_df = df.copy()
    # Clean column names intelligently
    cleaned_df.columns = [
        re.sub(r'[^\w\s]', '', str(col)).strip().replace(' ', '_').lower()
        for col in cleaned_df.columns
    ]
    # Remove completely empty rows/columns
    cleaned_df = cleaned_df.dropna(how='all').dropna(axis=1, how='all')
    # Smart type detection and conversion
    for col in cleaned_df.columns:
        if cleaned_df[col].dtype == 'object':
            sample = cleaned_df[col].dropna().astype(str).head(100)
            # Currency detection and conversion
            if sample.str.contains(r'[$£€¥₹]', regex=True).any():
                cleaned_df[col] = (cleaned_df[col].astype(str)
                                 .str.replace(r'[$£€¥₹,]', '', regex=True)
                                 .replace('', np.nan))
                cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
            # Percentage detection and conversion
            elif sample.str.contains('%').any():
                cleaned_df[col] = (cleaned_df[col].astype(str)
                                 .str.replace('%', '')
                                 .replace('', np.nan))
                cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce') / 100
            # Number with commas
            elif sample.str.match(r'^\d{1,3}(,\d{3})*\.?\d*$').any():
                cleaned_df[col] = (cleaned_df[col].astype(str)
                                 .str.replace(',', '')
                                 .replace('', np.nan))
                cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
            # Regular numeric conversion
            else:
                numeric_series = pd.to_numeric(cleaned_df[col], errors='coerce')
                if numeric_series.notna().mean() > 0.7:
                    cleaned_df[col] = numeric_series
    # Smart date detection
    for col in cleaned_df.columns:
        if cleaned_df[col].dtype == 'object':
            try:
                date_series = pd.to_datetime(cleaned_df[col], errors='coerce')
                if date_series.notna().mean() > 0.5:
                    cleaned_df[col] = date_series
            except:
                pass
    return cleaned_df

def detect_relationships_between_files(files_data: Dict[str, pd.DataFrame]) -> Dict:
    """Intelligently detect relationships between multiple files"""
    relationships = {
        'common_columns': {},
        'potential_joins': [],
        'time_alignment': {},
        'schema_similarity': {},
        'data_flow': []
    }
    file_names = list(files_data.keys())
    # Find common columns across files
    for i, file1 in enumerate(file_names):
        for j, file2 in enumerate(file_names[i+1:], i+1):
            df1, df2 = files_data[file1], files_data[file2]
            common_cols = set(df1.columns) & set(df2.columns)
            if common_cols:
                relationships['common_columns'][f"{file1} ↔ {file2}"] = list(common_cols)
                # Check for potential join keys
                for col in common_cols:
                    overlap = len(set(df1[col].dropna()) & set(df2[col].dropna()))
                    if overlap > 0:
                        relationships['potential_joins'].append({
                            'file1': file1,
                            'file2': file2,
                            'join_column': col,
                            'overlap_count': overlap,
                            'overlap_pct': (overlap / min(df1[col].nunique(), df2[col].nunique())) * 100
                        })
    return relationships

def generate_insights(df, context="single"):
    """Generate insights using Claude"""
    if client is None:
        return ["Claude API not available"], [], []
    # Get sample data for better analysis
    sample_data = df.head(20).to_dict('records')
    data_stats = {
        'columns': df.columns.tolist(),
        'dtypes': {col: str(df[col].dtype) for col in df.columns},
        'sample_values': {col: df[col].dropna().head(5).tolist() for col in df.columns},
        'null_counts': df.isnull().sum().to_dict(),
        'unique_counts': {col: df[col].nunique() for col in df.columns}
    }
    if context == "single":
        prompt = f"""
        Analyze this dataset and provide business insights:
        Dataset Overview:
        - Shape: {df.shape[0]} rows, {df.shape[1]} columns
        - Columns and types: {data_stats['dtypes']}
        - Sample values per column: {data_stats['sample_values']}
        - Numeric columns: {df.select_dtypes(include=[np.number]).columns.tolist()}
        - Date columns: {[col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]}
        Provide:
        1. 4-5 key business insights based on the data patterns
        2. 3-4 important metrics with actual calculation formulas
        3. 4-5 useful visualizations
        For metrics, provide actual pandas calculation code that can be executed.
        Respond in JSON format:
        {{
            "insights": ["Business insight 1", "Trend insight 2"],
            "metrics": [
                {{"name": "Metric Name", "calculation": "df['column'].sum()", "description": "what it means"}}
            ],
            "visualizations": [
                {{"title": "Chart Title", "type": "bar|line|scatter|pie", "x": "column", "y": "column", "description": "why useful"}}
            ]
        }}
        """
    else:
        prompt = f"""
        Analyze multiple related datasets and provide cross-file insights:
        Datasets: {list(df.keys()) if isinstance(df, dict) else "Multiple files"}
        Focus on insights only possible with multiple files.
        Respond in JSON format with cross-file insights, metrics, and visualizations.
        """
    try:
        response = client.messages.create(
            model="claude-3-7-sonnet-latest",
            max_tokens=1500,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}]
        )
        response_text = response.content[0].text
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            suggestions = json.loads(json_match.group(0))
            return (
                suggestions.get("insights", []),
                suggestions.get("metrics", []),
                suggestions.get("visualizations", [])
            )
        else:
            return ["Analysis complete"], [], []
    except Exception as e:
        return [f"Error getting insights: {str(e)}"], [], []

def create_visualization(df, viz_config):
    """Create visualizations based on config"""
    try:
        chart_type = viz_config.get("type", "bar")
        title = viz_config.get("title", "Chart")
        x_col = viz_config.get("x")
        y_col = viz_config.get("y")
        # Validate columns
        if x_col not in df.columns:
            return None
        if y_col and y_col not in df.columns:
            y_col = None
        # Create visualizations
        if chart_type == "bar":
            if y_col and pd.api.types.is_numeric_dtype(df[y_col]):
                agg_df = df.groupby(x_col)[y_col].sum().reset_index()
                fig = px.bar(agg_df, x=x_col, y=y_col, title=title)
            else:
                counts = df[x_col].value_counts().head(10)
                fig = px.bar(x=counts.index, y=counts.values, title=title)
        elif chart_type == "line":
            if y_col:
                if pd.api.types.is_datetime64_any_dtype(df[x_col]):
                    agg_df = df.groupby(x_col)[y_col].sum().reset_index()
                    fig = px.line(agg_df, x=x_col, y=y_col, title=title)
                else:
                    fig = px.line(df, x=x_col, y=y_col, title=title)
            else:
                counts = df[x_col].value_counts().sort_index()
                fig = px.line(x=counts.index, y=counts.values, title=title)
        elif chart_type == "scatter":
            if y_col:
                fig = px.scatter(df, x=x_col, y=y_col, title=title)
            else:
                fig = px.histogram(df, x=x_col, title=title)
        elif chart_type == "pie":
            counts = df[x_col].value_counts().head(8)
            fig = px.pie(values=counts.values, names=counts.index, title=title)
        else:
            # Default to bar
            counts = df[x_col].value_counts().head(10)
            fig = px.bar(x=counts.index, y=counts.values, title=title)
        fig.update_layout(title_font_size=16, margin=dict(t=50, b=50, l=50, r=50))
        return fig
    except Exception as e:
        st.error(f"Error creating {title}: {str(e)}")
        return None

def calculate_metrics(df, metrics_config):
    """Calculate metrics with actual values"""
    metrics = []
    # Try to calculate AI-suggested metrics
    for metric in metrics_config:
        try:
            if metric.get("calculation", "").startswith("df"):
                # Execute the calculation
                result = eval(metric["calculation"], {"df": df, "pd": pd, "np": np})
                # Format the result
                if isinstance(result, (int, float)):
                    if result > 1000000:
                        value = f"{result/1000000:.1f}M"
                    elif result > 1000:
                        value = f"{result/1000:.1f}K"
                    else:
                        value = f"{result:.2f}"
                else:
                    value = str(result)
                metrics.append({
                    'name': metric.get('name', 'Metric'),
                    'value': value,
                    'description': metric.get('description', '')
                })
        except:
            pass
    # Add basic metrics as fallback
    metrics.append({
        'name': 'Total Records',
        'value': f"{df.shape[0]:,}",
        'description': 'Number of rows in the dataset'
    })
    # Data completeness
    completeness = (df.count().sum() / (df.shape[0] * df.shape[1])) * 100
    metrics.append({
        'name': 'Data Completeness',
        'value': f"{completeness:.1f}%",
        'description': 'Percentage of non-empty cells'
    })
    return metrics

# Smart column analysis function
def analyze_numeric_column(col_data):
    """Analyze if a numeric column is suitable for aggregation"""
    col_data = col_data.dropna()
    if len(col_data) == 0:
        return False, "empty"
    # Check uniqueness ratio - if >80% unique values, likely an ID
    uniqueness_ratio = col_data.nunique() / len(col_data)
    # Check if values are sequential (like IDs)
    if len(col_data) > 10:
        sorted_vals = sorted(col_data)
        differences = [sorted_vals[i+1] - sorted_vals[i] for i in range(min(10, len(sorted_vals)-1))]
        avg_diff = sum(differences) / len(differences) if differences else 0
        is_sequential = avg_diff <= 2 and all(diff >= 0 for diff in differences)
    else:
        is_sequential = False
    # Check value ranges
    col_min, col_max = col_data.min(), col_data.max()
    value_range = col_max - col_min
    # Decision logic
    if uniqueness_ratio > 0.8 and (is_sequential or value_range > len(col_data) * 0.8):
        return False, "likely_id"  # Probably an ID column
    elif uniqueness_ratio > 0.95:
        return False, "too_unique"  # Too unique, probably identifiers
    elif col_data.min() >= 0 and col_data.max() <= 1 and col_data.dtype == 'float':
        return True, "percentage"  # Probably percentages/rates
    elif value_range > 0:
        return True, "metric"  # Good for aggregation
    else:
        return False, "constant"  # All same values

# Initialize session state
if 'analysis_data' not in st.session_state:
    st.session_state.analysis_data = None
if 'analysis_type' not in st.session_state:
    st.session_state.analysis_type = None
if 'hidden_metrics' not in st.session_state:
    st.session_state.hidden_metrics = set()
if 'custom_metrics' not in st.session_state:
    st.session_state.custom_metrics = []

# Sidebar with information
with st.sidebar:
    st.title("📊 About This Tool")
    st.markdown("""
    ### 🎯 Perfect for Business Teams
    - **Sales Teams:** Performance analysis, customer segmentation
    - **Customer Success:** Satisfaction trends, support analysis  
    - **Marketing:** Campaign analysis, lead conversion
    ### 🚀 What It Does
    - **Smart Data Cleaning:** Handles currencies, dates, percentages
    - **AI Insights:** Business-focused analysis using Claude
    - **Interactive Charts:** Professional visualizations
    - **Export Options:** Download cleaned data and reports
    ### 📁 Supported Files
    - CSV files (.csv)
    - Excel files (.xlsx, .xls)
    - Multiple sheets detected automatically
    - Up to 200MB per file
    ### 💡 Pro Tips
    - Include date columns for trend analysis
    - Use consistent naming across files
    - Clean obvious errors before upload
    - For multi-file: ensure common ID fields
    """)
    st.markdown("---")
    # Sample data options
    st.subheader("🎯 Try Sample Data")
    if st.button("📈 Sales Sample", use_container_width=True):
        np.random.seed(42)
        sample_data = pd.DataFrame({
            'Date': pd.date_range('2024-01-01', periods=500, freq='D')[:500],
            'Sales_Rep': np.random.choice(['Alice J.', 'Bob S.', 'Carol D.'], 500),
            'Region': np.random.choice(['North', 'South', 'East', 'West'], 500),
            'Revenue': np.random.normal(5000, 1500, 500).round(2),
            'Product': np.random.choice(['Software', 'Training', 'Support'], 500)
        })
        st.session_state.analysis_data = {'Sales_Data.csv': sample_data}
        st.session_state.analysis_type = 'single'
        st.rerun()
    if st.button("📋 Survey Sample", use_container_width=True):
        np.random.seed(43)
        sample_data = pd.DataFrame({
            'Customer_ID': [f'CUST_{i:04d}' for i in range(1, 301)],
            'Satisfaction': np.random.choice([1, 2, 3, 4, 5], 300, p=[0.05, 0.1, 0.2, 0.4, 0.25]),
            'Product_Rating': np.random.choice([1, 2, 3, 4, 5], 300),
            'Support_Rating': np.random.choice([1, 2, 3, 4, 5], 300),
            'Segment': np.random.choice(['Enterprise', 'SMB', 'Startup'], 300),
            'Survey_Date': pd.date_range('2024-01-01', periods=300)
        })
        st.session_state.analysis_data = {'Survey_Data.csv': sample_data}
        st.session_state.analysis_type = 'single'
        st.rerun()

# Main Application
st.title("🧠 Intelligent Data Analyzer")

# Main choice - Single or Multi-file
if not st.session_state.analysis_data:
    st.markdown("### Choose your analysis type:")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📄 Single File Analysis")
        st.write("Upload one file (CSV or Excel) for individual analysis")
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["csv", "xlsx", "xls"],
            key="single_file"
        )
        if uploaded_file:
            with st.spinner("Processing file..."):
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                        st.session_state.analysis_data = {uploaded_file.name: intelligent_data_cleaning(df)}
                        st.session_state.analysis_type = 'single'
                        st.rerun()
                    else:
                        # Excel file - check for sheets
                        excel_file = pd.ExcelFile(uploaded_file)
                        sheet_names = excel_file.sheet_names
                        if len(sheet_names) == 1:
                            df = pd.read_excel(uploaded_file)
                            st.session_state.analysis_data = {uploaded_file.name: intelligent_data_cleaning(df)}
                            st.session_state.analysis_type = 'single'
                            st.rerun()
                        else:
                            st.write(f"**Found {len(sheet_names)} sheets. Choose one:**")
                            selected_sheet = st.selectbox("Select sheet", sheet_names)
                            if st.button("Analyze Selected Sheet"):
                                df = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
                                file_name = f"{uploaded_file.name} - {selected_sheet}"
                                st.session_state.analysis_data = {file_name: intelligent_data_cleaning(df)}
                                st.session_state.analysis_type = 'single'
                                st.rerun()
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
    with col2:
        st.subheader("🔗 Multi-File Analysis")
        st.write("Upload multiple related files for cross-file insights")
        uploaded_files = st.file_uploader(
            "Choose multiple files",
            type=["csv", "xlsx", "xls"],
            accept_multiple_files=True,
            key="multi_files"
        )
        if uploaded_files:
            with st.spinner("Processing files..."):
                temp_data = {}
                for file in uploaded_files:
                    try:
                        if file.name.endswith('.csv'):
                            df = pd.read_csv(file)
                            temp_data[file.name] = intelligent_data_cleaning(df)
                        else:
                            excel_file = pd.ExcelFile(file)
                            sheet_names = excel_file.sheet_names
                            if len(sheet_names) == 1:
                                df = pd.read_excel(file)
                                temp_data[file.name] = intelligent_data_cleaning(df)
                            else:
                                for sheet in sheet_names:
                                    df = pd.read_excel(file, sheet_name=sheet)
                                    if df.shape[0] > 0:
                                        sheet_key = f"{file.name} - {sheet}"
                                        temp_data[sheet_key] = intelligent_data_cleaning(df)
                    except Exception as e:
                        st.error(f"Error with {file.name}: {str(e)}")
                if temp_data:
                    st.success(f"Found {len(temp_data)} datasets")
                    # Show datasets found
                    for name, df in temp_data.items():
                        st.write(f"• **{name}** - {df.shape[0]:,} rows, {df.shape[1]} columns")
                    # Let user select which to analyze
                    selected_datasets = st.multiselect(
                        "Select datasets to analyze",
                        options=list(temp_data.keys()),
                        default=list(temp_data.keys())
                    )
                    if selected_datasets and st.button("Start Multi-File Analysis"):
                        selected_data = {name: temp_data[name] for name in selected_datasets}
                        if len(selected_data) == 1:
                            st.session_state.analysis_data = selected_data
                            st.session_state.analysis_type = 'single'
                        else:
                            # Check relationships
                            relationships = detect_relationships_between_files(selected_data)
                            if relationships['potential_joins']:
                                st.session_state.analysis_data = selected_data
                                st.session_state.analysis_type = 'multi'
                            else:
                                st.warning("No relationships found. Will analyze separately.")
                                st.session_state.analysis_data = selected_data
                                st.session_state.analysis_type = 'separate'
                        st.rerun()

# Analysis Results
elif st.session_state.analysis_data:
    data = st.session_state.analysis_data
    analysis_type = st.session_state.analysis_type
    # Back button
    col1, col2 = st.columns([6, 1])
    with col1:
        if st.button("⬅️ Back to Upload"):
            st.session_state.analysis_data = None
            st.session_state.analysis_type = None
            st.session_state.hidden_metrics = set()
            st.session_state.custom_metrics = []
            st.rerun()
    with col2:
        if st.button("🔄 Reset KPIs"):
            st.session_state.hidden_metrics = set()
            st.session_state.custom_metrics = []
            st.rerun()

    if analysis_type == 'single':
        # Single file analysis
        file_name = list(data.keys())[0]
        df = list(data.values())[0]
        st.subheader(f"📊 Analysis: {file_name}")

        # Generate insights
        with st.spinner("Analyzing data..."):
            insights, ai_metrics, visualizations = generate_insights(df, "single")
            calculated_metrics = calculate_metrics(df, ai_metrics)

        # KPI Section at the top
        st.write("### 📈 Key Performance Indicators")
        # Settings for KPIs
        with st.expander("⚙️ Customize KPIs", expanded=False):
            st.write("Select which metrics to display:")
            # Define all possible metrics
            all_metrics = {
                "total_records": "Total Records",
                "data_completeness": "Data Completeness",
                "best_numeric": "Best Numeric Metric",
                "best_category": "Best Category Count",
                "date_range": "Date Range",
                "date_from": "Start Date",
                "date_to": "End Date",
                "trend": "Trend Analysis"
            }
            # Create checkboxes for each metric
            metric_cols = st.columns(4)
            for i, (key, label) in enumerate(all_metrics.items()):
                with metric_cols[i % 4]:
                    if st.checkbox(label, value=key not in st.session_state.hidden_metrics, key=f"show_{key}"):
                        st.session_state.hidden_metrics.discard(key)
                    else:
                        st.session_state.hidden_metrics.add(key)
        # Auto-generate KPIs based on data
        kpi_cols = st.columns(4)
        kpi_index = 0
        # Basic KPIs
        if "total_records" not in st.session_state.hidden_metrics and kpi_index < 4:
            with kpi_cols[kpi_index % 4]:
                st.metric(
                    "Total Records",
                    f"{df.shape[0]:,}",
                    help="Total number of rows in the dataset"
                )
                kpi_index += 1
        if "data_completeness" not in st.session_state.hidden_metrics and kpi_index < 4:
            with kpi_cols[kpi_index % 4]:
                completeness = (df.count().sum() / (df.shape[0] * df.shape[1])) * 100
                st.metric(
                    "Data Completeness",
                    f"{completeness:.1f}%",
                    help="Percentage of non-empty cells"
                )
                kpi_index += 1
        # Smart KPIs based on data content
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = [col for col in df.columns if df[col].dtype == 'object' and df[col].nunique() < 50]
        date_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]

        if numeric_cols and "best_numeric" not in st.session_state.hidden_metrics and kpi_index < 4:
            with kpi_cols[kpi_index % 4]:
                # Find best numeric column for KPI
                best_col = None
                best_type = None
                # Analyze all numeric columns
                for col in numeric_cols:
                    is_suitable, col_type = analyze_numeric_column(df[col])
                    if is_suitable:
                        # Prioritize business-sounding columns
                        col_lower = col.lower()
                        priority_score = 0
                        # High priority terms
                        if any(term in col_lower for term in ['revenue', 'sales', 'amount', 'value', 'price']):
                            priority_score += 10
                        # Medium priority terms
                        elif any(term in col_lower for term in ['cost', 'profit', 'margin', 'score', 'rating']):
                            priority_score += 5
                        # Time-related terms
                        elif any(term in col_lower for term in ['time', 'duration', 'hours', 'days']):
                            priority_score += 3
                        if best_col is None or priority_score > 0:
                            best_col = col
                            best_type = col_type
                            if priority_score > 0:
                                break  # Found a high-priority business metric
                if best_col:
                    col_data = df[best_col].dropna()
                    col_lower = best_col.lower()
                    # Format based on column type and content
                    if any(money_term in col_lower for money_term in ['revenue', 'sales', 'amount', 'value', 'price', 'cost']):
                        # Money formatting
                        total_value = col_data.sum()
                        if total_value > 1000000:
                            display_value = f"${total_value/1000000:.1f}M"
                        elif total_value > 1000:
                            display_value = f"${total_value/1000:.1f}K"
                        else:
                            display_value = f"${total_value:.0f}"
                        metric_name = f"Total {best_col.replace('_', ' ').title()}"
                    elif any(time_term in col_lower for time_term in ['time', 'duration', 'hours']):
                        # Time formatting
                        avg_time = col_data.mean()
                        if 'hour' in col_lower or avg_time > 24:
                            display_value = f"{avg_time:.1f}h"
                        elif avg_time < 1:
                            display_value = f"{avg_time*60:.0f}min"
                        else:
                            display_value = f"{avg_time:.1f}h"
                        metric_name = f"Avg {best_col.replace('_', ' ').title()}"
                    elif best_type == "percentage" or any(pct_term in col_lower for pct_term in ['rate', 'percent', 'ratio']):
                        # Percentage formatting
                        avg_value = col_data.mean()
                        if avg_value <= 1:
                            display_value = f"{avg_value*100:.1f}%"
                        else:
                            display_value = f"{avg_value:.1f}%"
                        metric_name = f"Avg {best_col.replace('_', ' ').title()}"
                    else:
                        # Generic numeric formatting
                        avg_value = col_data.mean()
                        if avg_value > 1000000:
                            display_value = f"{avg_value/1000000:.1f}M"
                        elif avg_value > 1000:
                            display_value = f"{avg_value/1000:.1f}K"
                        else:
                            display_value = f"{avg_value:.2f}"
                        metric_name = f"Avg {best_col.replace('_', ' ').title()}"
                    st.metric(
                        metric_name,
                        display_value,
                        help=f"Calculated from {best_col} column"
                    )
                    kpi_index += 1
                else:
                    # Fallback - count of records
                    st.metric(
                        "Data Points",
                        f"{df.shape[0]:,}",
                        help="Total number of records"
                    )
                    kpi_index += 1
        if categorical_cols and "best_category" not in st.session_state.hidden_metrics and kpi_index < 4:
            with kpi_cols[kpi_index % 4]:
                # Find most interesting categorical field
                best_cat_col = None
                max_priority = 0
                for col in categorical_cols:
                    col_lower = col.lower()
                    priority = 0
                    # Business entity priorities
                    if any(term in col_lower for term in ['customer', 'client', 'user']):
                        priority = 10
                    elif any(term in col_lower for term in ['product', 'item', 'service']):
                        priority = 9
                    elif any(term in col_lower for term in ['region', 'location', 'country', 'city']):
                        priority = 8
                    elif any(term in col_lower for term in ['category', 'type', 'status']):
                        priority = 7
                    elif any(term in col_lower for term in ['team', 'department', 'group']):
                        priority = 6
                    # Also consider uniqueness - more unique = more interesting
                    uniqueness = df[col].nunique()
                    if uniqueness > 1:  # Must have variety
                        priority += min(uniqueness / 10, 3)  # Bonus for diversity
                    if priority > max_priority:
                        max_priority = priority
                        best_cat_col = col
                if best_cat_col:
                    unique_count = df[best_cat_col].nunique()
                    st.metric(
                        f"Unique {best_cat_col.replace('_', ' ').title()}",
                        f"{unique_count}",
                        help=f"Number of distinct {best_cat_col} values"
                    )
                    kpi_index += 1
        # Display custom metrics if any
        if st.session_state.custom_metrics and kpi_index < 4:
            for custom_metric in st.session_state.custom_metrics[:4-kpi_index]:
                with kpi_cols[kpi_index % 4]:
                    st.metric(
                        custom_metric.get("name", "Custom Metric"),
                        custom_metric.get("value", "N/A"),
                        help=custom_metric.get("description", "Custom calculation")
                    )
                kpi_index += 1
        # Additional KPIs row if we have date data
        if date_cols and any(key not in st.session_state.hidden_metrics for key in ["date_range", "date_from", "date_to", "trend"]):
            st.write("")  # Add some space
            date_kpi_cols = st.columns(4)
            date_kpi_index = 0
            date_col = date_cols[0]
            date_range = df[date_col].max() - df[date_col].min()
            if "date_range" not in st.session_state.hidden_metrics and date_kpi_index < 4:
                with date_kpi_cols[date_kpi_index]:
                    st.metric(
                        "Date Range",
                        f"{date_range.days} days",
                        help="Time period covered by the data"
                    )
                    date_kpi_index += 1
            if "date_from" not in st.session_state.hidden_metrics and date_kpi_index < 4:
                with date_kpi_cols[date_kpi_index]:
                    st.metric(
                        "From",
                        df[date_col].min().strftime('%Y-%m-%d'),
                        help="Earliest date in dataset"
                    )
                    date_kpi_index += 1
            if "date_to" not in st.session_state.hidden_metrics and date_kpi_index < 4:
                with date_kpi_cols[date_kpi_index]:
                    st.metric(
                        "To",
                        df[date_col].max().strftime('%Y-%m-%d'),
                        help="Latest date in dataset"
                    )
                    date_kpi_index += 1
            # Check for trends if numeric columns exist
            if numeric_cols and "trend" not in st.session_state.hidden_metrics and date_kpi_index < 4:
                with date_kpi_cols[date_kpi_index]:
                    # Find a good numeric column for trend
                    trend_col = None
                    for col in numeric_cols:
                        is_suitable, _ = analyze_numeric_column(df[col])
                        if is_suitable:
                            trend_col = col
                            break
                    if trend_col:
                        # Calculate simple trend
                        df_sorted = df.sort_values(date_col)
                        first_quarter = df_sorted.head(len(df_sorted)//4)[trend_col].mean()
                        last_quarter = df_sorted.tail(len(df_sorted)//4)[trend_col].mean()
                        trend_pct = ((last_quarter - first_quarter) / first_quarter * 100) if first_quarter != 0 else 0
                        st.metric(
                            "Trend",
                            f"{'+' if trend_pct > 0 else ''}{trend_pct:.1f}%",
                            delta=f"{trend_col.replace('_', ' ').title()}",
                            help="Change from first to last quarter of data"
                        )
                        date_kpi_index += 1
        # Insights Section
        st.write("### 💡 Key Insights")
        if insights:
            for i, insight in enumerate(insights[:5], 1):
                st.info(f"**Insight {i}:** {insight}")
        else:
            st.info("Upload data to see AI-generated insights")
        # Custom Analysis Section
        st.write("### 🎯 Custom Analysis")
        with st.expander("Ask for specific metrics or insights", expanded=False):
            col1, col2 = st.columns([3, 1])
            with col1:
                user_prompt = st.text_area(
                    "What would you like to know about your data?",
                    placeholder="Examples:\n"
                                "- What's the average deal size by sales rep?\n"
                                "- Show me customer churn rate over time\n"
                                "- Calculate conversion rate by marketing channel\n"
                                "- Which product has the highest profit margin?\n"
                                "- Show team performance rankings",
                    height=100
                )
            with col2:
                st.write("")  # Spacing
                st.write("")
                analyze_button = st.button("🔍 Analyze", type="primary", use_container_width=True)
            if analyze_button and user_prompt:
                with st.spinner("Generating custom analysis..."):
                    # Custom analysis using Claude
                    if client:
                        # Get context about the data
                        sample_rows = df.head(10).to_dict('records')
                        col_info = {col: {
                            'dtype': str(df[col].dtype),
                            'unique_count': df[col].nunique(),
                            'sample_values': df[col].dropna().head(5).tolist()
                        } for col in df.columns}
                        custom_prompt = f"""
                        You are a business analyst helping analyze data. The user has asked: "{user_prompt}"
                        Dataset information:
                        - Columns and types: {col_info}
                        - Shape: {df.shape[0]} rows, {df.shape[1]} columns
                        - Sample rows: {sample_rows[:3]}
                        Provide practical, actionable metrics that would help make business decisions.
                        Focus on:
                        1. Performance metrics (who/what is doing well/poorly)
                        2. Trends and patterns
                        3. Actionable insights
                        4. Comparison metrics
                        For calculations, provide actual pandas code that can be executed.
                        Respond in JSON format:
                        {{
                            "custom_metrics": [
                                {{
                                    "name": "Metric Name",
                                    "value": "df.groupby('column')['value'].mean()",
                                    "description": "Business meaning and why it matters"
                                }}
                            ],
                            "custom_insights": ["Key finding 1", "Actionable recommendation 2"],
                            "custom_visualizations": [
                                {{
                                    "title": "Chart Title",
                                    "type": "bar|line|scatter|pie",
                                    "x": "column",
                                    "y": "column",
                                    "description": "What this shows"
                                }}
                            ]
                        }}
                        """
                        try:
                            response = client.messages.create(
                                model="claude-3-7-sonnet-latest",
                                max_tokens=1000,
                                temperature=0.3,
                                messages=[{"role": "user", "content": custom_prompt}]
                            )
                            response_text = response.content[0].text
                            st.write(f"DEBUG: Raw Claude Response:\n```json\n{response_text}\n```") # Add this line for debugging
                            
                            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                            if json_match:
                                try:
                                    custom_analysis = json.loads(json_match.group(0))
                                    # Display custom metrics
                                    if custom_analysis.get("custom_metrics"):
                                        st.write("#### 📊 Custom Metrics")
                                        custom_cols = st.columns(len(custom_analysis["custom_metrics"][:4]))
                                        # Store custom metrics in session state
                                        new_custom_metrics = []
                                        for i, metric in enumerate(custom_analysis["custom_metrics"][:4]):
                                            with custom_cols[i]:
                                                try:
                                                    # Execute the calculation
                                                    calc_str = metric.get("value", "")
                                                    if calc_str.startswith("df"):
                                                        result = eval(calc_str, {"df": df, "pd": pd, "np": np})
                                                        # Format result appropriately
                                                        if isinstance(result, pd.Series):
                                                            # If it's a series, show top value
                                                            top_item = result.idxmax()
                                                            top_value = result.max()
                                                            if isinstance(top_value, (int, float)):
                                                                if top_value > 1000000:
                                                                    display_value = f"{top_value/1000000:.1f}M"
                                                                elif top_value > 1000:
                                                                    display_value = f"{top_value/1000:.1f}K"
                                                                else:
                                                                    display_value = f"{top_value:.2f}"
                                                                display_value = f"{top_item}: {display_value}"
                                                            else:
                                                                display_value = f"{top_item}: {top_value}"
                                                        elif isinstance(result, (int, float)):
                                                            if result > 1000000:
                                                                display_value = f"{result/1000000:.1f}M"
                                                            elif result > 1000:
                                                                display_value = f"{result/1000:.1f}K"
                                                            elif result < 1 and result > 0:
                                                                display_value = f"{result*100:.1f}%"
                                                            else:
                                                                display_value = f"{result:.2f}"
                                                        else:
                                                            display_value = str(result)
                                                    else:
                                                        display_value = metric.get("value", "N/A")
                                                    st.metric(
                                                        metric["name"],
                                                        display_value,
                                                        help=metric.get("description", "")
                                                    )
                                                    # Save to session state
                                                    new_custom_metrics.append({
                                                        "name": metric["name"],
                                                        "value": display_value,
                                                        "description": metric.get("description", "")
                                                    })
                                                except Exception as e:
                                                    st.metric(
                                                        metric["name"],
                                                        "Error in calculation",
                                                        help=f"Error: {str(e)}"
                                                    )
                                        # Add to session state
                                        st.session_state.custom_metrics = new_custom_metrics
                                        # Option to add these to main KPIs
                                        if st.button("➕ Add to Main KPIs", use_container_width=True):
                                            st.rerun()
                                    # Display custom insights
                                    if custom_analysis.get("custom_insights"):
                                        st.write("#### 💡 Analysis Results")
                                        for insight in custom_analysis["custom_insights"]:
                                            st.success(insight)
                                    # Create custom visualizations
                                    if custom_analysis.get("custom_visualizations"):
                                        st.write("#### 📈 Custom Visualizations")
                                        viz_cols = st.columns(min(2, len(custom_analysis["custom_visualizations"])))
                                        for i, viz in enumerate(custom_analysis["custom_visualizations"][:4]):
                                            with viz_cols[i % len(viz_cols)]:
                                                custom_fig = create_visualization(df, viz)
                                                if custom_fig:
                                                    st.plotly_chart(custom_fig, use_container_width=True)
                                except json.JSONDecodeError as json_e:
                                    st.error(f"Error decoding JSON from Claude: {json_e}")
                                    st.text(f"Problematic JSON string (from match): {json_match.group(0)}")
                                    st.info("Claude's response might be malformed. Please try refining your prompt or re-running.")

                        except Exception as e:
                            st.error(f"Error generating custom analysis: {str(e)}")
                    else:
                        st.warning("Claude API not available for custom analysis")

        # Visualizations Section
        st.write("### 📊 Data Visualizations")
        # Create visualizations based on AI suggestions
        if visualizations:
            # Use AI-suggested visualizations
            viz_cols = st.columns(2)
            for i, viz_config in enumerate(visualizations[:6]):
                with viz_cols[i % 2]:
                    fig = create_visualization(df, viz_config)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
        else:
            # Fallback visualizations if AI suggestions aren't available
            viz_cols = st.columns(2)
            # Auto-generate visualizations based on data types
            viz_count = 0
            # 1. Time series if date column exists
            if date_cols and numeric_cols and viz_count < 4:
                date_col = date_cols[0]
                for num_col in numeric_cols[:2]:
                    is_suitable, _ = analyze_numeric_column(df[num_col])
                    if is_suitable:
                        with viz_cols[viz_count % 2]:
                            agg_df = df.groupby(date_col)[num_col].sum().reset_index()
                            fig = px.line(agg_df, x=date_col, y=num_col,
                                        title=f"{num_col.replace('_', ' ').title()} Over Time")
                            st.plotly_chart(fig, use_container_width=True)
                            viz_count += 1
                            if viz_count >= 4:
                                break
            # 2. Top categories bar chart
            if categorical_cols and viz_count < 4:
                for cat_col in categorical_cols[:2]:
                    if df[cat_col].nunique() <= 20:
                        with viz_cols[viz_count % 2]:
                            counts = df[cat_col].value_counts().head(10)
                            fig = px.bar(x=counts.index, y=counts.values,
                                         title=f"Distribution of {cat_col.replace('_', ' ').title()}")
                            fig.update_xaxis(title=cat_col)
                            fig.update_yaxis(title="Count")
                            st.plotly_chart(fig, use_container_width=True)
                            viz_count += 1
                            if viz_count >= 4:
                                break
            # 3. Numeric distribution
            if numeric_cols and viz_count < 4:
                for num_col in numeric_cols:
                    is_suitable, col_type = analyze_numeric_column(df[num_col])
                    if is_suitable and col_type != "percentage":
                        with viz_cols[viz_count % 2]:
                            fig = px.histogram(df, x=num_col, nbins=30,
                                               title=f"Distribution of {num_col.replace('_', ' ').title()}")
                            st.plotly_chart(fig, use_container_width=True)
                            viz_count += 1
                            if viz_count >= 4:
                                break
            # 4. Pie chart for main category
            if categorical_cols and viz_count < 4:
                best_cat = None
                for cat in categorical_cols:
                    if 2 <= df[cat].nunique() <= 8:
                        best_cat = cat
                        break
                if best_cat:
                    with viz_cols[viz_count % 2]:
                        counts = df[best_cat].value_counts()
                        fig = px.pie(values=counts.values, names=counts.index,
                                     title=f"{best_cat.replace('_', ' ').title()} Breakdown")
                        st.plotly_chart(fig, use_container_width=True)
                        viz_count += 1

        # Data Preview Section with Smart Aggregation
        st.write("### 📋 Data Summary")
        # Smart aggregation based on data type
        with st.expander("📊 View Options", expanded=True):
            view_option = st.radio(
                "Select view type",
                ["Smart Summary", "Raw Data", "Custom Pivot"],
                horizontal=True,
                help="Smart Summary shows aggregated data for better insights"
            )
        if view_option == "Smart Summary":
            # Automatically create a meaningful summary
            if categorical_cols and numeric_cols:
                # Find the best categorical column for grouping
                best_group_col = None
                for col in categorical_cols:
                    if 2 <= df[col].nunique() <= 20:  # Good range for grouping
                        best_group_col = col
                        break
                if best_group_col:
                    # Find the best numeric columns to aggregate
                    agg_cols = []
                    for col in numeric_cols[:3]:  # Limit to 3 metrics
                        is_suitable, _ = analyze_numeric_column(df[col])
                        if is_suitable:
                            agg_cols.append(col)
                    if agg_cols:
                        # Create summary
                        summary_df = df.groupby(best_group_col)[agg_cols].agg(['sum', 'mean', 'count']).round(2)
                        # Flatten column names
                        summary_df.columns = [f'{col}_{agg}' for col, agg in summary_df.columns]
                        summary_df = summary_df.reset_index()
                        # Sort by the first sum column
                        first_sum_col = [col for col in summary_df.columns if col.endswith('_sum')][0]
                        summary_df = summary_df.sort_values(first_sum_col, ascending=False)
                        st.write(f"**Summary by {best_group_col.replace('_', ' ').title()}**")
                        st.dataframe(
                            summary_df,
                            use_container_width=True,
                            hide_index=True
                        )
                        # Download summary
                        csv = summary_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Summary",
                            data=csv,
                            file_name=f"summary_{file_name}",
                            mime="text/csv"
                        )
                    else:
                        st.write("No suitable numeric columns for aggregation")
                else:
                    st.write("No suitable categorical columns for grouping")
            else:
                st.write("Need both categorical and numeric columns for summary")
        elif view_option == "Custom Pivot":
            # Custom pivot table creation
            col1, col2, col3 = st.columns(3)
            with col1:
                if categorical_cols:
                    row_col = st.selectbox("Rows", options=categorical_cols)
                else:
                    row_col = None
                    st.write("No categorical columns")
            with col2:
                if categorical_cols and len(categorical_cols) > 1:
                    col_options = [c for c in categorical_cols if c != row_col]
                    column_col = st.selectbox("Columns (optional)", options=[None] + col_options)
                else:
                    column_col = None
            with col3:
                if numeric_cols:
                    value_cols = st.multiselect("Values", options=numeric_cols)
                    agg_func = st.selectbox("Aggregation", options=['sum', 'mean', 'count', 'max', 'min'])
                else:
                    value_cols = []
            if row_col and value_cols:
                try:
                    if column_col:
                        # Create pivot table
                        pivot_df = pd.pivot_table(
                            df,
                            values=value_cols,
                            index=row_col,
                            columns=column_col,
                            aggfunc=agg_func,
                            fill_value=0
                        ).round(2)
                    else:
                        # Simple group by
                        pivot_df = df.groupby(row_col)[value_cols].agg(agg_func).round(2)
                    st.dataframe(pivot_df, use_container_width=True)
                    # Download pivot
                    csv = pivot_df.to_csv()
                    st.download_button(
                        label="📥 Download Pivot Table",
                        data=csv,
                        file_name=f"pivot_{file_name}",
                        mime="text/csv"
                    )
                except Exception as e:
                    st.error(f"Error creating pivot: {str(e)}")
        else:  # Raw Data
            # Add filters
            with st.expander("🔍 Filter Data", expanded=False):
                filter_cols = st.columns(3)
                filters = {}
                # Add filters for categorical columns
                cat_filter_count = 0
                for col in categorical_cols[:3]:
                    with filter_cols[cat_filter_count % 3]:
                        unique_vals = df[col].dropna().unique()
                        if len(unique_vals) <= 100:
                            selected = st.multiselect(f"Filter by {col}", unique_vals)
                            if selected:
                                filters[col] = selected
                            cat_filter_count += 1
                # Add numeric range filters
                if numeric_cols:
                    for col in numeric_cols[:2]:
                        is_suitable, _ = analyze_numeric_column(df[col])
                        if is_suitable:
                            with filter_cols[cat_filter_count % 3]:
                                min_val = float(df[col].min())
                                max_val = float(df[col].max())
                                range_vals = st.slider(
                                    f"{col} range",
                                    min_val, max_val, (min_val, max_val)
                                )
                                if range_vals != (min_val, max_val):
                                    filters[f"{col}_range"] = range_vals
                                cat_filter_count += 1
            # Apply filters
            filtered_df = df.copy()
            for col, values in filters.items():
                if "_range" in col:
                    actual_col = col.replace("_range", "")
                    filtered_df = filtered_df[
                        (filtered_df[actual_col] >= values[0]) &
                        (filtered_df[actual_col] <= values[1])
                    ]
                else:
                    filtered_df = filtered_df[filtered_df[col].isin(values)]
            # Show data
            st.dataframe(
                filtered_df.head(100),
                use_container_width=True,
                hide_index=True
            )
            if filtered_df.shape[0] != df.shape[0]:
                st.caption(f"Showing {min(100, filtered_df.shape[0])} of {filtered_df.shape[0]} filtered rows (from {df.shape[0]} total)")
            else:
                st.caption(f"Showing first {min(100, df.shape[0])} of {df.shape[0]} rows")
        # Download Section
        st.write("### 💾 Export Options")
        export_cols = st.columns(3)
        with export_cols[0]:
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Full Data (CSV)",
                data=csv,
                file_name=f"full_{file_name}",
                mime="text/csv"
            )
        with export_cols[1]:
            # Create summary report
            report = f"""# Data Analysis Report
## File: {file_name}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
### Dataset Overview
- Total Rows: {df.shape[0]:,}
- Total Columns: {df.shape[1]}
- Data Completeness: {completeness:.1f}%
### Key Insights
"""
            for i, insight in enumerate(insights[:5], 1):
                report += f"{i}. {insight}\n"
            # Add top performers if available
            if categorical_cols and numeric_cols:
                report += "\n### Top Performers\n"
                for cat_col in categorical_cols[:2]:
                    if df[cat_col].nunique() <= 20:
                        for num_col in numeric_cols[:1]:
                            is_suitable, _ = analyze_numeric_column(df[num_col])
                            if is_suitable:
                                top_performers = df.groupby(cat_col)[num_col].sum().sort_values(ascending=False).head(5)
                                report += f"\n**Top {cat_col} by {num_col}:**\n"
                                for idx, value in top_performers.items():
                                    report += f"- {idx}: {value:,.2f}\n"
                                break
            report += "\n### Column Summary\n"
            for col in df.columns:
                dtype = str(df[col].dtype)
                nunique = df[col].nunique()
                null_pct = (df[col].isnull().sum() / len(df)) * 100
                report += f"- **{col}**: {dtype}, {nunique} unique values, {null_pct:.1f}% missing\n"
            st.download_button(
                label="📄 Download Report (TXT)",
                data=report,
                file_name=f"report_{file_name}.txt",
                mime="text/plain"
            )
        with export_cols[2]:
            if st.button("🔄 Analyze Another File"):
                st.session_state.analysis_data = None
                st.session_state.analysis_type = None
                st.session_state.hidden_metrics = set()
                st.session_state.custom_metrics = []
                st.rerun()
    elif analysis_type == 'multi':
        # Multi-file analysis
        st.subheader("🔗 Multi-File Analysis")
        # Show relationships
        relationships = detect_relationships_between_files(data)
        if relationships['common_columns']:
            st.write("### 🔗 Detected Relationships")
            for files, columns in relationships['common_columns'].items():
                st.write(f"**{files}**: Common columns - {', '.join(columns)}")
        if relationships['potential_joins']:
            st.write("### 🔄 Potential Joins")
            for join in relationships['potential_joins'][:3]:
                st.info(
                    f"Can join **{join['file1']}** with **{join['file2']}** "
                    f"on column **{join['join_column']}** "
                    f"({join['overlap_pct']:.1f}% overlap)"
                )
        # Analyze each file
        st.write("### 📊 Individual File Analysis")
        tabs = st.tabs(list(data.keys()))
        for i, (file_name, df) in enumerate(data.items()):
            with tabs[i]:
                # Generate insights for this file
                with st.spinner(f"Analyzing {file_name}..."):
                    insights, ai_metrics, visualizations = generate_insights(df, "single")
                # KPIs for this file
                st.write("#### 📈 Key Metrics")
                metric_cols = st.columns(4)
                with metric_cols[0]:
                    st.metric("Records", f"{df.shape[0]:,}")
                with metric_cols[1]:
                    st.metric("Columns", f"{df.shape[1]}")
                with metric_cols[2]:
                    completeness = (df.count().sum() / (df.shape[0] * df.shape[1])) * 100
                    st.metric("Completeness", f"{completeness:.1f}%")
                # Quick visualization
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    with metric_cols[3]:
                        col = numeric_cols[0]
                        st.metric(f"Avg {col}", f"{df[col].mean():.2f}")
                # Show a sample visualization
                if len(numeric_cols) > 0:
                    col = numeric_cols[0]
                    fig = px.histogram(df, x=col, title=f"Distribution of {col}")
                    st.plotly_chart(fig, use_container_width=True)
                # Data preview
                st.write("#### Data Preview")
                st.dataframe(df.head(50), use_container_width=True, hide_index=True)
    elif analysis_type == 'separate':
        # Separate analysis for unrelated files
        st.warning("No relationships found between files. Analyzing separately.")
        tabs = st.tabs(list(data.keys()))
        for i, (file_name, df) in enumerate(data.items()):
            with tabs[i]:
                # Similar to single file analysis but in tabs
                st.write(f"### Analysis: {file_name}")
                # Generate insights
                with st.spinner("Analyzing..."):
                    insights, ai_metrics, visualizations = generate_insights(df, "single")
                # Show insights
                if insights:
                    for insight in insights[:3]:
                        st.info(insight)
                # Basic metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Rows", f"{df.shape[0]:,}")
                with col2:
                    st.metric("Columns", f"{df.shape[1]}")
                with col3:
                    completeness = (df.count().sum() / (df.shape[0] * df.shape[1])) * 100
                    st.metric("Completeness", f"{completeness:.1f}%")
                # Quick visualization
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    col = numeric_cols[0]
                    fig = px.histogram(df, x=col, title=f"Distribution of {col}")
                    st.plotly_chart(fig, use_container_width=True)
                # Data preview
                st.write("#### Data Preview")
                st.dataframe(df.head(50), use_container_width=True, hide_index=True)
