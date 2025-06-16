import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from dotenv import load_dotenv
import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import requests

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Claude AI Data Analyzer", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# ================================
# ENHANCED BUSINESS CALCULATION ENGINE
# ================================

def execute_safe_calculation(calc_str: str, df: pd.DataFrame):
    """Enhanced safe calculation engine for business metrics"""
    try:
        # Direct numeric values
        if calc_str.replace('.', '').replace('-', '').isdigit():
            return float(calc_str)
        
        # Safe pandas operations mapping
        safe_operations = {
            "len(df)": lambda: len(df),
            "df.shape[0]": lambda: df.shape[0],
            "df.shape[1]": lambda: df.shape[1],
        }
        
        # Check for direct matches first
        if calc_str in safe_operations:
            return safe_operations[calc_str]()
        
        import re
        
        # NEW PATTERN: Sum of two columns with mean - (df['col1'] + df['col2']).mean()
        sum_mean_pattern = r"\(df\['([^']+)'\]\s*\+\s*df\['([^']+)'\]\)\.mean\(\)"
        sum_mean_match = re.search(sum_mean_pattern, calc_str)
        if sum_mean_match:
            col1, col2 = sum_mean_match.groups()
            if col1 not in df.columns or col2 not in df.columns:
                raise ValueError(f"Column '{col1}' or '{col2}' not found")
            return (df[col1] + df[col2]).mean()
            
        # NEW PATTERN: Sum of two columns - df['col1'] + df['col2']
        sum_cols_pattern = r"df\['([^']+)'\]\s*\+\s*df\['([^']+)'\]"
        sum_cols_match = re.search(sum_cols_pattern, calc_str)
        if sum_cols_match:
            col1, col2 = sum_cols_match.groups()
            if col1 not in df.columns or col2 not in df.columns:
                raise ValueError(f"Column '{col1}' or '{col2}' not found")
            return df[col1] + df[col2]
        
        # Pattern 1: Simple column operations - df['column'].operation()
        pattern1 = r"df\['([^']+)'\]\.(\w+)\(\)"
        match1 = re.match(pattern1, calc_str)
        if match1:
            column, operation = match1.groups()
            if column not in df.columns:
                raise ValueError(f"Column '{column}' not found")
            
            operations = {
                'sum': lambda col: df[col].sum(),
                'mean': lambda col: df[col].mean(),
                'count': lambda col: df[col].count(),
                'max': lambda col: df[col].max(),
                'min': lambda col: df[col].min(),
                'std': lambda col: df[col].std(),
                'nunique': lambda col: df[col].nunique(),
                'median': lambda col: df[col].median()
            }
            
            if operation in operations:
                return operations[operation](column)
        
        # Pattern 2: Condition-based calculations - df[df['col']=='value'].shape[0]
        condition_count_pattern = r"df\[df\['([^']+)'\]\s*==\s*'([^']+)'\]\.shape\[0\]"
        condition_match = re.search(condition_count_pattern, calc_str)
        if condition_match:
            column, value = condition_match.groups()
            if column not in df.columns:
                raise ValueError(f"Column '{column}' not found")
            return len(df[df[column] == value])
        
        # Pattern 3: Percentage with condition - df[df['col']=='value'].shape[0] / df.shape[0] * 100
        percentage_pattern = r"df\[df\['([^']+)'\]\s*==\s*'([^']+)'\]\.shape\[0\]\s*/\s*df\.shape\[0\]\s*\*\s*100"
        percentage_match = re.search(percentage_pattern, calc_str)
        if percentage_match:
            column, value = percentage_match.groups()
            if column not in df.columns:
                raise ValueError(f"Column '{column}' not found")
            
            total_count = len(df)
            if total_count == 0:
                return 0
            filtered_count = len(df[df[column] == value])
            return (filtered_count / total_count) * 100
        
        # Pattern 4: Alternative percentage format - len(df[df['col']=='value']) / len(df) * 100
        alt_percentage_pattern = r"len\(df\[df\['([^']+)'\]==?'([^']+)'\]\)\s*/\s*len\(df\)\s*\*\s*100"
        alt_percentage_match = re.search(alt_percentage_pattern, calc_str)
        if alt_percentage_match:
            column, value = alt_percentage_match.groups()
            if column not in df.columns:
                raise ValueError(f"Column '{column}' not found")
            
            total_count = len(df)
            if total_count == 0:
                return 0
            filtered_count = len(df[df[column] == value])
            return (filtered_count / total_count) * 100
        
        # Pattern 5: Growth rate calculations
        growth_pattern = r"\(df\['([^']+)'\]\.iloc\[-1\]\s*-\s*df\['([^']+)'\]\.iloc\[0\]\)\s*/\s*df\['([^']+)'\]\.iloc\[0\]\s*\*\s*100"
        growth_match = re.search(growth_pattern, calc_str)
        if growth_match:
            col1, col2, col3 = growth_match.groups()
            if col1 not in df.columns:
                raise ValueError(f"Column '{col1}' not found")
            if len(df) < 2:
                return 0
            first_val = df[col1].iloc[0]
            last_val = df[col1].iloc[-1]
            if first_val == 0:
                return 0
            return ((last_val - first_val) / first_val) * 100
        
        # Pattern 6: Count distinct values
        distinct_pattern = r"df\['([^']+)'\]\.nunique\(\)"
        distinct_match = re.match(distinct_pattern, calc_str)
        if distinct_match:
            column = distinct_match.group(1)
            if column not in df.columns:
                raise ValueError(f"Column '{column}' not found")
            return df[column].nunique()
        
        # Pattern 7: Groupby operations
        groupby_pattern = r"df\.groupby\('([^']+)'\)\['([^']+)'\]\.(\w+)\(\)"
        groupby_match = re.match(groupby_pattern, calc_str)
        if groupby_match:
            group_col, value_col, operation = groupby_match.groups()
            if group_col not in df.columns or value_col not in df.columns:
                raise ValueError(f"Column not found")
            
            grouped = df.groupby(group_col)[value_col]
            if operation == 'sum':
                return grouped.sum().to_dict()
            elif operation == 'mean':
                return grouped.mean().to_dict()
            elif operation == 'count':
                return grouped.count().to_dict()
        
        # Pattern 8: Direct division for rates
        division_pattern = r"df\['([^']+)'\]\.sum\(\)\s*/\s*df\['([^']+)'\]\.sum\(\)"
        division_match = re.search(division_pattern, calc_str)
        if division_match:
            col1, col2 = division_match.groups()
            if col1 not in df.columns or col2 not in df.columns:
                raise ValueError("Column not found")
            denominator = df[col2].sum()
            if denominator == 0:
                return 0
            return df[col1].sum() / denominator
        
        # If nothing matches, return a safe default
        st.warning(f"Unsupported calculation: {calc_str}")
        return 0
        
    except Exception as e:
        st.warning(f"Calculation error for '{calc_str}': {str(e)}")
        return 0

def format_business_metric(value: Any, format_type: str = "number") -> str:
    """Format metrics for business users"""
    try:
        if isinstance(value, dict):
            # For top N results
            items = list(value.items())[:3]  # Show top 3
            return ", ".join([f"{k}: ${v:,.0f}" if format_type == "currency" else f"{k}: {v:,.0f}" for k, v in items])
        
        if pd.isna(value) or value is None:
            return "N/A"
        
        if format_type == "currency":
            if value >= 1000000000:
                return f"${value/1000000000:.1f}B"
            elif value >= 1000000:
                return f"${value/1000000:.1f}M"
            elif value >= 1000:
                return f"${value/1000:.1f}K"
            else:
                return f"${value:,.0f}"
        
        elif format_type == "percentage":
            if value <= 1:
                return f"{value*100:.1f}%"
            else:
                return f"{value:.1f}%"
        
        elif format_type == "percentage_direct":
            return f"{value:.1f}%"
        
        elif format_type == "number":
            if value >= 1000000000:
                return f"{value/1000000000:.1f}B"
            elif value >= 1000000:
                return f"{value/1000000:.1f}M"
            elif value >= 1000:
                return f"{value/1000:.1f}K"
            else:
                return f"{value:,.0f}"
        
        elif format_type == "decimal":
            return f"{value:.2f}"
        
        else:
            return str(value)
            
    except:
        return str(value)

# ================================
# CLAUDE 3.7 API INTEGRATION
# ================================

def initialize_claude_client():
    """Initialize Claude 3.7 API client with Streamlit secrets and .env support"""
    api_key = None
    source = ""
    
    # Try multiple sources for API key
    try:
        api_key = st.secrets["CLAUDE_API_KEY"]
        source = "Streamlit secrets"
    except:
        pass
    
    if not api_key:
        api_key = os.getenv("CLAUDE_API_KEY")
        if api_key:
            source = "Environment variable"
    
    if not api_key:
        api_key = os.environ.get("CLAUDE_API_KEY") 
        if api_key:
            source = ".env file"
    
    if not api_key:
        st.sidebar.info("No Claude API key found. Using smart fallback analysis.")
        st.sidebar.write("**To enable Claude AI features:**")
        st.sidebar.write("• **Streamlit Cloud**: Add `CLAUDE_API_KEY` in Secrets")
        st.sidebar.write("• **Local Development**: Set in `.env` file:")
        st.sidebar.code("CLAUDE_API_KEY=sk-ant-api03-your_key_here")
        return None
    
    try:
        headers = {
            'x-api-key': api_key,
            'Content-Type': 'application/json',
            'anthropic-version': '2023-06-01'
        }
        
        test_payload = {
            "model": "claude-3-7-sonnet-20250219",
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "Hi"}]
        }
        
        response = requests.post(
            'https://api.anthropic.com/v1/messages',
            headers=headers,
            json=test_payload,
            timeout=10
        )
        
        if response.status_code == 200:
            st.sidebar.success(f"✅ API key available from {source}")
            return {"api_key": api_key, "headers": headers}
        else:
            st.sidebar.warning(f"⚠️ Claude API issue (Status: {response.status_code}). Using smart fallback.")
            return None
            
    except Exception as e:
        st.sidebar.warning(f"⚠️ Claude API unavailable: {str(e)[:50]}... Using smart fallback.")
        return None

def call_claude_api(client_config, prompt, max_tokens=1500, temperature=0.3):
    """Call Claude API with proper formatting"""
    if not client_config:
        return {"error": "No valid Claude client configuration"}
    
    payload = {
        "model": "claude-3-7-sonnet-20250219",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}]
    }
    
    try:
        response = requests.post(
            'https://api.anthropic.com/v1/messages',
            headers=client_config["headers"],
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            response_data = response.json()
            if "content" in response_data and len(response_data["content"]) > 0:
                content = response_data["content"][0]["text"]
                return {"success": True, "content": content}
            else:
                return {"error": "Unexpected response format"}
        else:
            return {"error": f"API error {response.status_code}: {response.text[:100]}"}
            
    except Exception as e:
        return {"error": f"Request failed: {str(e)}"}

# Initialize client at startup
try:
    claude_client = initialize_claude_client()
except Exception as e:
    st.sidebar.error(f"❌ Error initializing Claude AI: {str(e)}")
    claude_client = None

# ================================
# DATA PROCESSING FUNCTIONS
# ================================

@st.cache_data
def intelligent_data_cleaning(df):
    """Smart data cleaning with type detection and conversion"""
    cleaned_df = df.copy()
    
    # Clean column names
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
            
            # Skip if no data
            if sample.empty:
                continue
            
            # Currency detection
            if sample.str.contains(r'[$£€¥₹]', regex=True).any():
                cleaned_df[col] = (cleaned_df[col].astype(str)
                                 .str.replace(r'[$£€¥₹,]', '', regex=True)
                                 .replace('', np.nan))
                cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
            
            # Percentage detection
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

# ================================
# ENHANCED VISUALIZATION ENGINE
# ================================

def create_enhanced_visualization(df, viz_config):
    """Create enhanced business visualizations with improved reliability"""
    try:
        chart_type = viz_config.get("type", "bar")
        title = viz_config.get("title", "Chart")
        x_col = viz_config.get("x")
        y_col = viz_config.get("y")
        
        # Debug info
        print(f"Creating visualization: {title}, type: {chart_type}, x: {x_col}, y: {y_col}")
        
        # Column validation with intelligent fallbacks
        if not x_col or x_col not in df.columns:
            # Try to find a suitable replacement
            if x_col:
                possible_x_cols = [col for col in df.columns if x_col.lower() in col.lower()]
                if possible_x_cols:
                    x_col = possible_x_cols[0]
                    print(f"Substituted x column with {x_col}")
                else:
                    # If no match, use the first appropriate column based on chart type
                    if chart_type in ['bar', 'pie']:
                        categorical_cols = [col for col in df.columns if df[col].dtype == 'object' and df[col].nunique() < 50]
                        if categorical_cols:
                            x_col = categorical_cols[0]
                            print(f"Using fallback categorical column: {x_col}")
                        else:
                            print(f"No suitable x column found for {title}")
                            return None
                    elif chart_type in ['line', 'scatter', 'histogram']:
                        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                        if numeric_cols:
                            x_col = numeric_cols[0]
                            print(f"Using fallback numeric column: {x_col}")
                        else:
                            print(f"No suitable x column found for {title}")
                            return None
            else:
                print(f"No x column specified for {title}")
                return None
            
        if y_col and y_col not in df.columns:
            # Try to find a suitable replacement for y column
            possible_y_cols = [col for col in df.columns if y_col and y_col.lower() in col.lower()]
            if possible_y_cols:
                y_col = possible_y_cols[0]
                print(f"Substituted y column with {y_col}")
            else:
                # Use the first numeric column as fallback
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    y_col = numeric_cols[0]
                    print(f"Using fallback numeric column for y: {y_col}")
                else:
                    # For charts that need y values, if we can't find one, default to count
                    y_col = None
                    print(f"No suitable y column found, will use count instead")
        
        # Enhanced color scheme
        colors = px.colors.qualitative.Plotly
        
        # Handle datetime columns specially
        if x_col in df.columns and (pd.api.types.is_datetime64_any_dtype(df[x_col]) or df[x_col].dtype == 'object'):
            # Try to convert to datetime if it's an object
            try:
                if df[x_col].dtype == 'object':
                    df[x_col] = pd.to_datetime(df[x_col], errors='coerce')
            except:
                pass
        
        # For time-based grouping (hour of day analysis)
        if x_col in df.columns and 'hour' in title.lower() and pd.api.types.is_datetime64_any_dtype(df[x_col]):
            # Extract hour from datetime
            hour_df = df.copy()
            hour_df['hour'] = hour_df[x_col].dt.hour
            
            if y_col and y_col in df.columns and pd.api.types.is_numeric_dtype(df[y_col]):
                agg_df = hour_df.groupby('hour')[y_col].mean().reset_index()
                fig = px.line(agg_df, x='hour', y=y_col, title=title,
                             markers=True)
            else:
                agg_df = hour_df.groupby('hour').size().reset_index(name='count')
                fig = px.line(agg_df, x='hour', y='count', title=title,
                             markers=True)
            
            fig.update_xaxis(tickmode='linear', tick0=0, dtick=1)
            fig.update_layout(xaxis_title="Hour of Day")
            
            # Enhanced styling
            fig.update_layout(
                title_font_size=18,
                title_font_color='#2E4057',
                margin=dict(t=80, b=60, l=60, r=60),
                height=450,
                template="plotly_white",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=12),
                hovermode='x unified'
            )
            
            return fig
        
        # Rest of the visualization logic continues as before...
        if chart_type == "bar":
            if y_col and y_col in df.columns and pd.api.types.is_numeric_dtype(df[y_col]):
                # Aggregate data to avoid duplicates
                agg_df = df.groupby(x_col)[y_col].sum().reset_index()
                agg_df = agg_df.nlargest(15, y_col)  # Top 15 values
                
                if agg_df.empty:
                    # Try count instead if sum gives empty result
                    agg_df = df.groupby(x_col).size().reset_index(name=y_col)
                    agg_df = agg_df.nlargest(15, y_col)
                    if agg_df.empty:
                        return None
                
                fig = px.bar(agg_df, x=x_col, y=y_col, title=title,
                            color=y_col, color_continuous_scale='Blues')
                
                # Add value labels on bars
                fig.update_traces(texttemplate='%{y:,.0f}', textposition='outside')
            else:
                # Count occurrences
                counts = df[x_col].value_counts().head(15)
                if counts.empty:
                    return None
                
                fig = px.bar(x=counts.index, y=counts.values, title=title,
                            color=counts.values, color_continuous_scale='Viridis')
                fig.update_layout(xaxis_title=x_col, yaxis_title="Count")
                fig.update_traces(texttemplate='%{y:,.0f}', textposition='outside')
                
        elif chart_type == "line":
            if y_col and y_col in df.columns and pd.api.types.is_numeric_dtype(df[y_col]):
                if pd.api.types.is_datetime64_any_dtype(df[x_col]):
                    # Aggregate by date
                    agg_df = df.groupby(pd.Grouper(key=x_col, freq='D'))[y_col].sum().reset_index()
                    if agg_df.empty:
                        # Try different frequency if daily is empty
                        agg_df = df.groupby(pd.Grouper(key=x_col, freq='M'))[y_col].sum().reset_index()
                        if agg_df.empty:
                            return None
                    
                    fig = px.line(agg_df, x=x_col, y=y_col, title=title,
                                 markers=True)
                    
                    # Add trend line only if enough points
                    if len(agg_df) > 7:
                        fig.add_trace(go.Scatter(
                            x=agg_df[x_col],
                            y=agg_df[y_col].rolling(window=min(7, len(agg_df)//2 or 1)).mean(),
                            mode='lines',
                            name='Moving Average',
                            line=dict(color='red', dash='dash')
                        ))
                else:
                    clean_df = df[[x_col, y_col]].dropna().sort_values(x_col)
                    if clean_df.empty:
                        return None
                    fig = px.line(clean_df, x=x_col, y=y_col, title=title,
                                 markers=True)
            else:
                # If y is not available, try doing a count line chart
                if pd.api.types.is_datetime64_any_dtype(df[x_col]):
                    # For date columns, count by day
                    counts = df.groupby(pd.Grouper(key=x_col, freq='D')).size().reset_index(name='count')
                    if counts.empty:
                        counts = df.groupby(pd.Grouper(key=x_col, freq='M')).size().reset_index(name='count')
                    fig = px.line(counts, x=x_col, y='count', title=title, markers=True)
                else:
                    # For non-date columns, just count values
                    counts = df[x_col].value_counts().sort_index()
                    if counts.empty:
                        return None
                    fig = px.line(x=counts.index, y=counts.values, title=title, markers=True)
                fig.update_layout(xaxis_title=x_col, yaxis_title="Count")
                
        elif chart_type == "scatter":
            if y_col and y_col in df.columns:
                # Try to ensure both columns are numeric for scatter plot
                x_numeric = pd.api.types.is_numeric_dtype(df[x_col])
                y_numeric = pd.api.types.is_numeric_dtype(df[y_col])
                
                if not x_numeric and df[x_col].dtype == 'object':
                    # Try to convert string to numeric if possible
                    try:
                        df[f'{x_col}_numeric'] = pd.to_numeric(df[x_col].str.replace('[^0-9.]', '', regex=True), errors='coerce')
                        x_col = f'{x_col}_numeric'
                        x_numeric = True
                    except:
                        pass
                
                if not y_numeric and df[y_col].dtype == 'object':
                    # Try to convert string to numeric if possible
                    try:
                        df[f'{y_col}_numeric'] = pd.to_numeric(df[y_col].str.replace('[^0-9.]', '', regex=True), errors='coerce')
                        y_col = f'{y_col}_numeric'
                        y_numeric = True
                    except:
                        pass
                
                if x_numeric and y_numeric:
                    clean_df = df[[x_col, y_col]].dropna()
                    if clean_df.empty or len(clean_df) < 5:  # Need enough points for scatter
                        return None
                    
                    # Sample if too many points
                    if len(clean_df) > 5000:
                        clean_df = clean_df.sample(5000)
                    
                    # Add trend line
                    try:
                        fig = px.scatter(clean_df, x=x_col, y=y_col, title=title,
                                       trendline="ols", trendline_color_override="red")
                    except:
                        # If trendline fails, create without it
                        fig = px.scatter(clean_df, x=x_col, y=y_col, title=title)
                else:
                    # If can't make scatter, fall back to bar chart
                    if y_numeric:
                        agg_df = df.groupby(x_col)[y_col].mean().reset_index()
                        agg_df = agg_df.nlargest(15, y_col)
                        fig = px.bar(agg_df, x=x_col, y=y_col, title=f"{title} (Bar Chart)", 
                                     color=y_col, color_continuous_scale='Blues')
                    else:
                        return None
            else:
                return None
                
        elif chart_type == "pie":
            # Make pie chart more reliable by handling edge cases
            try:
                counts = df[x_col].value_counts().head(8)
                if counts.empty or counts.sum() == 0:
                    return None
                
                # Add "Others" category if needed
                if len(df[x_col].value_counts()) > 8:
                    others_count = df[x_col].value_counts()[8:].sum()
                    counts['Others'] = others_count
                
                # Handle potentially problematic labels
                labels = counts.index.tolist()
                labels = [str(label)[:20] + ('...' if len(str(label)) > 20 else '') for label in labels]
                
                fig = px.pie(values=counts.values, names=labels, title=title,
                            hole=0.3)  # Donut chart
                
                # Add percentages
                fig.update_traces(textposition='inside', textinfo='percent+label')
            except Exception as e:
                print(f"Error creating pie chart: {e}")
                return None
            
        elif chart_type == "histogram":
            if pd.api.types.is_numeric_dtype(df[x_col]):
                clean_df = df[x_col].dropna()
                if clean_df.empty or clean_df.nunique() < 2:
                    return None
                
                # Automatically determine appropriate bin count
                bin_count = min(20, max(5, clean_df.nunique() // 2))
                
                fig = px.histogram(df, x=x_col, title=title, nbins=bin_count,
                                 color_discrete_sequence=['#1f77b4'])
                
                # Add mean line
                mean_val = clean_df.mean()
                fig.add_vline(x=mean_val, line_dash="dash", line_color="red",
                            annotation_text=f"Mean: {mean_val:.1f}")
            else:
                counts = df[x_col].value_counts().head(15)
                if counts.empty:
                    return None
                fig = px.bar(x=counts.index, y=counts.values, title=title)
                fig.update_layout(xaxis_title=x_col, yaxis_title="Count")
        
        else:
            # Default to bar chart
            counts = df[x_col].value_counts().head(15)
            if counts.empty:
                return None
            fig = px.bar(x=counts.index, y=counts.values, title=title)
            fig.update_layout(xaxis_title=x_col, yaxis_title="Count")
        
        # Enhanced styling for all charts
        fig.update_layout(
            title_font_size=18,
            title_font_color='#2E4057',
            margin=dict(t=80, b=60, l=60, r=60),
            height=450,
            showlegend=True if chart_type in ["pie", "line"] else False,
            template="plotly_white",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=12),
            hovermode='x unified' if chart_type == 'line' else 'closest'
        )
        
        # Improve grid lines
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#E8E8E8')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#E8E8E8')
        
        # Rotate x-axis labels if needed
        if chart_type in ["bar", "histogram"] and len(str(fig.data[0].x[0] if hasattr(fig.data[0], 'x') and len(fig.data[0].x) > 0 else '')) > 10:
            fig.update_xaxes(tickangle=-45)
            
        return fig
        
    except Exception as e:
        st.warning(f"Error creating visualization '{viz_config.get('title', 'Unknown')}': {str(e)}")
        # Return simple fallback chart if possible
        try:
            # Create a simple bar chart as fallback
            if 'x' in viz_config and viz_config['x'] in df.columns:
                x_col = viz_config['x']
                counts = df[x_col].value_counts().head(10)
                if not counts.empty:
                    fig = px.bar(
                        x=counts.index, 
                        y=counts.values, 
                        title=f"{viz_config.get('title', 'Chart')} (Fallback)"
                    )
                    fig.update_layout(height=400)
                    return fig
        except:
            pass
        return None

# ================================
# AI-DRIVEN ANALYSIS ENGINE
# ================================

def generate_ai_driven_analysis(df):
    """Claude AI suggests business-relevant KPIs and visualizations"""
    
    # Get basic data info for AI
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [col for col in df.columns if df[col].dtype == 'object' and df[col].nunique() < 50]
    date_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
    
    # Sample data for AI context (limit size for API)
    sample_data = df.head(3).to_dict('records')
    sample_str = str(sample_data)[:500] + "..." if len(str(sample_data)) > 500 else str(sample_data)
    
    if claude_client:
        prompt = f"""You are a business intelligence expert helping non-technical business users. 
Analyze this dataset and suggest SIMPLE, BUSINESS-RELEVANT KPIs and visualizations.

Dataset Info:
- Columns: {df.columns.tolist()[:20]}
- Numeric columns: {numeric_cols[:10]}
- Categorical columns: {categorical_cols[:10]}  
- Date columns: {date_cols[:5]}
- Sample data: {sample_str}
- Total rows: {df.shape[0]}
- IMPORTANT: These columns appear to be IDs or unique identifiers (not metrics)

Focus on BUSINESS METRICS that matter to executives, sales teams, and managers.
Avoid technical or statistical metrics. Think revenue, growth, top performers, trends.

CRITICAL RULES:
1. For numeric ID fields (columns that appear to be identifiers or have "id", "number", "transcript" in the name), use DISTINCT COUNT operations (nunique) instead of SUM/MEAN
2. Only suggest KPIs/visualizations using columns that actually exist
3. Use the exact correct column names from the dataset info above
4. Keep calculations simple and reliable
5. Do not use columns like 'transcript_id', 'chat_id', etc. as metrics - these should be counted as unique values
6. Suggested visualizations must match the data types - never use line charts for categorical data without a numeric component


Respond in this EXACT JSON format:
{{
    "suggested_kpis": [
        {{
            "name": "Total Revenue",
            "calculation": "df['revenue'].sum()",
            "format": "currency",
            "priority": 1,
            "description": "Total revenue across all transactions"
        }},
        {{
            "name": "Customer Count", 
            "calculation": "df['customer_id'].nunique()",
            "format": "number",
            "priority": 2,
            "description": "Number of unique customers"
        }},
        {{
            "name": "Average Order Value",
            "calculation": "df['revenue'].mean()",
            "format": "currency",
            "priority": 3,
            "description": "Average revenue per transaction"
        }},
        {{
            "name": "Top Product Category",
            "calculation": "df.groupby('category')['revenue'].sum().nlargest(1)",
            "format": "text",
            "priority": 4,
            "description": "Best performing product category"
        }}
    ],
    "suggested_visualizations": [
        {{
            "title": "Revenue Trend Over Time",
            "type": "line",
            "x": "date",
            "y": "revenue",
            "priority": 1,
            "insight": "Shows revenue performance and trends"
        }},
        {{
            "title": "Top 10 Customers by Revenue",
            "type": "bar", 
            "x": "customer_name",
            "y": "revenue",
            "priority": 2,
            "insight": "Identifies most valuable customers"
        }},
        {{
            "title": "Sales by Category",
            "type": "pie",
            "x": "category",
            "y": null,
            "priority": 3,
            "insight": "Shows revenue distribution across categories"
        }},
        {{
            "title": "Monthly Growth Trend",
            "type": "line",
            "x": "month",
            "y": "revenue",
            "priority": 4,
            "insight": "Month-over-month performance"
        }}
    ],
    "business_insights": [
        "Clear, actionable finding about business performance",
        "Key opportunity or risk identified in the data",
        "Recommendation for business improvement"
    ]
}}

IMPORTANT: 
- Only suggest KPIs/visualizations using columns that actually exist
- Focus on money, counts, averages, and trends
- Make it relevant for business decision-makers
- Keep calculations simple and meaningful"""

        response = call_claude_api(claude_client, prompt, max_tokens=1500, temperature=0.3)
        
        if response.get("success"):
            try:
                # Extract JSON from response
                content = response["content"]
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    ai_suggestions = json.loads(json_match.group(0))
                    return ai_suggestions
            except json.JSONDecodeError as e:
                st.warning(f"AI response parsing error: {e}")
    
    # FALLBACK: Smart business-focused suggestions
    return generate_business_fallback_suggestions(df, numeric_cols, categorical_cols, date_cols)

def generate_business_fallback_suggestions(df, numeric_cols, categorical_cols, date_cols):
    """Generate business-focused suggestions when Claude AI is unavailable"""
    
    suggested_kpis = []
    suggested_visualizations = []
    business_insights = []
    
    # Always add total records
    suggested_kpis.append({
        "name": "Total Records",
        "calculation": "len(df)",
        "format": "number",
        "priority": 1,
        "description": "Total number of entries in dataset"
    })
    
    # Look for business-relevant numeric columns
    money_cols = []
    count_cols = []
    score_cols = []
    
    for col in numeric_cols:
        col_lower = col.lower()
        if any(term in col_lower for term in ['revenue', 'sales', 'amount', 'value', 'price', 'cost', 'profit', 'spend']):
            money_cols.append(col)
        elif any(term in col_lower for term in ['count', 'quantity', 'qty', 'number', 'volume']):
            count_cols.append(col)
        elif any(term in col_lower for term in ['score', 'rating', 'satisfaction', 'nps']):
            score_cols.append(col)
    
    # Add money-related KPIs
    if money_cols:
        primary_money = money_cols[0]
        suggested_kpis.extend([
            {
                "name": f"Total {primary_money.replace('_', ' ').title()}",
                "calculation": f"df['{primary_money}'].sum()",
                "format": "currency",
                "priority": 2,
                "description": f"Sum of all {primary_money}"
            },
            {
                "name": f"Average {primary_money.replace('_', ' ').title()}",
                "calculation": f"df['{primary_money}'].mean()",
                "format": "currency",
                "priority": 3,
                "description": f"Average {primary_money} per record"
            }
        ])
        
        # Add month-over-month growth if we have dates
        if date_cols:
            suggested_kpis.append({
                "name": "Growth Rate",
                "calculation": f"(df['{primary_money}'].iloc[-1] - df['{primary_money}'].iloc[0]) / df['{primary_money}'].iloc[0] * 100",
                "format": "percentage",
                "priority": 4,
                "description": "Overall growth from first to last record"
            })
    
    # Add customer/category insights
    if categorical_cols:
        # Find the best categorical column (not too many unique values)
        best_cat = None
        for col in categorical_cols:
            unique_count = df[col].nunique()
            if 2 <= unique_count <= 20:
                col_lower = col.lower()
                if any(term in col_lower for term in ['customer', 'client', 'user', 'account']):
                    best_cat = col
                    break
                elif any(term in col_lower for term in ['product', 'category', 'type', 'segment']):
                    best_cat = col
        
        if best_cat:
            suggested_kpis.append({
                "name": f"Unique {best_cat.replace('_', ' ').title()}",
                "calculation": f"df['{best_cat}'].nunique()",
                "format": "number",
                "priority": 5,
                "description": f"Number of unique {best_cat}"
            })
    
    # Score/Rating KPIs
    if score_cols:
        primary_score = score_cols[0]
        suggested_kpis.append({
            "name": f"Average {primary_score.replace('_', ' ').title()}",
            "calculation": f"df['{primary_score}'].mean()",
            "format": "decimal",
            "priority": 6,
            "description": f"Average {primary_score}"
        })
    
    # Business insights
    business_insights.append(f"Dataset contains {df.shape[0]:,} records across {df.shape[1]} columns")
    
    if money_cols and df[money_cols[0]].sum() > 0:
        total_value = df[money_cols[0]].sum()
        business_insights.append(f"Total {money_cols[0]} value: {format_business_metric(total_value, 'currency')}")
    
    completeness = (df.notna().sum().sum() / (df.shape[0] * df.shape[1])) * 100
    business_insights.append(f"Data quality score: {completeness:.1f}% complete")
    
    # Visualization suggestions
    if date_cols and numeric_cols:
        suggested_visualizations.append({
            "title": f"{numeric_cols[0].replace('_', ' ').title()} Over Time",
            "type": "line",
            "x": date_cols[0],
            "y": numeric_cols[0],
            "priority": 1,
            "insight": "Trend analysis over time"
        })
    
    if categorical_cols and money_cols:
        best_cat = min(categorical_cols, key=lambda x: df[x].nunique() if df[x].nunique() > 1 else float('inf'))
        suggested_visualizations.append({
            "title": f"Top {best_cat.replace('_', ' ').title()} by {money_cols[0].replace('_', ' ').title()}",
            "type": "bar",
            "x": best_cat,
            "y": money_cols[0],
            "priority": 2,
            "insight": "Identify top performers"
        })
    
    if categorical_cols:
        cat_col = categorical_cols[0]
        if df[cat_col].nunique() <= 10:
            suggested_visualizations.append({
                "title": f"Distribution by {cat_col.replace('_', ' ').title()}",
                "type": "pie",
                "x": cat_col,
                "y": None,
                "priority": 3,
                "insight": "Breakdown by category"
            })
    
    if numeric_cols:
        suggested_visualizations.append({
            "title": f"{numeric_cols[0].replace('_', ' ').title()} Distribution",
            "type": "histogram",
            "x": numeric_cols[0],
            "y": None,
            "priority": 4,
            "insight": "Value distribution pattern"
        })
    
    return {
        "suggested_kpis": suggested_kpis[:6],  # Limit to 6 KPIs
        "suggested_visualizations": suggested_visualizations[:4],  # Limit to 4 charts
        "business_insights": business_insights[:3]  # Limit to 3 insights
    }

def create_ai_driven_kpis(df, suggested_kpis):
    """Create KPIs based on AI suggestions with enhanced formatting"""
    created_kpis = []
    
    for kpi in suggested_kpis[:6]:  # Show up to 6 KPIs
        try:
            calc_str = kpi.get("calculation", "")
            if calc_str:
                result = execute_safe_calculation(calc_str, df)
                
                # Format based on AI's suggestion
                format_type = kpi.get("format", "number")
                display_value = format_business_metric(result, format_type)
                
                # Handle special cases
                if isinstance(result, dict) and result:
                    # For top N results, show the top item
                    top_key = list(result.keys())[0]
                    top_val = result[top_key]
                    if format_type == "currency":
                        display_value = f"{top_key}: {format_business_metric(top_val, 'currency')}"
                    else:
                        display_value = f"{top_key}: {format_business_metric(top_val, 'number')}"
                
                created_kpis.append({
                    "name": kpi.get("name", "Metric"),
                    "value": display_value,
                    "description": kpi.get("description", ""),
                    "priority": kpi.get("priority", 1),
                    "delta": None  # Could add period-over-period comparison
                })
                
        except Exception as e:
            st.warning(f"Error calculating KPI '{kpi.get('name', 'Unknown')}': {str(e)}")
    
    return sorted(created_kpis, key=lambda x: x.get("priority", 999))

def create_ai_driven_visualizations(df, suggested_visualizations):
    """Create visualizations based on AI suggestions with enhanced reliability"""
    created_charts = []
    
    # Get some data characteristics for fallbacks
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [col for col in df.columns if df[col].dtype == 'object' and df[col].nunique() < 50]
    date_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
    
    # Process AI suggested visualizations
    for viz in suggested_visualizations[:6]:  # Try up to 6 instead of 4
        try:
            x_col = viz.get("x")
            y_col = viz.get("y")
            
            # Skip charts that would definitely fail
            if not x_col:
                continue
                
            fig = create_enhanced_visualization(df, viz)
            if fig:
                created_charts.append({
                    "figure": fig,
                    "priority": viz.get("priority", 999),
                    "insight": viz.get("insight", "")
                })
        except Exception as e:
            st.warning(f"Error creating visualization '{viz.get('title', 'Unknown')}': {str(e)}")
    
    # If we don't have enough charts, add generic ones based on data characteristics
    if len(created_charts) < 4:
        st.info(f"Adding supplementary visualizations to complete your dashboard...")
        
        # Visualization 1: Distribution of first numeric column (if exists)
        if numeric_cols and len(created_charts) < 4:
            viz_config = {
                "title": f"Distribution of {numeric_cols[0].replace('_', ' ').title()}",
                "type": "histogram",
                "x": numeric_cols[0],
                "y": None
            }
            fig = create_enhanced_visualization(df, viz_config)
            if fig:
                created_charts.append({
                    "figure": fig,
                    "priority": 90,
                    "insight": f"Distribution pattern of {numeric_cols[0].replace('_', ' ').title()}"
                })
        
        # Visualization 2: Bar chart of categorical data
        if categorical_cols and len(created_charts) < 4:
            cat_col = categorical_cols[0]
            viz_config = {
                "title": f"{cat_col.replace('_', ' ').title()} Breakdown",
                "type": "bar",
                "x": cat_col,
                "y": None
            }
            fig = create_enhanced_visualization(df, viz_config)
            if fig:
                created_charts.append({
                    "figure": fig,
                    "priority": 91,
                    "insight": f"Count breakdown by {cat_col.replace('_', ' ').title()}"
                })
        
        # Visualization 3: Pie chart of another categorical (if exists)
        if len(categorical_cols) > 1 and len(created_charts) < 4:
            cat_col = categorical_cols[1]
            viz_config = {
                "title": f"{cat_col.replace('_', ' ').title()} Distribution",
                "type": "pie",
                "x": cat_col,
                "y": None
            }
            fig = create_enhanced_visualization(df, viz_config)
            if fig:
                created_charts.append({
                    "figure": fig,
                    "priority": 92,
                    "insight": f"Proportional analysis of {cat_col.replace('_', ' ').title()}"
                })
        
        # Visualization 4: Time series if date column exists
        if date_cols and numeric_cols and len(created_charts) < 4:
            date_col = date_cols[0]
            num_col = numeric_cols[0]
            viz_config = {
                "title": f"{num_col.replace('_', ' ').title()} Over Time",
                "type": "line",
                "x": date_col,
                "y": num_col
            }
            fig = create_enhanced_visualization(df, viz_config)
            if fig:
                created_charts.append({
                    "figure": fig,
                    "priority": 93,
                    "insight": f"Trend analysis of {num_col.replace('_', ' ').title()} over time"
                })
        
        # Visualization 5: Scatter plot if multiple numeric columns
        if len(numeric_cols) >= 2 and len(created_charts) < 4:
            viz_config = {
                "title": f"{numeric_cols[0].replace('_', ' ').title()} vs {numeric_cols[1].replace('_', ' ').title()}",
                "type": "scatter",
                "x": numeric_cols[0],
                "y": numeric_cols[1]
            }
            fig = create_enhanced_visualization(df, viz_config)
            if fig:
                created_charts.append({
                    "figure": fig,
                    "priority": 94,
                    "insight": f"Correlation between {numeric_cols[0].replace('_', ' ').title()} and {numeric_cols[1].replace('_', ' ').title()}"
                })
        
        # Visualization 6: More creative visualization with any remaining column combinations
        if categorical_cols and numeric_cols and len(created_charts) < 4:
            cat_col = categorical_cols[0]
            num_col = numeric_cols[-1] if len(numeric_cols) > 1 else numeric_cols[0]
            viz_config = {
                "title": f"Top 10 {cat_col.replace('_', ' ').title()} by {num_col.replace('_', ' ').title()}",
                "type": "bar",
                "x": cat_col,
                "y": num_col
            }
            fig = create_enhanced_visualization(df, viz_config)
            if fig:
                created_charts.append({
                    "figure": fig,
                    "priority": 95,
                    "insight": f"Top performers analysis by {num_col.replace('_', ' ').title()}"
                })
    
    # Sort by priority and return
    created_charts.sort(key=lambda x: x.get("priority", 999))
    return created_charts[:4]  # Return at most 4 charts

def run_ai_driven_analysis(df):
    """Main function that runs AI-driven analysis"""
    
    with st.spinner("🧠 AI is analyzing your data and creating business insights..."):
        ai_suggestions = generate_ai_driven_analysis(df)
    
    # Show AI status
    if claude_client:
        st.success("🤖 AI analysis complete! Business KPIs and visualizations ready.")
    else:
        st.info("📊 Smart analysis complete! Using intelligent business metrics ( AI unavailable)")
    
    # CREATE the AI-suggested KPIs
    if ai_suggestions.get("suggested_kpis"):
        st.write("### 📈 Key Business Metrics")
        
        created_kpis = create_ai_driven_kpis(df, ai_suggestions["suggested_kpis"])
        
        if created_kpis:
            # Display KPIs in a grid
            kpi_cols = st.columns(min(3, len(created_kpis)))
            for i, kpi in enumerate(created_kpis):
                with kpi_cols[i % len(kpi_cols)]:
                    # Enhanced metric display
                    delta = kpi.get("delta")
                    if delta:
                        st.metric(
                            kpi["name"],
                            kpi["value"],
                            delta=delta,
                            help=kpi["description"]
                        )
                    else:
                        st.metric(
                            kpi["name"],
                            kpi["value"],
                            help=kpi["description"]
                        )
    
    # CREATE the AI-suggested visualizations  
    if ai_suggestions.get("suggested_visualizations"):
        st.write("### 📊 Business Intelligence Dashboard")
        
        created_charts = create_ai_driven_visualizations(df, ai_suggestions["suggested_visualizations"])
        
        if created_charts:
            # Create 2-column layout for charts
            for i in range(0, len(created_charts), 2):
                cols = st.columns(2)
                for j in range(2):
                    if i + j < len(created_charts):
                        with cols[j]:
                            chart = created_charts[i + j]
                            st.plotly_chart(chart["figure"], use_container_width=True)
                            if chart.get("insight"):
                                st.caption(f"💡 {chart['insight']}")
    
    # SHOW AI business insights
    if ai_suggestions.get("business_insights"):
        st.write("### 💡 Executive Summary")
        for i, insight in enumerate(ai_suggestions["business_insights"], 1):
            st.info(f"**Finding {i}:** {insight}")
    
    return ai_suggestions

def generate_custom_analysis(df, user_prompt):
    """Generate custom business analysis using Claude AI"""
    
    if not claude_client:
        return {
            "custom_metrics": [],
            "custom_insights": ["Claude AI not available - please use the standard analysis features"],
            "custom_visualizations": []
        }
    
    # Prepare data context
    sample_rows = df.head(3).to_dict('records')
    sample_str = str(sample_rows)[:300] + "..." if len(str(sample_rows)) > 300 else str(sample_rows)
    
    col_info = {}
    for col in df.columns[:20]:  # Limit to 20 columns
        col_info[col] = {
            'dtype': str(df[col].dtype),
            'unique_count': df[col].nunique(),
            'sample_values': df[col].dropna().head(3).tolist()[:3]
        }
    
    custom_prompt = f"""You are a business analyst helping non-technical users. 
The user asked: "{user_prompt}"

Dataset context:
- Columns: {list(col_info.keys())}
- Column details: {col_info}
- Shape: {df.shape[0]} rows, {df.shape[1]} columns
- Sample data: {sample_str}

Provide SIMPLE BUSINESS ANALYSIS in this JSON format:
{{
    "custom_metrics": [
        {{
            "name": "Metric Name",
            "value": "df['column'].sum()",
            "description": "What this tells us about the business"
        }}
    ],
    "custom_insights": [
        "Clear business finding that answers the question",
        "Actionable recommendation"
    ],
    "custom_visualizations": [
        {{
            "title": "Chart Title",
            "type": "bar",
            "x": "column_name",
            "y": "value_column", 
            "description": "What this shows"
        }}
    ]
}}

Focus on money, counts, averages, and trends. Keep it simple for business users."""

    try:
        response = call_claude_api(claude_client, custom_prompt, max_tokens=1500, temperature=0.3)
        
        if response.get("success"):
            response_text = response["content"]
            
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(0))
                except json.JSONDecodeError as e:
                    st.error(f"Error parsing custom analysis: {e}")
                    return {
                        "custom_metrics": [],
                        "custom_insights": [f"Analysis error: {e}"],
                        "custom_visualizations": []
                    }
            else:
                return {
                    "custom_metrics": [],
                    "custom_insights": ["Could not parse AI response"],
                    "custom_visualizations": []
                }
        else:
            return {
                "custom_metrics": [],
                "custom_insights": [f"Claude AI error: {response.get('error', 'Unknown error')}"],
                "custom_visualizations": []
            }
            
    except Exception as e:
        return {
            "custom_metrics": [],
            "custom_insights": [f"Error: {str(e)}"],
            "custom_visualizations": []
        }

# ================================
# SESSION STATE INITIALIZATION
# ================================

if 'analysis_data' not in st.session_state:
    st.session_state.analysis_data = None
if 'analysis_type' not in st.session_state:
    st.session_state.analysis_type = None
if 'custom_metrics' not in st.session_state:
    st.session_state.custom_metrics = []

# ================================
# SIDEBAR
# ================================

with st.sidebar:
    st.title("🧠  AI Data Analyzer")
    st.markdown("""
    ### 🎯 Perfect for Business Teams
    - **Sales:** Revenue KPIs & trends
    - **Marketing:** Campaign performance  
    - **Operations:** Efficiency metrics
    - **Executive:** High-level insights
    
    ### 🚀 Key Features
    - **AI-Driven Business KPIs**
    - **Smart Visualizations**
    - **Custom Business Questions**
    - **Executive Summaries**
    
    ### 📁 Supported Files
    - CSV files (.csv)
    - Excel files (.xlsx, .xls)
    - Multiple sheets supported
    - Up to 200MB per file
    """)
    st.markdown("---")
    
    # Sample data options
    st.subheader("🎯 Try Sample Data")
    if st.button("📈 Sales Dashboard", use_container_width=True):
       np.random.seed(42)
       dates = pd.date_range('2024-01-01', periods=500, freq='D')
       sample_data = pd.DataFrame({
           'date': dates,
           'sales_rep': np.random.choice(['Alice Johnson', 'Bob Smith', 'Carol Davis', 'David Lee', 'Emma Wilson'], 500),
           'region': np.random.choice(['North', 'South', 'East', 'West'], 500),
           'revenue': np.random.normal(5000, 1500, 500).clip(min=0).round(2),
           'product_category': np.random.choice(['Software', 'Training', 'Support', 'Consulting'], 500),
           'deal_size': np.random.choice(['Small', 'Medium', 'Large', 'Enterprise'], 500, p=[0.4, 0.3, 0.2, 0.1]),
           'customer_type': np.random.choice(['New', 'Existing', 'Renewal'], 500, p=[0.3, 0.5, 0.2]),
           'profit_margin': np.random.uniform(0.15, 0.45, 500),
           'units_sold': np.random.randint(1, 50, 500)
       })
       st.session_state.analysis_data = {'Sales_Dashboard.csv': sample_data}
       st.session_state.analysis_type = 'single'
       st.rerun()
       
    if st.button("📊 Customer Analytics", use_container_width=True):
       np.random.seed(43)
       sample_data = pd.DataFrame({
           'customer_id': [f'CUST_{i:04d}' for i in range(1, 301)],
           'customer_name': [f'Company_{i}' for i in range(1, 301)],
           'satisfaction_score': np.random.choice([1, 2, 3, 4, 5], 300, p=[0.05, 0.1, 0.2, 0.4, 0.25]),
           'lifetime_value': np.random.lognormal(8, 1.5, 300).round(2),
           'industry': np.random.choice(['Tech', 'Finance', 'Healthcare', 'Retail', 'Manufacturing'], 300),
           'account_age_months': np.random.randint(1, 60, 300),
           'monthly_spend': np.random.lognormal(7.5, 1, 300).round(2),
           'support_tickets': np.random.poisson(3, 300),
           'churn_risk': np.random.choice(['Low', 'Medium', 'High'], 300, p=[0.6, 0.3, 0.1])
       })
       st.session_state.analysis_data = {'Customer_Analytics.csv': sample_data}
       st.session_state.analysis_type = 'single'
       st.rerun()

# ================================
# MAIN APPLICATION
# ================================

st.title("🔢 AI Business Intelligence Analyzer")
st.markdown("**Upload your data and get instant business insights powered by AI.**")

# Main choice - Single or Multi-file
if not st.session_state.analysis_data:
   st.markdown("### Choose your analysis type:")
   col1, col2 = st.columns(2)
   
   with col1:
       st.subheader("📄 Single File Analysis")
       st.write("Upload one file for focused business analysis")
       uploaded_file = st.file_uploader(
           "Choose a file",
           type=["csv", "xlsx", "xls"],
           key="single_file"
       )
       
       if uploaded_file:
           with st.spinner("Processing your data..."):
               try:
                   if uploaded_file.name.endswith('.csv'):
                       df = pd.read_csv(uploaded_file)
                       st.session_state.analysis_data = {uploaded_file.name: intelligent_data_cleaning(df)}
                       st.session_state.analysis_type = 'single'
                       st.rerun()
                   else:
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
       st.write("Compare data across multiple files")
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
                   for name, df in temp_data.items():
                       st.write(f"• **{name}** - {df.shape[0]:,} rows, {df.shape[1]} columns")
                   
                   selected_datasets = st.multiselect(
                       "Select datasets to analyze",
                       options=list(temp_data.keys()),
                       default=list(temp_data.keys())
                   )
                   
                   if selected_datasets and st.button("Start Multi-File Analysis", type="primary"):
                       selected_data = {name: temp_data[name] for name in selected_datasets}
                       if len(selected_data) == 1:
                           st.session_state.analysis_data = selected_data
                           st.session_state.analysis_type = 'single'
                       else:
                           st.session_state.analysis_data = selected_data
                           st.session_state.analysis_type = 'multi'
                       st.rerun()

# ================================
# ANALYSIS RESULTS
# ================================

elif st.session_state.analysis_data:
   data = st.session_state.analysis_data
   analysis_type = st.session_state.analysis_type
   
   # Back button
   col1, col2 = st.columns([6, 1])
   with col1:
       if st.button("⬅️ Back to Upload"):
           st.session_state.analysis_data = None
           st.session_state.analysis_type = None
           st.session_state.custom_metrics = []
           st.rerun()
   with col2:
       if st.button("🔄 Refresh Analysis"):
           st.session_state.custom_metrics = []
           st.rerun()

   if analysis_type == 'single':
       # ================================
       # SINGLE FILE AI-DRIVEN ANALYSIS
       # ================================
       file_name = list(data.keys())[0]
       df = list(data.values())[0]
       st.subheader(f"📊 Business Intelligence Report: {file_name}")
       
       # Display basic data info with business context
       col1, col2, col3, col4 = st.columns(4)
       with col1:
           st.metric("📋 Total Records", f"{df.shape[0]:,}")
       with col2:
           st.metric("📊 Data Points", f"{df.shape[1]}")
       with col3:
           completeness = (df.notna().sum().sum() / (df.shape[0] * df.shape[1])) * 100
           st.metric("✅ Data Quality", f"{completeness:.1f}%")
       with col4:
           numeric_cols = df.select_dtypes(include=[np.number]).columns
           st.metric("🔢 Metrics Available", f"{len(numeric_cols)}")

       # RUN AI-DRIVEN ANALYSIS
       ai_analysis = run_ai_driven_analysis(df)

       # ================================
       # CUSTOM AI ANALYSIS SECTION
       # ================================
       st.write("### 🎯 Ask Business Questions")
       with st.expander("Get custom insights from AI", expanded=False):
           col1, col2 = st.columns([3, 1])
           with col1:
               user_prompt = st.text_area(
                   "What business insights do you need?",
                   placeholder="Examples:\n"
                               "- What's our best performing product?\n"
                               "- Which customers generate the most revenue?\n"
                               "- Show me sales trends by region\n"
                               "- What's our customer retention rate?\n"
                               "- Identify growth opportunities",
                   height=120
               )
           with col2:
               st.write("")
               st.write("")
               analyze_button = st.button("🧠 Ask Claude", type="primary", use_container_width=True)
           
           # AI CUSTOM ANALYSIS
           if analyze_button and user_prompt:
               with st.spinner("🤖 Claude is analyzing your business question..."):
                   custom_analysis = generate_custom_analysis(df, user_prompt)
                   
                   # Display custom metrics
                   if custom_analysis.get("custom_metrics"):
                       st.write("#### 📊 Answer to Your Question")
                       custom_cols = st.columns(min(3, len(custom_analysis["custom_metrics"][:3])))
                       
                       new_custom_metrics = []
                       for i, metric in enumerate(custom_analysis["custom_metrics"][:3]):
                           with custom_cols[i]:
                               try:
                                   calc_str = metric.get("value", "")
                                   
                                   if calc_str.startswith("df"):
                                       result = execute_safe_calculation(calc_str, df)
                                       
                                       # Smart formatting based on metric name
                                       metric_name = metric["name"].lower()
                                       if any(term in metric_name for term in ['revenue', 'sales', 'cost', 'price', 'value', 'spend']):
                                           format_type = "currency"
                                       elif any(term in metric_name for term in ['rate', 'percentage', '%']):
                                           format_type = "percentage"
                                       else:
                                           format_type = "number"
                                       
                                       display_value = format_business_metric(result, format_type)
                                   else:
                                       display_value = metric.get("value", "N/A")
                                   
                                   st.metric(
                                       metric["name"],
                                       display_value,
                                       help=metric.get("description", "")
                                   )
                                   
                                   new_custom_metrics.append({
                                       "name": metric["name"],
                                       "value": display_value,
                                       "description": metric.get("description", "")
                                   })
                                   
                               except Exception as e:
                                   st.metric(
                                       metric["name"],
                                       "Error",
                                       help=f"Calculation error: {str(e)}"
                                   )
                       
                       st.session_state.custom_metrics = new_custom_metrics
                   
                   # Display custom insights
                   if custom_analysis.get("custom_insights"):
                       st.write("#### 💡 Business Insights")
                       for insight in custom_analysis["custom_insights"]:
                           st.success(f"🤖 {insight}")
                   
                   # Create custom visualizations
                   if custom_analysis.get("custom_visualizations"):
                       st.write("#### 📈 Visualizations")
                       viz_cols = st.columns(min(2, len(custom_analysis["custom_visualizations"])))
                       for i, viz in enumerate(custom_analysis["custom_visualizations"][:4]):
                           with viz_cols[i % len(viz_cols)]:
                               custom_fig = create_enhanced_visualization(df, viz)
                               if custom_fig:
                                   st.plotly_chart(custom_fig, use_container_width=True)
                                   if viz.get("description"):
                                       st.caption(f"💡 {viz['description']}")

       # ================================
       # DATA EXPLORATION SECTION
       # ================================
       st.write("### 📋 Data Explorer")
       with st.expander("🔍 Explore Your Data", expanded=True):
           view_option = st.radio(
               "View type",
               ["Executive Summary", "Detailed Analysis", "Raw Data"],
               horizontal=True
           )
       
       if view_option == "Executive Summary":
           numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
           categorical_cols = [col for col in df.columns if df[col].dtype == 'object' and df[col].nunique() < 50]
           
           # Find key business columns
           money_cols = [col for col in numeric_cols if any(term in col.lower() for term in ['revenue', 'sales', 'cost', 'price', 'value', 'amount'])]
           
           if money_cols and categorical_cols:
               # Show top performers
               primary_money = money_cols[0]
               best_cat = min(categorical_cols, key=lambda x: abs(df[x].nunique() - 10))  # Find categorical with ~10 values
               
               col1, col2 = st.columns(2)
               
               with col1:
                   st.write(f"**🏆 Top 5 {best_cat.replace('_', ' ').title()} by {primary_money.replace('_', ' ').title()}**")
                   top_performers = df.groupby(best_cat)[primary_money].sum().nlargest(5).round(2)
                   
                   # Create a simple bar chart
                   fig = px.bar(
                       x=top_performers.values,
                       y=top_performers.index,
                       orientation='h',
                       labels={'x': primary_money.replace('_', ' ').title(), 'y': best_cat.replace('_', ' ').title()},
                       color=top_performers.values,
                       color_continuous_scale='Blues'
                   )
                   fig.update_layout(
                       showlegend=False,
                       height=300,
                       margin=dict(l=0, r=0, t=0, b=0)
                   )
                   st.plotly_chart(fig, use_container_width=True)
               
               with col2:
                   st.write("**📊 Performance Metrics**")
                   total_value = df[primary_money].sum()
                   avg_value = df[primary_money].mean()
                   max_value = df[primary_money].max()
                   
                   st.metric("Total", format_business_metric(total_value, "currency"))
                   st.metric("Average", format_business_metric(avg_value, "currency"))
                   st.metric("Maximum", format_business_metric(max_value, "currency"))
           
           # Summary statistics table
           st.write("**📈 Key Statistics**")
           summary_data = []
           
           for col in numeric_cols[:8]:  # Limit to 8 columns
               summary_data.append({
                   'Metric': col.replace('_', ' ').title(),
                   'Total': format_business_metric(df[col].sum(), "number" if col not in money_cols else "currency"),
                   'Average': format_business_metric(df[col].mean(), "decimal" if col not in money_cols else "currency"),
                   'Min': format_business_metric(df[col].min(), "number" if col not in money_cols else "currency"),
                   'Max': format_business_metric(df[col].max(), "number" if col not in money_cols else "currency")
               })
           
           summary_df = pd.DataFrame(summary_data)
           st.dataframe(summary_df, use_container_width=True, hide_index=True)
               
       elif view_option == "Detailed Analysis":
           col1, col2, col3 = st.columns(3)
           categorical_cols = [col for col in df.columns if df[col].dtype == 'object' and df[col].nunique() < 50]
           numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
           
           with col1:
               if categorical_cols:
                   group_col = st.selectbox("📊 Group by", options=categorical_cols)
               else:
                   group_col = None
                   st.write("No categorical columns")
           
           with col2:
               if numeric_cols:
                   selected_metrics = st.multiselect("📈 Select Metrics", 
                                                    options=numeric_cols,
                                                    default=numeric_cols[:min(3, len(numeric_cols))])
               else:
                   selected_metrics = []
           
           with col3:
               agg_func = st.selectbox("🔢 Calculation",
                                     options=['Sum', 'Average', 'Count', 'Maximum', 'Minimum'],
                                     index=0)
               agg_map = {
                   'Sum': 'sum',
                   'Average': 'mean',
                   'Count': 'count',
                   'Maximum': 'max',
                   'Minimum': 'min'
               }
           
           if group_col and selected_metrics:
               try:
                   # Create aggregation
                   agg_dict = {metric: agg_map[agg_func] for metric in selected_metrics}
                   summary_df = df.groupby(group_col).agg(agg_dict).round(2)
                   
                   # Sort by first metric
                   summary_df = summary_df.sort_values(selected_metrics[0], ascending=False)
                   
                   # Format the columns
                   for col in summary_df.columns:
                       if any(term in col.lower() for term in ['revenue', 'sales', 'cost', 'price', 'value', 'amount', 'spend']):
                           summary_df[col] = summary_df[col].apply(lambda x: format_business_metric(x, "currency"))
                   
                   st.write(f"**📊 {agg_func} by {group_col.replace('_', ' ').title()}**")
                   st.dataframe(summary_df, use_container_width=True)
                   
                   # Quick visualization
                   if len(selected_metrics) == 1:
                       fig = px.bar(
                           summary_df.reset_index(),
                           x=group_col,
                           y=selected_metrics[0],
                           title=f"{selected_metrics[0].replace('_', ' ').title()} by {group_col.replace('_', ' ').title()}",
                           color=selected_metrics[0],
                           color_continuous_scale='Blues'
                       )
                       fig.update_layout(showlegend=False, height=400)
                       st.plotly_chart(fig, use_container_width=True)
                   
                   # Download button
                   csv = summary_df.to_csv()
                   st.download_button(
                       label="📥 Download Analysis",
                       data=csv,
                       file_name=f"analysis_{group_col}_{file_name}",
                       mime="text/csv"
                   )
                   
               except Exception as e:
                   st.error(f"Error creating analysis: {str(e)}")
       
       else:  # Raw Data
           with st.expander("🔍 Filter Options", expanded=False):
               filter_cols = st.columns(3)
               filters = {}
               categorical_cols = [col for col in df.columns if df[col].dtype == 'object' and df[col].nunique() < 100]
               numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
               
               # Categorical filters
               for i, col in enumerate(categorical_cols[:3]):
                   with filter_cols[i % 3]:
                       unique_vals = sorted(df[col].dropna().unique())
                       if len(unique_vals) <= 50:
                           selected = st.multiselect(f"Filter {col}", unique_vals)
                           if selected:
                               filters[col] = selected
               
               # Numeric range filters
               for i, col in enumerate(numeric_cols[:2], start=len(categorical_cols[:3])):
                   if df[col].nunique() > 5:
                       with filter_cols[i % 3]:
                           min_val = float(df[col].min())
                           max_val = float(df[col].max())
                           range_vals = st.slider(
                               f"{col} range",
                               min_val, max_val, (min_val, max_val)
                           )
                           if range_vals != (min_val, max_val):
                               filters[f"{col}_range"] = range_vals
           
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
           
           # Display options
           col1, col2 = st.columns([3, 1])
           with col1:
               search = st.text_input("🔍 Search in data", "")
           with col2:
               rows_to_show = st.selectbox("Rows", [50, 100, 200, 500], index=0)
           
           # Apply search
           if search:
               mask = filtered_df.astype(str).apply(lambda x: x.str.contains(search, case=False, na=False)).any(axis=1)
               filtered_df = filtered_df[mask]
           
           # Show data
           st.dataframe(filtered_df.head(rows_to_show), use_container_width=True, hide_index=True)
           
           if filtered_df.shape[0] != df.shape[0]:
               st.caption(f"Showing {min(rows_to_show, filtered_df.shape[0])} of {filtered_df.shape[0]} filtered rows (from {df.shape[0]} total)")
           else:
               st.caption(f"Showing first {min(rows_to_show, df.shape[0])} of {df.shape[0]} rows")

       # ================================
       # EXPORT SECTION
       # ================================
       st.write("### 💾 Export & Share")
       export_cols = st.columns(4)
       
       with export_cols[0]:
           csv = df.to_csv(index=False)
           st.download_button(
               label="📥 Download Data (CSV)",
               data=csv,
               file_name=f"data_{file_name}",
               mime="text/csv"
           )
       
       with export_cols[1]:
           # Executive Report
           report = f"""EXECUTIVE REPORT
Generated by Claude AI Business Intelligence
Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}

FILE: {file_name}

DATASET OVERVIEW
================
• Total Records: {df.shape[0]:,}
• Data Points: {df.shape[1]}
• Data Quality: {((df.notna().sum().sum() / (df.shape[0] * df.shape[1])) * 100):.1f}% complete

KEY METRICS
==========="""
           
           # Add created KPIs
           created_kpis = create_ai_driven_kpis(df, ai_analysis.get("suggested_kpis", []))
           for kpi in created_kpis[:6]:
               report += f"\n• {kpi['name']}: {kpi['value']}"
               if kpi['description']:
                   report += f" ({kpi['description']})"
           
           report += "\n\nEXECUTIVE INSIGHTS\n=================="
           if ai_analysis.get("business_insights"):
               for i, insight in enumerate(ai_analysis["business_insights"], 1):
                   report += f"\n{i}. {insight}"
           
           # Add custom analysis if available
           if st.session_state.custom_metrics:
               report += "\n\nCUSTOM ANALYSIS\n==============="
               for metric in st.session_state.custom_metrics:
                   report += f"\n• {metric['name']}: {metric['value']}"
           
           report += "\n\nDATA COLUMNS\n============"
           for col in df.columns:
               dtype = str(df[col].dtype)
               nunique = df[col].nunique()
               report += f"\n• {col}: {dtype} ({nunique} unique values)"
           
           st.download_button(
               label="📄 Executive Report",
               data=report,
               file_name=f"executive_report_{file_name}.txt",
               mime="text/plain"
           )
       
       with export_cols[2]:
           # Create PowerBI/Tableau ready format
           bi_ready_df = df.copy()
           
           # Add calculated columns that BI tools often need
           if any(col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])):
               date_col = next(col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col]))
               bi_ready_df['Year'] = df[date_col].dt.year
               bi_ready_df['Month'] = df[date_col].dt.month
               bi_ready_df['Quarter'] = df[date_col].dt.quarter
               bi_ready_df['Day_of_Week'] = df[date_col].dt.day_name()
           
           bi_csv = bi_ready_df.to_csv(index=False)
           st.download_button(
               label="📊 BI-Ready Export",
               data=bi_csv,
               file_name=f"bi_ready_{file_name}",
               mime="text/csv"
           )
       
       with export_cols[3]:
           if st.button("🔄 New Analysis", type="primary"):
               st.session_state.analysis_data = None
               st.session_state.analysis_type = None
               st.session_state.custom_metrics = []
               st.rerun()

   elif analysis_type == 'multi':
       # ================================
       # MULTI-FILE ANALYSIS
       # ================================
       st.subheader("🔗 Multi-File Business Analysis")
       
       # Overview of all files
       st.write("### 📊 File Overview")
       overview_cols = st.columns(len(data))
       for i, (file_name, df) in enumerate(data.items()):
           with overview_cols[i]:
               st.metric(file_name, f"{df.shape[0]:,} rows")
               completeness = (df.notna().sum().sum() / (df.shape[0] * df.shape[1])) * 100
               st.caption(f"Quality: {completeness:.1f}%")
       
       st.write("### 📈 Individual File Analysis")
       tabs = st.tabs(list(data.keys()))
       for i, (file_name, df) in enumerate(data.items()):
           with tabs[i]:
               st.write(f"#### {file_name}")
               
               # Quick stats
               col1, col2, col3 = st.columns(3)
               with col1:
                   st.metric("Records", f"{df.shape[0]:,}")
               with col2:
                   st.metric("Columns", f"{df.shape[1]}")
               with col3:
                   numeric_cols = df.select_dtypes(include=[np.number]).columns
                   st.metric("Metrics", f"{len(numeric_cols)}")
               
               # Run AI analysis for this file
               with st.spinner(f"Analyzing {file_name}..."):
                   ai_analysis = run_ai_driven_analysis(df)
               
               # Data preview
               st.write("#### Quick View")
               st.dataframe(df.head(20), use_container_width=True, hide_index=True)
               
               # Export options
               col1, col2 = st.columns(2)
               with col1:
                   csv = df.to_csv(index=False)
                   st.download_button(
                       label=f"📥 Download {file_name}",
                       data=csv,
                       file_name=f"analyzed_{file_name}",
                       mime="text/csv",
                       key=f"download_{i}"
                   )
               with col2:
                   # Quick insights
                   if ai_analysis.get("business_insights"):
                       st.info(f"💡 Key Insight: {ai_analysis['business_insights'][0]}")

# ================================
# FOOTER
# ================================
st.markdown("---")
st.markdown(
   """
   <div style='text-align: center; color: #666;'>
   Powered by Claude AI & Streamlit | Built for Business Intelligence
   </div>
   """, 
   unsafe_allow_html=True
)
