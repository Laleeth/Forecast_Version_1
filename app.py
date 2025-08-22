#!/usr/bin/env python3
"""
Dish Forecasting ML Pipeline - Main Application (FIXED VERSION)

A comprehensive Streamlit application for end-to-end dish demand forecasting
using multiple time series models with advanced data ingestion capabilities.

Author: Lalith Thomala
Version: 1.0.1 (Fixed)
Date: August 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

try:
    from dish_forecasting.data.data_ingestion import DataIngestion
    from dish_forecasting.models.time_series_models import TimeSeriesModels
    from dish_forecasting.visualization.visualization import Visualization
    from dish_forecasting.utils.utils import *
except ImportError as e:
    st.error(f"❌ Import error: {e}")
    st.error("Please ensure all modules are properly installed and the directory structure is correct.")
    st.stop()

# === STREAMLIT PAGE CONFIGURATION ===
st.set_page_config(
    page_title="Dish Forecasting ML Pipeline",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === HEADER ===
st.title("🚀 Dish Forecasting ML Pipeline")
st.markdown("""
**End-to-End Machine Learning Pipeline for Dish Demand Forecasting**

*Professional ML Solution for Restaurant Industry*

---
""")

# === SIDEBAR (REMOVED AUTHOR INFO) ===
with st.sidebar:
    st.header("🔧 Pipeline Controls")
    st.markdown("*Navigate through the ML pipeline steps*")

# === INITIALIZE SESSION STATE ===
if 'df' not in st.session_state:
    st.session_state.df = None
if 'dish_list' not in st.session_state:
    st.session_state.dish_list = None
if 'forecasts_complete' not in st.session_state:
    st.session_state.forecasts_complete = False
if 'pipeline_step' not in st.session_state:
    st.session_state.pipeline_step = 1

# === PIPELINE PROGRESS INDICATOR ===
steps = [
    "📊 Data Ingestion",
    "⚙️ Configuration", 
    "🔍 EDA",
    "🤖 Model Training",
    "📊 Results",
    "🎨 Visualization",
    "💾 Export"
]

cols = st.columns(len(steps))
for i, (col, step) in enumerate(zip(cols, steps)):
    with col:
        if i + 1 <= st.session_state.pipeline_step:
            st.success(f"✅ Step {i+1}")
        else:
            st.info(f"⏳ Step {i+1}")
        st.caption(step)

st.markdown("---")

# === STEP 1: DATA INGESTION ===
st.header("📊 Step 1: Data Ingestion")

try:
    data_ingestion = DataIngestion()

    # Choose data source
    data_source = st.radio(
        "Choose Data Source:",
        ["📁 File Upload", "🗄️ SQL Database"],
        horizontal=True,
        help="Select your preferred data ingestion method"
    )

    if data_source == "📁 File Upload":
        df, dish_list = data_ingestion.load_from_files()
        if df is not None:
            st.session_state.df = df
            st.session_state.dish_list = dish_list
            st.session_state.pipeline_step = max(st.session_state.pipeline_step, 2)

    elif data_source == "🗄️ SQL Database":
        df, dish_list = data_ingestion.load_from_sql()
        if df is not None:
            st.session_state.df = df
            st.session_state.dish_list = dish_list
            st.session_state.pipeline_step = max(st.session_state.pipeline_step, 2)

except Exception as e:
    st.error(f"❌ Error in data ingestion: {str(e)}")
    st.error("Please check your data files and try again.")

# === DATA OVERVIEW ===
if st.session_state.df is not None:
    st.success("✅ Data loaded successfully!")

    with st.expander("📋 Data Overview", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Total Records", f"{len(st.session_state.df):,}")
        with col2:
            st.metric("🍽️ Unique Dishes", f"{len(st.session_state.dish_list):,}")
        with col3:
            st.metric("🏙️ Cities", f"{st.session_state.df['kitchenName'].nunique():,}")
        with col4:
            date_range = (st.session_state.df['deliverydate'].max() - 
                         st.session_state.df['deliverydate'].min()).days
            st.metric("📅 Date Range (days)", f"{date_range:,}")

        # Sample data
        st.subheader("📄 Sample Data")
        st.dataframe(st.session_state.df.head(10), use_container_width=True)

# === STEP 2: CONFIGURATION ===
if st.session_state.df is not None:
    st.header("⚙️ Step 2: Pipeline Configuration")

    col1, col2 = st.columns(2)

    with col1:
        cities = st.session_state.df['kitchenName'].dropna().unique().tolist()
        selected_city = st.selectbox(
            "🏙️ Select City/Location", 
            cities,
            help="Choose the city/location for forecasting"
        )

    with col2:
        weeks = sorted(st.session_state.df['Week_num'].dropna().unique())
        selected_week = st.selectbox(
            "📅 Select Target Week", 
            weeks,
            help="Choose the week number to forecast"
        )

    # Get target dates and filter data
    current_year = st.session_state.df['deliverydate'].dt.year.max()
    target_dates = get_week_dates(current_year, selected_week)

    st.info(f"🎯 **Forecasting Target**: Week {selected_week} ({current_year}) | "
           f"{target_dates[0].strftime('%d/%m/%Y')} to {target_dates[4].strftime('%d/%m/%Y')}")

    # Filter data for training
    df_filtered = st.session_state.df[
        (st.session_state.df['kitchenName'] == selected_city) & 
        (st.session_state.df['Week_num'] < selected_week)
    ]

    # Store in session state
    st.session_state.df_filtered = df_filtered
    st.session_state.selected_city = selected_city
    st.session_state.selected_week = selected_week
    st.session_state.target_dates = target_dates
    st.session_state.pipeline_step = max(st.session_state.pipeline_step, 3)

    # Data quality check
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🎯 Training Records", f"{len(df_filtered):,}")
    with col2:
        dishes_with_data = df_filtered['variantname'].nunique()
        st.metric("🍽️ Dishes with Data", f"{dishes_with_data:,}")
    with col3:
        data_completeness = (dishes_with_data / len(st.session_state.dish_list) * 100) if st.session_state.dish_list else 0
        st.metric("📊 Data Completeness", f"{data_completeness:.1f}%")

# === STEP 3: EXPLORATORY DATA ANALYSIS ===
if st.session_state.get('df_filtered') is not None:
    st.header("🔍 Step 3: Exploratory Data Analysis")

    try:
        viz = Visualization()

        col1, col2 = st.columns(2)

        with col1:
            if st.button("📊 Generate EDA Dashboard", type="primary"):
                with st.spinner("🔄 Generating EDA dashboard..."):
                    viz.create_eda_dashboard(
                        st.session_state.df_filtered, 
                        st.session_state.dish_list[:10]
                    )
                st.session_state.pipeline_step = max(st.session_state.pipeline_step, 4)

        with col2:
            selected_dishes_eda = st.multiselect(
                "📈 Select Dishes for Historical Analysis",
                st.session_state.dish_list[:20],
                default=st.session_state.dish_list[:3] if len(st.session_state.dish_list) >= 3 else st.session_state.dish_list,
                help="Choose dishes to analyze historical patterns"
            )

        if selected_dishes_eda and st.button("📈 Show Historical Analysis"):
            with st.spinner("📊 Analyzing historical patterns..."):
                viz.plot_historical_data(st.session_state.df_filtered, selected_dishes_eda)

    except Exception as e:
        st.error(f"❌ Error in EDA: {str(e)}")

# === STEP 4: MODEL TRAINING & FORECASTING ===
if st.session_state.get('df_filtered') is not None:
    st.header("🤖 Step 4: Model Training & Forecasting")

    try:
        ts_models = TimeSeriesModels()
        available_models = ts_models.get_available_models()

        # Enhanced model selection with ALL option
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🎯 Model Selection")

            # Add "All Models" option
            use_all_models = st.checkbox("🚀 Use All Available Models", value=False,
                                       help="Select all available models for comprehensive comparison")

            if use_all_models:
                selected_models = available_models.copy()
                st.success(f"✅ Selected {len(selected_models)} models: {', '.join(selected_models)}")
            else:
                selected_models = st.multiselect(
                    "🎯 Select Specific Models",
                    available_models,
                    default=['Prophet', 'Random Forest', 'XGBoost'],
                    help="Choose specific machine learning models to train and compare"
                )

        with col2:
            st.subheader("🍽️ Dish Selection")

            # Add "All Dishes" option
            use_all_dishes = st.checkbox("🍽️ Forecast All Dishes", value=False,
                                       help="Forecast all available dishes")

            if use_all_dishes:
                dishes_to_forecast = st.session_state.dish_list.copy()
                st.success(f"✅ Selected {len(dishes_to_forecast)} dishes for forecasting")
            else:
                max_dishes = min(50, len(st.session_state.dish_list))  # Limit for performance
                dishes_to_forecast = st.multiselect(
                    "🎯 Select Specific Dishes",
                    st.session_state.dish_list,
                    default=st.session_state.dish_list[:10] if len(st.session_state.dish_list) >= 10 else st.session_state.dish_list,
                    help=f"Choose specific dishes to forecast (max {max_dishes} for performance)"
                )

                if len(dishes_to_forecast) > max_dishes:
                    st.warning(f"⚠️ Too many dishes selected. Using first {max_dishes} for performance.")
                    dishes_to_forecast = dishes_to_forecast[:max_dishes]

        # Model information
        with st.expander("🔬 Available Models Information", expanded=False):
            model_descriptions = ts_models.get_model_descriptions()

            # Group models by type
            core_models = ['Prophet', 'Linear Regression', 'Random Forest', 'XGBoost', 'Moving Average']
            statistical_models = ['ARIMA', 'Exponential Smoothing', 'Seasonal Decompose']
            deep_learning_models = ['LSTM']

            st.markdown("**🏆 Core Models (Always Available):**")
            for model in core_models:
                if model in model_descriptions:
                    availability = "✅ Available" if model in available_models else "❌ Not Available"
                    st.write(f"• **{model}**: {model_descriptions[model]} ({availability})")

            st.markdown("**📊 Statistical Models (Requires statsmodels):**")
            for model in statistical_models:
                if model in model_descriptions:
                    availability = "✅ Available" if model in available_models else "❌ Not Available (pip install statsmodels)"
                    st.write(f"• **{model}**: {model_descriptions[model]} ({availability})")

            st.markdown("**🧠 Deep Learning Models (Requires tensorflow):**")
            for model in deep_learning_models:
                if model in model_descriptions:
                    availability = "✅ Available" if model in available_models else "❌ Not Available (pip install tensorflow)"
                    st.write(f"• **{model}**: {model_descriptions[model]} ({availability})")

        # Run forecasting pipeline
        if selected_models and dishes_to_forecast and st.button("🚀 Run ML Pipeline", type="primary"):

            st.subheader("⚡ ML Pipeline Execution")

            # Initialize containers
            all_forecasts = {}
            all_scores = {}
            all_averages = {}
            all_previous = {}
            dish_metadata = {}

            # Progress tracking
            progress_bar = st.progress(0)
            status_placeholder = st.empty()
            metrics_placeholder = st.empty()

            total_dishes = len(dishes_to_forecast)
            processed_count = 0
            successful_count = 0

            # Process each dish
            for i, dish in enumerate(dishes_to_forecast):
                try:
                    # Update progress
                    progress = (i + 1) / total_dishes
                    progress_bar.progress(progress)
                    status_placeholder.text(f"🔄 Processing: {dish} ({i+1}/{total_dishes})")

                    # Run forecasting
                    forecasts, error = ts_models.forecast_models(
                        st.session_state.df_filtered, dish, st.session_state.target_dates, selected_models
                    )

                    processed_count += 1

                    # Handle results
                    if forecasts:
                        # Handle special cases
                        if 'New Dish Strategy' in forecasts or 'Limited Data Strategy' in forecasts:
                            special_strategy = 'New Dish Strategy' if 'New Dish Strategy' in forecasts else 'Limited Data Strategy'

                            all_forecasts[dish] = {special_strategy: forecasts[special_strategy]['forecast']}
                            all_scores[dish] = {special_strategy: forecasts[special_strategy]['score']}

                            dish_metadata[dish] = {
                                'last_delivery_date': 'No history' if special_strategy == 'New Dish Strategy' else 'Limited',
                                'prev_week_num': 'N/A',
                                'data_status': 'New Dish' if special_strategy == 'New Dish Strategy' else 'Limited Data'
                            }
                        else:
                            # Regular forecasts
                            all_forecasts[dish] = {}
                            all_scores[dish] = {}

                            for model_name in selected_models:
                                if model_name in forecasts and forecasts[model_name] is not None:
                                    all_forecasts[dish][model_name] = forecasts[model_name]['forecast']
                                    all_scores[dish][model_name] = forecasts[model_name]['score']

                            dish_metadata[dish] = {
                                'last_delivery_date': 'Available',
                                'prev_week_num': st.session_state.selected_week - 1,
                                'data_status': 'Normal'
                            }

                        # Get historical data
                        avg_values = get_historical_average(st.session_state.df_filtered, dish, st.session_state.target_dates)
                        prev_values, prev_week_num, last_delivery_date = get_previous_week_actual(st.session_state.df_filtered, dish, st.session_state.target_dates)

                        all_averages[dish] = avg_values
                        all_previous[dish] = prev_values

                        successful_count += 1

                    # Update metrics
                    with metrics_placeholder.container():
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("📊 Processed", processed_count)
                        with col2:
                            st.metric("✅ Successful", successful_count)
                        with col3:
                            success_rate = (successful_count / processed_count * 100) if processed_count > 0 else 0
                            st.metric("📈 Success Rate", f"{success_rate:.1f}%")

                except Exception as e:
                    st.warning(f"⚠️ Error processing {dish}: {str(e)}")
                    processed_count += 1

            # Clean up progress indicators
            progress_bar.empty()
            status_placeholder.empty()

            # Store results in session state
            st.session_state.forecasts_complete = True
            st.session_state.all_forecasts = all_forecasts
            st.session_state.all_scores = all_scores
            st.session_state.all_averages = all_averages
            st.session_state.all_previous = all_previous
            st.session_state.dish_metadata = dish_metadata
            st.session_state.selected_models = selected_models
            st.session_state.dishes_to_forecast = dishes_to_forecast

            # Success message
            st.success(f"🎉 **Pipeline completed successfully!**")
            st.success(f"✅ Processed {successful_count}/{processed_count} dishes successfully")

            # Update pipeline step
            st.session_state.pipeline_step = max(st.session_state.pipeline_step, 5)

    except Exception as e:
        st.error(f"❌ Error in model training: {str(e)}")

# === STEP 5: RESULTS & ANALYSIS ===
if st.session_state.get('forecasts_complete', False):
    st.header("📊 Step 5: Results & Model Performance")

    try:
        # Model Performance Analysis
        display_model_performance(
            st.session_state.all_forecasts,
            st.session_state.all_scores, 
            st.session_state.dish_metadata,
            st.session_state.selected_models
        )

        st.session_state.pipeline_step = max(st.session_state.pipeline_step, 6)

    except Exception as e:
        st.error(f"❌ Error in results analysis: {str(e)}")

# === STEP 6: VISUALIZATION ===
if st.session_state.get('forecasts_complete', False):
    st.header("🎨 Step 6: Interactive Visualization")

    try:
        # Get dishes that have forecasts
        available_dishes = list(st.session_state.all_forecasts.keys())

        selected_dishes_viz = st.multiselect(
            "📈 Select Dishes to Visualize",
            available_dishes,
            default=available_dishes[:5] if len(available_dishes) >= 5 else available_dishes,
            help="Choose dishes for forecast visualization"
        )

        if selected_dishes_viz:
            col1, col2 = st.columns(2)

            with col1:
                if st.button("📊 Show Forecast Charts"):
                    with st.spinner("🎨 Creating forecast visualizations..."):
                        viz = Visualization()
                        main_model = st.session_state.selected_models[0] if st.session_state.selected_models else 'Prophet'
                        viz.plot_forecasts(
                            st.session_state.df_filtered,
                            selected_dishes_viz,
                            st.session_state.all_forecasts,
                            st.session_state.target_dates,
                            main_model
                        )

            with col2:
                if st.button("🔬 Compare Models"):
                    if selected_dishes_viz:
                        with st.spinner("🔬 Comparing model performances..."):
                            viz = Visualization() 
                            viz.compare_models(
                                selected_dishes_viz[0],
                                st.session_state.all_forecasts,
                                st.session_state.selected_models,
                                st.session_state.target_dates
                            )

        st.session_state.pipeline_step = max(st.session_state.pipeline_step, 7)

    except Exception as e:
        st.error(f"❌ Error in visualization: {str(e)}")

# === STEP 7: RESULTS TABLES & EXPORT ===
if st.session_state.get('forecasts_complete', False):
    st.header("📋 Step 7: Detailed Results & Export")

    try:
        # Results Tables - Fixed to check if models exist
        if hasattr(st.session_state, 'selected_models') and st.session_state.selected_models:
            display_results_tables(
                st.session_state.all_forecasts,
                st.session_state.all_scores,
                st.session_state.dish_metadata,
                st.session_state.selected_models,
                st.session_state.target_dates,
                st.session_state.selected_week
            )
        else:
            st.warning("⚠️ No model results available for detailed display")

        # Export Options
        st.subheader("💾 Export Results")
        if hasattr(st.session_state, 'selected_models') and st.session_state.selected_models:
            export_results(
                st.session_state.all_forecasts,
                st.session_state.selected_models,
                st.session_state.target_dates,
                st.session_state.selected_week
            )

    except Exception as e:
        st.error(f"❌ Error in results display: {str(e)}")

# === FOOTER ===
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><strong>Dish Forecasting ML Pipeline v1.0.1</strong></p>
    <p>🚀 End-to-End Machine Learning for Demand Forecasting</p>
</div>
""", unsafe_allow_html=True)

# === SIDEBAR INFO (REMOVED AUTHOR) ===
with st.sidebar:
    st.markdown("---")
    st.info("🚀 **Ready to forecast!**\n\nFollow the pipeline steps to get started with your dish demand predictions.")

    # Pipeline status
    if st.session_state.get('forecasts_complete', False):
        st.success("✅ **Pipeline Complete!**")
        if hasattr(st.session_state, 'dishes_to_forecast'):
            st.write(f"📊 Dishes: {len(st.session_state.dishes_to_forecast)}")
        if hasattr(st.session_state, 'selected_models'):
            st.write(f"🤖 Models: {len(st.session_state.selected_models)}")


if __name__ == "__main__":
    pass
