"""
Utility Functions for Dish Forecasting Pipeline

This module contains all utility functions for data processing, evaluation,
and display functionality for the dish forecasting ML pipeline.

Author: Lalith Thomala
Version: 1.0.1 (Fixed)
Date: August 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error

__author__ = "Lalith Thomala"
__version__ = "1.0.1"

# === EVALUATION FUNCTIONS ===

def evaluate(y_true, y_pred):
    """
    Evaluate model performance using MAE and RMSE metrics.

    Args:
        y_true: Actual values
        y_pred: Predicted values

    Returns:
        dict: Dictionary containing MAE and RMSE scores
    """
    try:
        return {
            "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": mean_squared_error(y_true, y_pred, squared=False)
        }
    except Exception as e:
        st.warning(f"⚠️ Error in evaluation: {str(e)}")
        return {"MAE": 0, "RMSE": 0}

# === DATE UTILITY FUNCTIONS ===

def get_week_dates(year, week_num):
    """
    Get the Monday-Friday dates for a specific week number.

    Args:
        year (int): Year
        week_num (int): Week number

    Returns:
        list: List of datetime objects for Monday to Friday
    """
    try:
        year = int(year)
        week_num = int(week_num)

        # Get first day of year
        jan1 = datetime(year, 1, 1)

        # Find Monday of week 1
        if jan1.weekday() == 0:  # Monday
            week1_monday = jan1
        else:
            week1_monday = jan1 + timedelta(days=7 - jan1.weekday())

        # Calculate Monday of target week
        target_monday = week1_monday + timedelta(weeks=week_num - 1)

        # Generate Monday to Friday dates
        week_dates = []
        for i in range(5):  # Monday to Friday
            week_dates.append(target_monday + timedelta(days=i))

        return week_dates

    except Exception as e:
        st.error(f"❌ Error in get_week_dates: {str(e)}")
        return []

# === DATA PROCESSING FUNCTIONS ===

def get_historical_average(df, dish, target_dates):
    """
    Get historical average quantities for the same weekdays.

    Args:
        df: DataFrame with historical data
        dish: Dish name
        target_dates: List of target dates

    Returns:
        list: Historical averages for each target date
    """
    try:
        dish_df = df[df['variantname'].str.strip() == dish.strip()]

        if dish_df.empty:
            return [0] * 5

        daily = dish_df.groupby('deliverydate')['Quantity'].sum().reset_index()
        daily['deliverydate'] = pd.to_datetime(daily['deliverydate'])
        daily['weekday'] = daily['deliverydate'].dt.dayofweek

        averages = []
        for target_date in target_dates:
            target_weekday = target_date.weekday()
            weekday_data = daily[daily['weekday'] == target_weekday]['Quantity']
            avg = weekday_data.mean() if len(weekday_data) > 0 else 0
            averages.append(round(avg, 1))

        return averages

    except Exception as e:
        st.warning(f"⚠️ Error getting historical average for {dish}: {str(e)}")
        return [0] * 5

def get_previous_week_actual(df, dish, target_dates):
    """
    Get actual quantities from the previous week.

    Args:
        df: DataFrame with historical data
        dish: Dish name  
        target_dates: List of target dates

    Returns:
        tuple: (actuals, prev_week_num, last_delivery_date)
    """
    try:
        dish_df = df[df['variantname'].str.strip() == dish.strip()]

        if dish_df.empty:
            return [0] * 5, None, None

        # Get daily data
        daily = dish_df.groupby('deliverydate')['Quantity'].sum().reset_index()
        daily['deliverydate'] = pd.to_datetime(daily['deliverydate'])
        daily = daily.sort_values('deliverydate')

        if daily.empty:
            return [0] * 5, None, None

        last_delivery_date = daily['deliverydate'].max().strftime('%d/%m/%Y')

        # Get previous week number
        dish_with_week = df[df['variantname'].str.strip() == dish.strip()]
        prev_week_num = None

        if not dish_with_week.empty:
            prev_week_num = dish_with_week['Week_num'].max()

        # Get actual values for each target date
        actuals = []
        for target_date in target_dates:
            target_weekday = target_date.weekday()

            # Look for exact match from previous week
            prev_week_date = target_date - timedelta(days=7)
            exact_match = daily[daily['deliverydate'].dt.date == prev_week_date.date()]

            if not exact_match.empty:
                actuals.append(int(exact_match['Quantity'].iloc[0]))
                continue

            # Look for most recent data for this weekday
            weekday_data = daily[daily['deliverydate'].dt.dayofweek == target_weekday]
            if not weekday_data.empty:
                most_recent = weekday_data.iloc[-1]
                actuals.append(int(most_recent['Quantity']))
                continue

            # Look for recent data within last 2 weeks
            two_weeks_ago = target_date - timedelta(days=14)
            recent_data = daily[
                (daily['deliverydate'] >= two_weeks_ago) &
                (daily['deliverydate'].dt.dayofweek == target_weekday)
            ]

            if not recent_data.empty:
                avg_quantity = recent_data['Quantity'].mean()
                actuals.append(int(round(avg_quantity)))
                continue

            # Default to 0 if no data found
            actuals.append(0)

        return actuals, prev_week_num, last_delivery_date

    except Exception as e:
        st.warning(f"⚠️ Error getting previous week actual for {dish}: {str(e)}")
        return [0] * 5, None, None

# === SIMILARITY AND FORECASTING HELPER FUNCTIONS ===

def get_similar_dishes(df, new_dish, dish_list, top_n=3):
    """
    Find similar dishes based on name similarity.

    Args:
        df: DataFrame with historical data
        new_dish: New dish name
        dish_list: List of all dishes
        top_n: Number of similar dishes to return

    Returns:
        list: List of similar dishes with similarity scores
    """
    try:
        similar_dishes = []
        new_dish_clean = new_dish.strip().lower()

        for dish in dish_list:
            if dish.strip() != new_dish.strip():
                dish_clean = dish.strip().lower()

                # Calculate similarity based on common words
                new_words = set(new_dish_clean.split())
                dish_words = set(dish_clean.split())

                if new_words and dish_words:
                    similarity = len(new_words.intersection(dish_words)) / len(new_words.union(dish_words))

                    # Check if dish has data
                    dish_data = df[df['variantname'].str.strip() == dish.strip()]
                    if not dish_data.empty:
                        similar_dishes.append({
                            'dish': dish,
                            'similarity': similarity,
                            'data_points': len(dish_data)
                        })

        # Sort by similarity and data points
        similar_dishes.sort(key=lambda x: (x['similarity'], x['data_points']), reverse=True)
        return similar_dishes[:top_n]

    except Exception as e:
        st.warning(f"⚠️ Error finding similar dishes for {new_dish}: {str(e)}")
        return []

def forecast_new_dish(df, new_dish, dish_list, target_dates):
    """
    Forecast for new dishes using similar dishes' patterns.

    Args:
        df: Historical data
        new_dish: New dish name
        dish_list: List of all dishes
        target_dates: Target forecast dates

    Returns:
        tuple: (forecasts, explanation)
    """
    try:
        similar_dishes = get_similar_dishes(df, new_dish, dish_list)

        if not similar_dishes:
            # Use overall average if no similar dishes found
            all_dishes_data = df.groupby(['deliverydate', 'variantname'])['Quantity'].sum().reset_index()
            if not all_dishes_data.empty:
                daily_avg = all_dishes_data.groupby('deliverydate')['Quantity'].mean()
                overall_avg = daily_avg.mean()
                return [max(1, int(overall_avg * 0.5))] * 5, "Overall average (50% conservative)"
            else:
                return [1] * 5, "Default minimum (no data available)"

        # Use weighted average of similar dishes
        weighted_forecasts = []

        for target_date in target_dates:
            target_weekday = target_date.weekday()
            weighted_sum = 0
            weight_total = 0

            for similar in similar_dishes:
                similar_dish = similar['dish']
                weight = similar['similarity'] * (similar['data_points'] / 100)  # Normalize

                # Get historical average for this weekday
                dish_data = df[df['variantname'].str.strip() == similar_dish.strip()]
                daily = dish_data.groupby('deliverydate')['Quantity'].sum().reset_index()
                daily['deliverydate'] = pd.to_datetime(daily['deliverydate'])
                daily['weekday'] = daily['deliverydate'].dt.dayofweek

                weekday_data = daily[daily['weekday'] == target_weekday]['Quantity']

                if len(weekday_data) > 0:
                    avg_quantity = weekday_data.mean()
                    weighted_sum += avg_quantity * weight
                    weight_total += weight

            if weight_total > 0:
                forecast_value = max(1, int(weighted_sum / weight_total))
            else:
                forecast_value = 1

            weighted_forecasts.append(forecast_value)

        # Create explanation
        similar_names = [s['dish'] for s in similar_dishes[:2]]
        explanation = f"Based on similar dishes: {', '.join(similar_names)}"

        return weighted_forecasts, explanation

    except Exception as e:
        st.warning(f"⚠️ Error forecasting new dish {new_dish}: {str(e)}")
        return [1] * 5, f"Error occurred: {str(e)}"

def forecast_insufficient_data(df, dish, target_dates, min_required=10):
    """
    Handle dishes with insufficient historical data.

    Args:
        df: Historical data
        dish: Dish name
        target_dates: Target dates
        min_required: Minimum data points required

    Returns:
        tuple: (forecasts, explanation)
    """
    try:
        dish_df = df[df['variantname'].str.strip() == dish.strip()]

        if dish_df.empty:
            return forecast_new_dish(df, dish, df['variantname'].unique(), target_dates)

        # Get available data
        daily = dish_df.groupby('deliverydate')['Quantity'].sum().reset_index()
        daily['deliverydate'] = pd.to_datetime(daily['deliverydate'])
        daily['weekday'] = daily['deliverydate'].dt.dayofweek
        daily = daily[daily['weekday'] < 5]  # Only weekdays

        data_points = len(daily)

        if data_points == 0:
            return forecast_new_dish(df, dish, df['variantname'].unique(), target_dates)

        # Use simple moving average with available data
        forecasts = []
        for target_date in target_dates:
            target_weekday = target_date.weekday()

            # Get data for this weekday
            weekday_data = daily[daily['weekday'] == target_weekday]['Quantity']

            if len(weekday_data) > 0:
                # Use available data for this weekday
                if len(weekday_data) >= 2:
                    forecast_val = int(weekday_data.tail(2).mean())  # Last 2 observations
                else:
                    forecast_val = int(weekday_data.iloc[0])  # Single observation
            else:
                # No data for this weekday, use overall average
                overall_avg = daily['Quantity'].mean()
                forecast_val = int(overall_avg)

            forecasts.append(max(1, forecast_val))

        explanation = f"Limited data ({data_points} points) - using recent averages"
        return forecasts, explanation

    except Exception as e:
        st.warning(f"⚠️ Error forecasting insufficient data for {dish}: {str(e)}")
        return [1] * 5, f"Error occurred: {str(e)}"

# === DISPLAY FUNCTIONS (FIXED) ===

def display_model_performance(all_forecasts, all_scores, dish_metadata, selected_models):
    """
    Display comprehensive model performance comparison and recommendations.

    Args:
        all_forecasts: Dictionary of all forecasts
        all_scores: Dictionary of all scores
        dish_metadata: Metadata about dishes
        selected_models: List of selected models
    """
    try:
        st.subheader("🏆 Model Performance Analysis")

        if not all_forecasts:
            st.warning("⚠️ No forecasts available for performance analysis")
            return

        # Calculate performance metrics
        model_performance = {}
        special_cases = {'new_dishes': 0, 'limited_data': 0, 'normal': 0}

        # Include special strategies in model list
        all_model_names = list(selected_models) + ['New Dish Strategy', 'Limited Data Strategy']

        for model_name in all_model_names:
            mae_scores = []
            rmse_scores = []
            dish_count = 0

            for dish in all_forecasts.keys():
                # Count special cases (only count each dish once)
                data_status = dish_metadata.get(dish, {}).get('data_status', 'Normal')

                # Collect scores
                if model_name in all_scores.get(dish, {}):
                    score = all_scores[dish][model_name]
                    if isinstance(score, dict) and score.get('MAE', 0) > 0:
                        mae_scores.append(score['MAE'])
                        rmse_scores.append(score['RMSE'])
                        dish_count += 1

            # Calculate model performance metrics
            if mae_scores:
                model_performance[model_name] = {
                    'Avg_MAE': round(np.mean(mae_scores), 2),
                    'Avg_RMSE': round(np.mean(rmse_scores), 2),
                    'Dishes_Count': dish_count,
                    'Median_MAE': round(np.median(mae_scores), 2),
                    'Std_MAE': round(np.std(mae_scores), 2)
                }

        # Count special cases properly (once per dish)
        for dish in all_forecasts.keys():
            data_status = dish_metadata.get(dish, {}).get('data_status', 'Normal')
            if data_status == 'New Dish':
                special_cases['new_dishes'] += 1
            elif data_status == 'Limited Data':
                special_cases['limited_data'] += 1
            else:
                special_cases['normal'] += 1

        # Display special cases summary
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🆕 New Dishes", special_cases['new_dishes'])
        with col2:
            st.metric("⚠️ Limited Data", special_cases['limited_data']) 
        with col3:
            st.metric("✅ Normal Dishes", special_cases['normal'])
        with col4:
            total_dishes_processed = sum(special_cases.values())
            st.metric("📊 Total Processed", total_dishes_processed)

        # Performance comparison table (exclude special strategies)
        regular_models = {k: v for k, v in model_performance.items() 
                         if k not in ['New Dish Strategy', 'Limited Data Strategy']}

        if regular_models:
            st.subheader("📊 Model Performance Comparison")
            perf_df = pd.DataFrame(regular_models).T
            perf_df = perf_df.sort_values('Avg_MAE')  # Sort by MAE (lower is better)

            st.dataframe(perf_df, use_container_width=True)

            # Best model recommendation
            if len(perf_df) > 0:
                best_model = perf_df.index[0]
                best_mae = perf_df.loc[best_model, 'Avg_MAE']

                st.success(f"🥇 **Recommended Model: {best_model}** (Average MAE: {best_mae})")
        else:
            st.warning("⚠️ No regular model performance data available")

    except Exception as e:
        st.error(f"❌ Error displaying model performance: {str(e)}")

def display_results_tables(all_forecasts, all_scores, dish_metadata, selected_models, target_dates, selected_week):
    """
    Display detailed results tables for each model with FIXED error handling.
    """
    try:
        st.subheader("📋 Detailed Results by Model")

        if not all_forecasts:
            st.warning("⚠️ No forecasts available to display")
            return

        # FIXED: Ensure selected_models is not empty and contains valid models
        if not selected_models:
            st.warning("⚠️ No models selected for display")
            return

        # Check which models actually have results
        available_models = []
        for model in selected_models:
            has_forecasts = any(model in all_forecasts.get(dish, {}) for dish in all_forecasts.keys())
            if has_forecasts:
                available_models.append(model)

        # Add special strategies if they exist
        for special in ['New Dish Strategy', 'Limited Data Strategy']:
            has_special = any(special in all_forecasts.get(dish, {}) for dish in all_forecasts.keys())
            if has_special and special not in available_models:
                available_models.append(special)

        if not available_models:
            st.warning("⚠️ No model results available for display")
            return

        # FIXED: Create tabs only for models with actual results
        model_tabs = st.tabs([f"📊 {model}" for model in available_models])

        date_columns = [date.strftime("%d/%m/%Y") for date in target_dates]

        for tab_idx, model_name in enumerate(available_models):
            with model_tabs[tab_idx]:
                # Create forecast table for this model
                model_forecasts = {}
                model_scores_list = []

                for dish in all_forecasts.keys():
                    if model_name in all_forecasts[dish]:
                        forecast = all_forecasts[dish][model_name]
                        if forecast:  # Ensure forecast is not None or empty
                            model_forecasts[dish] = forecast

                            # Add scores if available
                            if model_name in all_scores.get(dish, {}):
                                score = all_scores[dish][model_name]
                                if isinstance(score, dict):
                                    model_scores_list.append({
                                        'Dish': dish,
                                        'MAE': round(score.get('MAE', 0), 2),
                                        'RMSE': round(score.get('RMSE', 0), 2),
                                        'Last_Delivery': dish_metadata.get(dish, {}).get('last_delivery_date', 'N/A'),
                                        'Data_Status': dish_metadata.get(dish, {}).get('data_status', 'Unknown')
                                    })

                if model_forecasts:
                    # Main forecast table
                    forecast_df = pd.DataFrame(model_forecasts).T
                    forecast_df.columns = date_columns
                    forecast_df.insert(0, "Dish", forecast_df.index)
                    forecast_df = forecast_df.reset_index(drop=True)

                    # Add totals row
                    totals_row = ['🔺 TOTAL'] + [forecast_df[col].sum() for col in date_columns]
                    totals_df = pd.DataFrame([totals_row], columns=['Dish'] + date_columns)
                    forecast_df = pd.concat([forecast_df, totals_df], ignore_index=True)

                    st.subheader(f"{model_name} - Week {selected_week} Forecast")
                    st.dataframe(forecast_df, use_container_width=True)

                    # Summary metrics
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        total_forecast = sum([sum(values) for values in model_forecasts.values()])
                        st.metric("📊 Total Weekly Forecast", f"{total_forecast:,}")

                    with col2:
                        avg_daily = total_forecast / 5 if total_forecast > 0 else 0
                        st.metric("📈 Average Daily Forecast", f"{avg_daily:.1f}")

                    with col3:
                        num_dishes = len(model_forecasts)
                        st.metric("🍽️ Number of Dishes", f"{num_dishes:,}")

                else:
                    st.warning(f"⚠️ No forecast data available for {model_name}")

    except Exception as e:
        st.error(f"❌ Error displaying results tables: {str(e)}")

def export_results(all_forecasts, selected_models, target_dates, selected_week):
    """
    Export results to downloadable CSV files.
    """
    try:
        st.subheader("💾 Export Forecast Results")

        if not all_forecasts:
            st.warning("⚠️ No forecast results to export")
            return

        date_columns = [date.strftime("%d/%m/%Y") for date in target_dates]

        col1, col2, col3 = st.columns(3)

        # Export main model results
        with col1:
            if selected_models and all_forecasts:
                # Find first available model
                main_model = None
                for model in selected_models:
                    if any(model in all_forecasts.get(dish, {}) for dish in all_forecasts.keys()):
                        main_model = model
                        break

                if main_model:
                    model_forecasts = {}

                    # Collect forecasts for main model
                    for dish in all_forecasts.keys():
                        if main_model in all_forecasts[dish]:
                            forecast = all_forecasts[dish][main_model]
                            if forecast:  # Ensure forecast is not None
                                model_forecasts[dish] = forecast

                    if model_forecasts:
                        # Create DataFrame
                        forecast_df = pd.DataFrame(model_forecasts).T
                        forecast_df.columns = date_columns
                        forecast_df.insert(0, "Dish", forecast_df.index)
                        forecast_df = forecast_df.reset_index(drop=True)

                        # Add totals row
                        totals = ['TOTAL'] + [forecast_df[col].sum() for col in date_columns]
                        totals_df = pd.DataFrame([totals], columns=['Dish'] + date_columns)
                        forecast_df = pd.concat([forecast_df, totals_df], ignore_index=True)

                        st.download_button(
                            f"⬇️ Download {main_model} Forecast",
                            forecast_df.to_csv(index=False),
                            f"{main_model.lower().replace(' ', '_')}_forecast_week_{selected_week}.csv",
                            mime="text/csv",
                            help=f"Download {main_model} forecast results as CSV"
                        )
                    else:
                        st.info(f"ℹ️ No {main_model} results to export")
                else:
                    st.info("ℹ️ No model results available for export")

        # Export summary report
        with col2:
            if all_forecasts and selected_models:
                summary_data = []

                for dish in all_forecasts.keys():
                    for model in selected_models + ['New Dish Strategy', 'Limited Data Strategy']:
                        if model in all_forecasts[dish]:
                            forecast = all_forecasts[dish][model]
                            if forecast:  # Ensure forecast is not None
                                summary_data.append({
                                    'Dish': dish,
                                    'Model': model,
                                    'Total_Weekly': sum(forecast),
                                    'Avg_Daily': round(np.mean(forecast), 2),
                                    'Max_Day': max(forecast),
                                    'Min_Day': min(forecast),
                                    'Std_Dev': round(np.std(forecast), 2)
                                })

                if summary_data:
                    summary_df = pd.DataFrame(summary_data)

                    st.download_button(
                        "⬇️ Download Summary Report",
                        summary_df.to_csv(index=False),
                        f"forecast_summary_week_{selected_week}.csv",
                        mime="text/csv",
                        help="Download summary statistics as CSV"
                    )
                else:
                    st.info("ℹ️ No summary data to export")

    except Exception as e:
        st.error(f"❌ Error in export functionality: {str(e)}")
