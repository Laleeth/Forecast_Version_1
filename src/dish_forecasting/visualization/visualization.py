"""
Advanced Visualization Module for Dish Forecasting Pipeline

This module provides comprehensive visualization capabilities including EDA dashboards,
forecast plotting, model comparison, and interactive charts.

Author: Lalith Thomala
Version: 1.0.1
Date: August 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

__author__ = "Lalith Thomala"
__version__ = "1.0.1"

class Visualization:
    """
    Comprehensive visualization class for the dish forecasting pipeline.

    Features:
    - EDA dashboards with multiple chart types
    - Historical data analysis and patterns
    - Forecast visualization with confidence intervals
    - Model comparison and performance metrics
    - Interactive plotly charts with professional styling
    """

    def __init__(self):
        """Initialize the Visualization class with color schemes and styling."""
        self.colors = px.colors.qualitative.Set1
        self.color_palette = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e', 
            'success': '#2ca02c',
            'warning': '#d62728',
            'info': '#9467bd',
            'light': '#8c564b',
            'dark': '#e377c2'
        }
        self.chart_config = {
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d']
        }

    def create_eda_dashboard(self, df, dish_sample):
        """
        Create comprehensive EDA dashboard with multiple visualizations.

        Args:
            df: Historical data DataFrame
            dish_sample: Sample of dishes for analysis
        """
        try:
            st.subheader("📊 Exploratory Data Analysis Dashboard")

            if df.empty:
                st.warning("⚠️ No data available for EDA")
                return

            # === OVERVIEW METRICS ===
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                total_quantity = df['Quantity'].sum()
                st.metric("📦 Total Demand", f"{total_quantity:,}")

            with col2:
                avg_daily = df.groupby('deliverydate')['Quantity'].sum().mean()
                st.metric("📈 Avg Daily Demand", f"{avg_daily:.0f}")

            with col3:
                peak_day = df.groupby('deliverydate')['Quantity'].sum().max()
                st.metric("🚀 Peak Day Demand", f"{peak_day:,}")

            with col4:
                unique_dishes = df['variantname'].nunique()
                st.metric("🍽️ Active Dishes", f"{unique_dishes:,}")

            # === TIME SERIES ANALYSIS ===
            st.subheader("📈 Time Series Analysis")

            daily_demand = df.groupby('deliverydate')['Quantity'].sum().reset_index()
            daily_demand['deliverydate'] = pd.to_datetime(daily_demand['deliverydate'])
            daily_demand = daily_demand.sort_values('deliverydate')

            # Main time series plot
            fig_ts = go.Figure()

            fig_ts.add_trace(go.Scatter(
                x=daily_demand['deliverydate'],
                y=daily_demand['Quantity'],
                mode='lines+markers',
                name='Daily Demand',
                line=dict(color=self.color_palette['primary'], width=2),
                marker=dict(size=4),
                hovertemplate='<b>Date</b>: %{x}<br><b>Demand</b>: %{y:,}<extra></extra>'
            ))

            fig_ts.update_layout(
                title=dict(
                    text="📈 Daily Demand Over Time",
                    font=dict(size=16)
                ),
                xaxis_title="Date",
                yaxis_title="Quantity",
                height=450,
                hovermode='x unified',
                showlegend=True,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )

            st.plotly_chart(fig_ts, use_container_width=True, config=self.chart_config)

            # === DISH ANALYSIS ===
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("🏆 Top Performing Dishes")
                top_dishes = df.groupby('variantname')['Quantity'].sum().nlargest(10).reset_index()

                fig_bar = px.bar(
                    top_dishes,
                    x='Quantity',
                    y='variantname',
                    orientation='h',
                    title="Top 10 Dishes by Total Demand",
                    color='Quantity',
                    color_continuous_scale='viridis'
                )

                fig_bar.update_layout(
                    height=400,
                    yaxis={'categoryorder': 'total ascending'},
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )

                st.plotly_chart(fig_bar, use_container_width=True, config=self.chart_config)

            with col2:
                st.subheader("📅 Weekly Demand Pattern")
                df_weekday = df.copy()
                df_weekday['weekday'] = pd.to_datetime(df_weekday['deliverydate']).dt.day_name()
                weekday_demand = df_weekday.groupby('weekday')['Quantity'].mean().reset_index()

                # Reorder weekdays
                weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                weekday_demand['weekday'] = pd.Categorical(weekday_demand['weekday'], categories=weekday_order, ordered=True)
                weekday_demand = weekday_demand.sort_values('weekday')

                fig_week = px.bar(
                    weekday_demand,
                    x='weekday',
                    y='Quantity',
                    title="Average Demand by Weekday",
                    color='Quantity',
                    color_continuous_scale='plasma'
                )

                fig_week.update_layout(
                    height=400,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )

                st.plotly_chart(fig_week, use_container_width=True, config=self.chart_config)

        except Exception as e:
            st.error(f"❌ Error creating EDA dashboard: {str(e)}")

    def plot_historical_data(self, df, selected_dishes):
        """
        Plot historical data patterns for selected dishes.

        Args:
            df: Historical data DataFrame
            selected_dishes: List of selected dish names
        """
        try:
            st.subheader("📈 Historical Data Analysis")

            if not selected_dishes:
                st.warning("⚠️ Please select dishes to analyze")
                return

            # Limit dishes for performance
            dishes_to_plot = selected_dishes[:6]

            # Create subplot structure
            rows = min(len(dishes_to_plot), 3)
            cols = 2 if len(dishes_to_plot) > 1 else 1

            if len(dishes_to_plot) <= 2:
                rows, cols = 1, len(dishes_to_plot)
            elif len(dishes_to_plot) <= 4:
                rows, cols = 2, 2
            else:
                rows, cols = 3, 2

            fig = make_subplots(
                rows=rows,
                cols=cols,
                subplot_titles=[f"🍽️ {dish}" for dish in dishes_to_plot],
                vertical_spacing=0.12,
                horizontal_spacing=0.1
            )

            for i, dish in enumerate(dishes_to_plot):
                row = (i // cols) + 1
                col = (i % cols) + 1

                dish_data = df[df['variantname'].str.strip() == dish.strip()]

                if not dish_data.empty:
                    daily_data = dish_data.groupby('deliverydate')['Quantity'].sum().reset_index()
                    daily_data['deliverydate'] = pd.to_datetime(daily_data['deliverydate'])
                    daily_data = daily_data.sort_values('deliverydate')

                    # Add line trace
                    fig.add_trace(
                        go.Scatter(
                            x=daily_data['deliverydate'],
                            y=daily_data['Quantity'],
                            mode='lines+markers',
                            name=dish,
                            line=dict(color=self.colors[i % len(self.colors)], width=2),
                            marker=dict(size=4),
                            hovertemplate=f'<b>{dish}</b><br>Date: %{{x}}<br>Quantity: %{{y:,}}<extra></extra>',
                            showlegend=False
                        ),
                        row=row,
                        col=col
                    )

            fig.update_layout(
                title=dict(
                    text="📊 Historical Demand Patterns",
                    font=dict(size=18)
                ),
                height=300 * rows,
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )

            fig.update_xaxes(title_text="Date", showgrid=True, gridcolor='lightgray')
            fig.update_yaxes(title_text="Quantity", showgrid=True, gridcolor='lightgray')

            st.plotly_chart(fig, use_container_width=True, config=self.chart_config)

            # === STATISTICAL SUMMARY ===
            st.subheader("📋 Statistical Summary")

            stats_data = []
            for dish in selected_dishes:
                dish_data = df[df['variantname'].str.strip() == dish.strip()]

                if not dish_data.empty:
                    daily_data = dish_data.groupby('deliverydate')['Quantity'].sum()

                    stats_data.append({
                        '🍽️ Dish': dish,
                        '📦 Total Demand': f"{daily_data.sum():,}",
                        '📈 Avg Daily': f"{daily_data.mean():.1f}",
                        '🚀 Peak Day': f"{daily_data.max():,}",
                        '📉 Min Day': f"{daily_data.min():,}",
                        '📊 Std Dev': f"{daily_data.std():.1f}",
                        '📅 Data Points': f"{len(daily_data):,}"
                    })

            if stats_data:
                stats_df = pd.DataFrame(stats_data)
                st.dataframe(stats_df, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Error plotting historical data: {str(e)}")

    def plot_forecasts(self, df, selected_dishes, all_forecasts, target_dates, main_model):
        """
        Plot forecast results with historical data comparison.

        Args:
            df: Historical data DataFrame
            selected_dishes: List of selected dishes
            all_forecasts: Dictionary of all forecasts
            target_dates: List of target forecast dates
            main_model: Primary model for forecasting
        """
        try:
            st.subheader("🔮 Forecast Visualization")

            for dish in selected_dishes[:4]:  # Limit for performance
                if dish not in all_forecasts or main_model not in all_forecasts[dish]:
                    st.warning(f"⚠️ No forecast available for {dish} with {main_model}")
                    continue

                # Get historical data
                dish_data = df[df['variantname'].str.strip() == dish.strip()]

                if dish_data.empty:
                    st.warning(f"⚠️ No historical data for {dish}")
                    continue

                daily_data = dish_data.groupby('deliverydate')['Quantity'].sum().reset_index()
                daily_data['deliverydate'] = pd.to_datetime(daily_data['deliverydate'])
                daily_data = daily_data.sort_values('deliverydate')

                # Get forecast data
                forecast_values = all_forecasts[dish][main_model]
                forecast_df = pd.DataFrame({
                    'date': target_dates,
                    'forecast': forecast_values
                })

                # Create interactive plot
                fig = go.Figure()

                # Historical data
                fig.add_trace(go.Scatter(
                    x=daily_data['deliverydate'],
                    y=daily_data['Quantity'],
                    mode='lines+markers',
                    name='📊 Historical Data',
                    line=dict(color=self.color_palette['primary'], width=2),
                    marker=dict(size=5),
                    hovertemplate='<b>Historical</b><br>Date: %{x}<br>Quantity: %{y:,}<extra></extra>'
                ))

                # Forecast data
                fig.add_trace(go.Scatter(
                    x=forecast_df['date'],
                    y=forecast_df['forecast'],
                    mode='lines+markers',
                    name=f'🔮 Forecast ({main_model})',
                    line=dict(color=self.color_palette['secondary'], width=3, dash='dash'),
                    marker=dict(size=8, symbol='diamond'),
                    hovertemplate=f'<b>Forecast ({main_model})</b><br>Date: %{{x}}<br>Quantity: %{{y:,}}<extra></extra>'
                ))

                # Add vertical separator line
                if not daily_data.empty:
                    last_date = daily_data['deliverydate'].max()
                    fig.add_vline(
                        x=last_date,
                        line_dash="dot",
                        line_color="gray",
                        annotation_text="📅 Forecast Start",
                        annotation_position="top"
                    )

                fig.update_layout(
                    title=dict(
                        text=f"📊 {dish} - Historical vs Forecast",
                        font=dict(size=16)
                    ),
                    xaxis_title="Date",
                    yaxis_title="Quantity",
                    height=450,
                    hovermode='x unified',
                    legend=dict(x=0, y=1, bgcolor='rgba(255,255,255,0.8)'),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )

                fig.update_xaxes(showgrid=True, gridcolor='lightgray')
                fig.update_yaxes(showgrid=True, gridcolor='lightgray')

                st.plotly_chart(fig, use_container_width=True, config=self.chart_config)

                # Forecast summary metrics
                total_forecast = sum(forecast_values)
                avg_forecast = total_forecast / len(forecast_values) if forecast_values else 0

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("📦 Weekly Forecast", f"{total_forecast:,}")

                with col2:
                    st.metric("📊 Daily Average", f"{avg_forecast:.1f}")

                with col3:
                    if not daily_data.empty:
                        historical_avg = daily_data['Quantity'].mean()
                        change = ((avg_forecast - historical_avg) / historical_avg * 100) if historical_avg > 0 else 0
                        st.metric("📈 vs Historical", f"{change:+.1f}%")
                    else:
                        st.metric("📈 vs Historical", "N/A")

                with col4:
                    volatility = np.std(forecast_values) if len(forecast_values) > 1 else 0
                    st.metric("📊 Forecast Volatility", f"{volatility:.1f}")

                st.markdown("---")

        except Exception as e:
            st.error(f"❌ Error plotting forecasts: {str(e)}")

    def compare_models(self, dish, all_forecasts, selected_models, target_dates):
        """
        Compare different models for a single dish.

        Args:
            dish: Dish name
            all_forecasts: Dictionary of all forecasts
            selected_models: List of selected models
            target_dates: List of target dates
        """
        try:
            st.subheader(f"🔬 Model Comparison: {dish}")

            if dish not in all_forecasts:
                st.warning(f"⚠️ No forecasts available for {dish}")
                return

            # Prepare comparison data
            comparison_data = []
            model_colors = {}

            for i, model in enumerate(selected_models):
                if model in all_forecasts[dish]:
                    forecast = all_forecasts[dish][model]
                    model_colors[model] = self.colors[i % len(self.colors)]

                    for j, (date, value) in enumerate(zip(target_dates, forecast)):
                        comparison_data.append({
                            'Date': date.strftime('%a %d/%m'),
                            'Day': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'][j],
                            'Model': model,
                            'Forecast': value,
                            'DayNum': j
                        })

            if not comparison_data:
                st.warning("⚠️ No model forecasts to compare")
                return

            comparison_df = pd.DataFrame(comparison_data)

            # Create comparison plot
            fig = go.Figure()

            for model in selected_models:
                if model in all_forecasts[dish]:
                    model_data = comparison_df[comparison_df['Model'] == model]

                    fig.add_trace(go.Scatter(
                        x=model_data['Day'],
                        y=model_data['Forecast'],
                        mode='lines+markers',
                        name=f'🤖 {model}',
                        line=dict(color=model_colors.get(model, self.colors[0]), width=3),
                        marker=dict(size=8),
                        hovertemplate=f'<b>{model}</b><br>Day: %{{x}}<br>Forecast: %{{y:,}}<extra></extra>'
                    ))

            fig.update_layout(
                title=dict(
                    text=f"📊 Model Comparison for {dish}",
                    font=dict(size=16)
                ),
                xaxis_title="Day of Week",
                yaxis_title="Forecasted Quantity",
                height=500,
                hovermode='x unified',
                legend=dict(bgcolor='rgba(255,255,255,0.8)'),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )

            fig.update_xaxes(showgrid=True, gridcolor='lightgray')
            fig.update_yaxes(showgrid=True, gridcolor='lightgray')

            st.plotly_chart(fig, use_container_width=True, config=self.chart_config)

            # === MODEL COMPARISON TABLE ===
            st.subheader("📋 Detailed Forecast Comparison")

            model_summary = []
            for model in selected_models:
                if model in all_forecasts[dish]:
                    forecast = all_forecasts[dish][model]
                    model_summary.append({
                        '🤖 Model': model,
                        '📦 Total Weekly': f"{sum(forecast):,}",
                        '📊 Average Daily': f"{np.mean(forecast):.1f}",
                        '🚀 Peak Day': f"{max(forecast):,}",
                        '📉 Min Day': f"{min(forecast):,}",
                        '📊 Volatility': f"{np.std(forecast):.1f}",
                        '📅 Monday': f"{forecast[0]:,}",
                        '📅 Tuesday': f"{forecast[1]:,}",
                        '📅 Wednesday': f"{forecast[2]:,}",
                        '📅 Thursday': f"{forecast[3]:,}",
                        '📅 Friday': f"{forecast[4]:,}"
                    })

            if model_summary:
                summary_df = pd.DataFrame(model_summary)
                st.dataframe(summary_df, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Error in model comparison: {str(e)}")


if __name__ == "__main__":
    # Test the visualization module
    st.title("🧪 Visualization Module Test")

    viz = Visualization()
    st.success("✅ Visualization module initialized successfully!")
