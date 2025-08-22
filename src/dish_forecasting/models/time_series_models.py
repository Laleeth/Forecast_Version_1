"""
Enhanced Time Series Models for Dish Forecasting Pipeline

This module contains multiple time series forecasting algorithms including
deep learning models while preserving the original user logic.

Author: Lalith Thomala (Enhanced version)
Original Logic: User's Implementation 
Version: 1.0.1 (Fixed with more models)
Date: August 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from prophet import Prophet

# Enhanced time series models (optional)
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU
    from sklearn.preprocessing import MinMaxScaler
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# Import utility functions
try:
    from ..utils.utils import evaluate, get_similar_dishes, forecast_new_dish, forecast_insufficient_data
except ImportError:
    # Fallback imports for standalone usage
    def evaluate(y_true, y_pred):
        return {
            "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": mean_squared_error(y_true, y_pred, squared=False)
        }

__author__ = "Lalith Thomala"
__version__ = "1.0.1"

class TimeSeriesModels:
    """
    Enhanced time series forecasting models with additional algorithms.

    Includes ALL available models:
    - Core models: Prophet, Linear Regression, Random Forest, XGBoost, Moving Average
    - Statistical models: ARIMA, SARIMA, Exponential Smoothing, Seasonal Decompose
    - Deep learning: LSTM, GRU, Ensemble Neural Networks
    """

    def __init__(self):
        """Initialize the TimeSeriesModels class."""
        self.available_models = self._get_available_models()
        self.model_descriptions = self._get_model_descriptions()

    def _get_available_models(self):
        """Get list of available models based on installed packages."""
        # Core models (always available)
        models = [
            'Prophet',
            'Linear Regression', 
            'Random Forest',
            'XGBoost',
            'Moving Average'
        ]

        # Add statistical models if available
        if STATSMODELS_AVAILABLE:
            models.extend([
                'ARIMA',
                'SARIMA',
                'Exponential Smoothing',
                'Seasonal Decompose'
            ])

        # Add deep learning models if available
        if TENSORFLOW_AVAILABLE:
            models.extend([
                'LSTM',
                'GRU',
                'Neural Network Ensemble'
            ])

        return models

    def _get_model_descriptions(self):
        """Get descriptions for all available models."""
        descriptions = {
            # Core Models
            'Prophet': 'Facebook Prophet - Best for seasonal patterns and trends',
            'Linear Regression': 'Linear model with time features - Simple and interpretable',
            'Random Forest': 'Ensemble model - Good for complex non-linear patterns',
            'XGBoost': 'Gradient boosting - High accuracy for large datasets',
            'Moving Average': 'Simple baseline - Average of recent values',

            # Statistical Models
            'ARIMA': 'Auto-Regressive Integrated Moving Average - Classical time series',
            'SARIMA': 'Seasonal ARIMA - Handles seasonal patterns statistically',
            'Exponential Smoothing': 'Holt-Winters - Seasonal exponential smoothing',
            'Seasonal Decompose': 'Trend + Seasonality - Statistical decomposition',

            # Deep Learning Models
            'LSTM': 'Long Short-Term Memory - Neural network for sequences',
            'GRU': 'Gated Recurrent Unit - Efficient alternative to LSTM',
            'Neural Network Ensemble': 'Multiple neural networks combined'
        }

        return {model: descriptions.get(model, 'Advanced forecasting model') 
                for model in self.available_models}

    def get_available_models(self):
        """Public method to get available models."""
        return self.available_models

    def get_model_descriptions(self):
        """Public method to get model descriptions."""
        return self.model_descriptions

    def forecast_models(self, df, dish, target_dates, selected_models=None):
        """
        Main forecasting method with all models (PRESERVES ORIGINAL LOGIC).

        Args:
            df: Historical data DataFrame
            dish: Dish name to forecast
            target_dates: List of target forecast dates
            selected_models: List of models to use (None = all available)

        Returns:
            tuple: (forecasts_dict, error_message)
        """
        if selected_models is None:
            selected_models = self.available_models

        result = {}
        dish_df = df[df['variantname'].str.strip() == dish.strip()]

        # === ORIGINAL LOGIC: Handle new dishes (no history) ===
        if dish_df.empty:
            try:
                from ..utils.utils import forecast_new_dish
                forecasts, explanation = forecast_new_dish(df, dish, df['variantname'].unique(), target_dates)
            except ImportError:
                forecasts, explanation = [1] * 5, "New dish - using default values"

            return {
                'New Dish Strategy': {
                    'forecast': forecasts,
                    'score': {'MAE': 0, 'RMSE': 0},
                    'explanation': explanation
                }
            }, None

        # === ORIGINAL LOGIC: Prepare daily data ===
        daily = (
            dish_df.groupby('deliverydate')['Quantity']
            .sum()
            .reset_index()
            .sort_values('deliverydate')
        )
        daily['ds'] = pd.to_datetime(daily['deliverydate'])
        daily['y'] = daily['Quantity']
        daily['weekday'] = daily['ds'].dt.dayofweek
        daily = daily[daily['weekday'] < 5]  # Only weekdays

        # === ORIGINAL LOGIC: Handle insufficient data ===
        if len(daily) < 10:
            try:
                from ..utils.utils import forecast_insufficient_data
                forecasts, explanation = forecast_insufficient_data(df, dish, target_dates)
            except ImportError:
                forecasts, explanation = [1] * 5, "Insufficient data - using default values"

            return {
                'Limited Data Strategy': {
                    'forecast': forecasts,
                    'score': {'MAE': 0, 'RMSE': 0},
                    'explanation': explanation
                }
            }, None

        # === RUN SELECTED MODELS ===

        # Core models (preserve exact logic)
        if 'Prophet' in selected_models:
            prophet_result = self._forecast_prophet(daily, target_dates)
            if prophet_result:
                result['Prophet'] = prophet_result

        if 'Linear Regression' in selected_models:
            lr_result = self._forecast_linear_regression(daily, target_dates)
            if lr_result:
                result['Linear Regression'] = lr_result

        if 'Random Forest' in selected_models:
            rf_result = self._forecast_random_forest(daily, target_dates)
            if rf_result:
                result['Random Forest'] = rf_result

        if 'XGBoost' in selected_models:
            xgb_result = self._forecast_xgboost(daily, target_dates)
            if xgb_result:
                result['XGBoost'] = xgb_result

        if 'Moving Average' in selected_models:
            ma_result = self._forecast_moving_average(daily, target_dates)
            if ma_result:
                result['Moving Average'] = ma_result

        # Statistical models
        if 'ARIMA' in selected_models and STATSMODELS_AVAILABLE:
            arima_result = self._forecast_arima(daily, target_dates)
            if arima_result:
                result['ARIMA'] = arima_result

        if 'SARIMA' in selected_models and STATSMODELS_AVAILABLE:
            sarima_result = self._forecast_sarima(daily, target_dates)
            if sarima_result:
                result['SARIMA'] = sarima_result

        if 'Exponential Smoothing' in selected_models and STATSMODELS_AVAILABLE:
            es_result = self._forecast_exponential_smoothing(daily, target_dates)
            if es_result:
                result['Exponential Smoothing'] = es_result

        if 'Seasonal Decompose' in selected_models and STATSMODELS_AVAILABLE:
            sd_result = self._forecast_seasonal_decompose(daily, target_dates)
            if sd_result:
                result['Seasonal Decompose'] = sd_result

        # Deep learning models
        if 'LSTM' in selected_models and TENSORFLOW_AVAILABLE:
            lstm_result = self._forecast_lstm(daily, target_dates)
            if lstm_result:
                result['LSTM'] = lstm_result

        if 'GRU' in selected_models and TENSORFLOW_AVAILABLE:
            gru_result = self._forecast_gru(daily, target_dates)
            if gru_result:
                result['GRU'] = gru_result

        if 'Neural Network Ensemble' in selected_models and TENSORFLOW_AVAILABLE:
            ensemble_result = self._forecast_neural_ensemble(daily, target_dates)
            if ensemble_result:
                result['Neural Network Ensemble'] = ensemble_result

        return result, None

    # === ORIGINAL MODEL IMPLEMENTATIONS (EXACT LOGIC PRESERVED) ===

    def _forecast_prophet(self, daily, target_dates):
        """Facebook Prophet model (ORIGINAL LOGIC)."""
        try:
            m = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False)
            m.fit(daily[['ds', 'y']])

            future_df = pd.DataFrame({'ds': target_dates})
            forecast = m.predict(future_df)

            prophet_forecast = forecast[['ds', 'yhat']].copy()
            prophet_forecast['yhat'] = prophet_forecast['yhat'].round().astype(int)
            prophet_forecast['yhat'] = np.maximum(prophet_forecast['yhat'], 0)  # Ensure non-negative

            # Calculate historical performance
            historical_pred = m.predict(daily[['ds']])
            prophet_scores = evaluate(daily['y'], historical_pred['yhat'])

            return {
                'forecast': prophet_forecast['yhat'].tolist(),
                'score': prophet_scores
            }
        except Exception as e:
            st.warning(f"Prophet model failed: {str(e)}")
            return None

    def _forecast_linear_regression(self, daily, target_dates):
        """Linear Regression with time features (ORIGINAL LOGIC)."""
        try:
            df_ml = daily.copy()
            df_ml['day_of_week'] = df_ml['ds'].dt.dayofweek
            df_ml['day_of_year'] = df_ml['ds'].dt.dayofyear
            df_ml['week_of_year'] = df_ml['ds'].dt.isocalendar().week
            df_ml['month'] = df_ml['ds'].dt.month
            df_ml['trend'] = range(len(df_ml))

            X = df_ml[['day_of_week', 'day_of_year', 'week_of_year', 'month', 'trend']]
            y = df_ml['y']

            # Create features for target dates
            X_future = pd.DataFrame({
                'day_of_week': [date.weekday() for date in target_dates],
                'day_of_year': [date.timetuple().tm_yday for date in target_dates],
                'week_of_year': [date.isocalendar()[1] for date in target_dates],
                'month': [date.month for date in target_dates],
                'trend': [len(df_ml) + i for i in range(5)]
            })

            lr = LinearRegression()
            lr.fit(X, y)
            lr_pred = lr.predict(X_future)
            lr_pred = np.maximum(lr_pred, 0).round().astype(int)  # Ensure non-negative

            lr_scores = evaluate(y, lr.predict(X))

            return {
                'forecast': lr_pred.tolist(),
                'score': lr_scores
            }
        except Exception as e:
            st.warning(f"Linear Regression failed: {str(e)}")
            return None

    def _forecast_random_forest(self, daily, target_dates):
        """Random Forest model (ORIGINAL LOGIC)."""
        try:
            df_ml = daily.copy()
            df_ml['day_of_week'] = df_ml['ds'].dt.dayofweek
            df_ml['day_of_year'] = df_ml['ds'].dt.dayofyear
            df_ml['week_of_year'] = df_ml['ds'].dt.isocalendar().week
            df_ml['month'] = df_ml['ds'].dt.month
            df_ml['trend'] = range(len(df_ml))

            X = df_ml[['day_of_week', 'day_of_year', 'week_of_year', 'month', 'trend']]
            y = df_ml['y']

            X_future = pd.DataFrame({
                'day_of_week': [date.weekday() for date in target_dates],
                'day_of_year': [date.timetuple().tm_yday for date in target_dates],
                'week_of_year': [date.isocalendar()[1] for date in target_dates],
                'month': [date.month for date in target_dates],
                'trend': [len(df_ml) + i for i in range(5)]
            })

            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X, y)
            rf_pred = rf.predict(X_future)
            rf_pred = np.maximum(rf_pred, 0).round().astype(int)

            rf_scores = evaluate(y, rf.predict(X))

            return {
                'forecast': rf_pred.tolist(),
                'score': rf_scores
            }
        except Exception as e:
            st.warning(f"Random Forest failed: {str(e)}")
            return None

    def _forecast_xgboost(self, daily, target_dates):
        """XGBoost model (ORIGINAL LOGIC)."""
        try:
            df_ml = daily.copy()
            df_ml['day_of_week'] = df_ml['ds'].dt.dayofweek
            df_ml['day_of_year'] = df_ml['ds'].dt.dayofyear
            df_ml['week_of_year'] = df_ml['ds'].dt.isocalendar().week
            df_ml['month'] = df_ml['ds'].dt.month
            df_ml['trend'] = range(len(df_ml))

            X = df_ml[['day_of_week', 'day_of_year', 'week_of_year', 'month', 'trend']]
            y = df_ml['y']

            X_future = pd.DataFrame({
                'day_of_week': [date.weekday() for date in target_dates],
                'day_of_year': [date.timetuple().tm_yday for date in target_dates],
                'week_of_year': [date.isocalendar()[1] for date in target_dates],
                'month': [date.month for date in target_dates],
                'trend': [len(df_ml) + i for i in range(5)]
            })

            xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
            xgb_model.fit(X, y)
            xgb_pred = xgb_model.predict(X_future)
            xgb_pred = np.maximum(xgb_pred, 0).round().astype(int)

            xgb_scores = evaluate(y, xgb_model.predict(X))

            return {
                'forecast': xgb_pred.tolist(),
                'score': xgb_scores
            }
        except Exception as e:
            st.warning(f"XGBoost failed: {str(e)}")
            return None

    def _forecast_moving_average(self, daily, target_dates):
        """Moving Average baseline (ORIGINAL LOGIC)."""
        try:
            # Use last 4 weeks average for each weekday
            ma_pred = []
            for target_date in target_dates:
                target_weekday = target_date.weekday()
                weekday_data = daily[daily['weekday'] == target_weekday]['y']

                if len(weekday_data) >= 4:
                    ma_value = weekday_data.tail(4).mean()
                elif len(weekday_data) > 0:
                    ma_value = weekday_data.mean()
                else:
                    ma_value = 0

                ma_pred.append(max(0, round(ma_value)))

            return {
                'forecast': ma_pred,
                'score': {'MAE': 0, 'RMSE': 0}  # Simple baseline, scores not meaningful
            }
        except Exception as e:
            st.warning(f"Moving Average failed: {str(e)}")
            return None

    # === NEW ENHANCED MODELS ===

    def _forecast_arima(self, daily, target_dates):
        """ARIMA model (ENHANCED)."""
        if not STATSMODELS_AVAILABLE:
            return None

        try:
            # Prepare data - use daily quantities
            ts_data = daily.set_index('ds')['y'].resample('D').sum().fillna(0)

            # Fit ARIMA model with simple parameters
            model = ARIMA(ts_data, order=(1, 1, 1))
            fitted_model = model.fit()

            # Generate forecast for target dates
            forecast_steps = len(target_dates)
            forecast = fitted_model.forecast(steps=forecast_steps)
            forecast = np.maximum(forecast, 0).round().astype(int)

            # Calculate in-sample performance
            fitted_values = fitted_model.fittedvalues
            if len(fitted_values) > 0:
                arima_scores = evaluate(ts_data.values[-len(fitted_values):], fitted_values)
            else:
                arima_scores = {'MAE': 0, 'RMSE': 0}

            return {
                'forecast': forecast.tolist(),
                'score': arima_scores
            }
        except Exception as e:
            st.warning(f"ARIMA model failed: {str(e)}")
            return None

    def _forecast_sarima(self, daily, target_dates):
        """SARIMA model (NEW ENHANCED MODEL)."""
        if not STATSMODELS_AVAILABLE:
            return None

        try:
            # Prepare data
            ts_data = daily.set_index('ds')['y'].resample('D').sum().fillna(0)

            if len(ts_data) < 14:  # Need at least 2 weeks for seasonal
                return None

            # Fit SARIMA model with weekly seasonality
            model = SARIMAX(ts_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
            fitted_model = model.fit(disp=False)

            # Generate forecast
            forecast_steps = len(target_dates)
            forecast = fitted_model.forecast(steps=forecast_steps)
            forecast = np.maximum(forecast, 0).round().astype(int)

            # Calculate performance
            fitted_values = fitted_model.fittedvalues
            sarima_scores = evaluate(ts_data.values, fitted_values)

            return {
                'forecast': forecast.tolist(),
                'score': sarima_scores
            }
        except Exception as e:
            st.warning(f"SARIMA model failed: {str(e)}")
            return None

    def _forecast_exponential_smoothing(self, daily, target_dates):
        """Exponential Smoothing model (ENHANCED)."""
        if not STATSMODELS_AVAILABLE:
            return None

        try:
            # Prepare data
            ts_data = daily.set_index('ds')['y'].resample('D').sum().fillna(0)

            # Fit Exponential Smoothing model
            model = ExponentialSmoothing(
                ts_data,
                trend='add',
                seasonal='add' if len(ts_data) > 14 else None,
                seasonal_periods=7 if len(ts_data) > 14 else None
            )
            fitted_model = model.fit()

            # Generate forecast
            forecast_steps = len(target_dates)
            forecast = fitted_model.forecast(steps=forecast_steps)
            forecast = np.maximum(forecast, 0).round().astype(int)

            # Calculate performance
            fitted_values = fitted_model.fittedvalues
            es_scores = evaluate(ts_data.values, fitted_values)

            return {
                'forecast': forecast.tolist(),
                'score': es_scores
            }
        except Exception as e:
            st.warning(f"Exponential Smoothing failed: {str(e)}")
            return None

    def _forecast_seasonal_decompose(self, daily, target_dates):
        """Seasonal Decompose + Trend forecasting (ENHANCED)."""
        if not STATSMODELS_AVAILABLE:
            return None

        try:
            # Prepare data
            ts_data = daily.set_index('ds')['y'].resample('D').sum().fillna(0)

            if len(ts_data) < 14:  # Need at least 2 weeks for weekly seasonality
                return None

            # Perform seasonal decomposition
            decomposition = seasonal_decompose(ts_data, model='additive', period=7)

            # Simple trend extrapolation + seasonal pattern
            trend = decomposition.trend.dropna()
            seasonal = decomposition.seasonal

            # Linear trend extrapolation
            if len(trend) > 0:
                trend_values = trend.values
                x = np.arange(len(trend_values))
                trend_slope = np.polyfit(x, trend_values, 1)[0]

                # Forecast
                forecast_values = []
                for i, target_date in enumerate(target_dates):
                    # Get seasonal component for this day of week
                    day_of_week = target_date.weekday()
                    seasonal_component = seasonal.iloc[day_of_week % 7] if len(seasonal) > 7 else 0

                    # Extrapolate trend
                    trend_component = trend_values[-1] + trend_slope * (i + 1)

                    # Combine
                    forecast_value = trend_component + seasonal_component
                    forecast_values.append(max(0, round(forecast_value)))

                # Calculate performance on fitted values
                fitted = decomposition.trend + decomposition.seasonal
                fitted = fitted.dropna()
                actual = ts_data[fitted.index]
                decompose_scores = evaluate(actual.values, fitted.values)

                return {
                    'forecast': forecast_values,
                    'score': decompose_scores
                }

            return None
        except Exception as e:
            st.warning(f"Seasonal Decompose failed: {str(e)}")
            return None

    def _forecast_lstm(self, daily, target_dates):
        """LSTM Neural Network model (ENHANCED)."""
        if not TENSORFLOW_AVAILABLE:
            return None

        try:
            # Prepare data
            ts_data = daily['y'].values.reshape(-1, 1)

            if len(ts_data) < 20:  # Need sufficient data for LSTM
                return None

            # Scale data
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(ts_data)

            # Create sequences
            sequence_length = min(10, len(scaled_data) // 2)
            X, y = [], []

            for i in range(sequence_length, len(scaled_data)):
                X.append(scaled_data[i-sequence_length:i, 0])
                y.append(scaled_data[i, 0])

            X, y = np.array(X), np.array(y)

            if len(X) < 5:  # Need minimum samples
                return None

            X = np.reshape(X, (X.shape[0], X.shape[1], 1))

            # Build LSTM model
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)),
                Dropout(0.2),
                LSTM(50),
                Dropout(0.2),
                Dense(1)
            ])

            model.compile(optimizer='adam', loss='mse')

            # Train model (with minimal epochs for speed)
            model.fit(X, y, epochs=10, batch_size=1, verbose=0)

            # Generate forecast
            last_sequence = scaled_data[-sequence_length:].reshape(1, sequence_length, 1)
            forecast_scaled = []

            for _ in range(len(target_dates)):
                pred = model.predict(last_sequence, verbose=0)
                forecast_scaled.append(pred[0, 0])

                # Update sequence for next prediction
                last_sequence = np.roll(last_sequence, -1, axis=1)
                last_sequence[0, -1, 0] = pred[0, 0]

            # Inverse transform
            forecast_scaled = np.array(forecast_scaled).reshape(-1, 1)
            forecast = scaler.inverse_transform(forecast_scaled)
            forecast = np.maximum(forecast, 0).round().astype(int).flatten()

            # Calculate performance (on training data)
            train_pred = model.predict(X, verbose=0)
            train_pred_rescaled = scaler.inverse_transform(train_pred)
            train_actual_rescaled = scaler.inverse_transform(y.reshape(-1, 1))

            lstm_scores = evaluate(train_actual_rescaled.flatten(), train_pred_rescaled.flatten())

            return {
                'forecast': forecast.tolist(),
                'score': lstm_scores
            }
        except Exception as e:
            st.warning(f"LSTM model failed: {str(e)}")
            return None

    def _forecast_gru(self, daily, target_dates):
        """GRU Neural Network model (NEW)."""
        if not TENSORFLOW_AVAILABLE:
            return None

        try:
            # Prepare data (similar to LSTM)
            ts_data = daily['y'].values.reshape(-1, 1)

            if len(ts_data) < 20:
                return None

            # Scale data
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(ts_data)

            # Create sequences
            sequence_length = min(10, len(scaled_data) // 2)
            X, y = [], []

            for i in range(sequence_length, len(scaled_data)):
                X.append(scaled_data[i-sequence_length:i, 0])
                y.append(scaled_data[i, 0])

            X, y = np.array(X), np.array(y)

            if len(X) < 5:
                return None

            X = np.reshape(X, (X.shape[0], X.shape[1], 1))

            # Build GRU model
            model = Sequential([
                GRU(50, return_sequences=True, input_shape=(sequence_length, 1)),
                Dropout(0.2),
                GRU(50),
                Dropout(0.2),
                Dense(1)
            ])

            model.compile(optimizer='adam', loss='mse')

            # Train model
            model.fit(X, y, epochs=10, batch_size=1, verbose=0)

            # Generate forecast (similar to LSTM)
            last_sequence = scaled_data[-sequence_length:].reshape(1, sequence_length, 1)
            forecast_scaled = []

            for _ in range(len(target_dates)):
                pred = model.predict(last_sequence, verbose=0)
                forecast_scaled.append(pred[0, 0])

                last_sequence = np.roll(last_sequence, -1, axis=1)
                last_sequence[0, -1, 0] = pred[0, 0]

            # Inverse transform
            forecast_scaled = np.array(forecast_scaled).reshape(-1, 1)
            forecast = scaler.inverse_transform(forecast_scaled)
            forecast = np.maximum(forecast, 0).round().astype(int).flatten()

            # Calculate performance
            train_pred = model.predict(X, verbose=0)
            train_pred_rescaled = scaler.inverse_transform(train_pred)
            train_actual_rescaled = scaler.inverse_transform(y.reshape(-1, 1))

            gru_scores = evaluate(train_actual_rescaled.flatten(), train_pred_rescaled.flatten())

            return {
                'forecast': forecast.tolist(),
                'score': gru_scores
            }
        except Exception as e:
            st.warning(f"GRU model failed: {str(e)}")
            return None

    def _forecast_neural_ensemble(self, daily, target_dates):
        """Neural Network Ensemble model (NEW)."""
        if not TENSORFLOW_AVAILABLE:
            return None

        try:
            # Run both LSTM and GRU, then ensemble
            lstm_result = self._forecast_lstm(daily, target_dates)
            gru_result = self._forecast_gru(daily, target_dates)

            if lstm_result and gru_result:
                # Simple ensemble: average of LSTM and GRU
                lstm_forecast = np.array(lstm_result['forecast'])
                gru_forecast = np.array(gru_result['forecast'])

                ensemble_forecast = ((lstm_forecast + gru_forecast) / 2).round().astype(int)

                # Average scores
                ensemble_scores = {
                    'MAE': (lstm_result['score']['MAE'] + gru_result['score']['MAE']) / 2,
                    'RMSE': (lstm_result['score']['RMSE'] + gru_result['score']['RMSE']) / 2
                }

                return {
                    'forecast': ensemble_forecast.tolist(),
                    'score': ensemble_scores
                }

            return None
        except Exception as e:
            st.warning(f"Neural Network Ensemble failed: {str(e)}")
            return None


if __name__ == "__main__":
    # Test the time series models
    st.title("🧪 Enhanced Time Series Models Test")

    models = TimeSeriesModels()
    st.write(f"Available models: {models.get_available_models()}")

    for model, description in models.get_model_descriptions().items():
        st.write(f"**{model}**: {description}")
