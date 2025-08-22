"""
Data Ingestion Module for Dish Forecasting Pipeline

This module handles data loading and validation from various sources including
file uploads and SQL databases.

Author: Lalith Thomala
Version: 1.0.1 (Fixed)
Date: August 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime

# Optional database drivers
try:
    import pymysql
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False

try:
    import psycopg2
    POSTGRESQL_AVAILABLE = True
except ImportError:
    POSTGRESQL_AVAILABLE = False

__author__ = "Lalith Thomala"
__version__ = "1.0.1"

class DataIngestion:
    """
    Handles data ingestion from various sources for the dish forecasting pipeline.

    Supports:
    - File uploads (CSV, Excel)
    - SQL databases (SQLite, MySQL, PostgreSQL)
    """

    def __init__(self):
        """Initialize the DataIngestion class."""
        self.df = None
        self.dish_list = None
        self.connection_status = {"connected": False, "source": None}

    def load_from_files(self):
        """
        Load data from uploaded files.

        Returns:
            tuple: (DataFrame, dish_list) or (None, None) if failed
        """
        st.subheader("📁 File Upload Data Source")

        col1, col2 = st.columns(2)

        with col1:
            data_file = st.file_uploader(
                "📊 Upload Main Dataset", 
                type=["csv", "xlsx"],
                help="Upload your main dataset containing delivery data with columns: deliverydate, variantname, Quantity, kitchenName, Week_num"
            )

        with col2:
            dishes_file = st.file_uploader(
                "🍽️ Upload Dishes List", 
                type=["csv", "xlsx"],
                help="Upload list of dishes to forecast (single column with dish names)"
            )

        if data_file and dishes_file:
            try:
                # Load main dataset
                with st.spinner("📊 Loading main dataset..."):
                    if data_file.name.endswith('.csv'):
                        df = pd.read_csv(data_file)
                    else:
                        df = pd.read_excel(data_file)

                # Load dish list
                with st.spinner("🍽️ Loading dishes list..."):
                    if dishes_file.name.endswith('.csv'):
                        dish_list_df = pd.read_csv(dishes_file)
                    else:
                        dish_list_df = pd.read_excel(dishes_file)

                # Validate main dataset
                if self._validate_main_dataset(df):
                    # Process dates
                    df['deliverydate'] = pd.to_datetime(df['deliverydate'], dayfirst=True, errors='coerce')

                    # Extract dish list
                    if dish_list_df.shape[1] >= 1:
                        dish_list = dish_list_df.iloc[:, 0].dropna().astype(str).tolist()

                        # Clean dish names
                        dish_list = [dish.strip() for dish in dish_list if dish.strip()]

                        self.connection_status = {"connected": True, "source": "File Upload"}

                        st.success(f"✅ Successfully loaded {len(df):,} records and {len(dish_list):,} dishes")

                        return df, dish_list
                    else:
                        st.error("❌ Dish list must have at least one column")
                        return None, None
                else:
                    return None, None

            except Exception as e:
                st.error(f"❌ Error loading files: {str(e)}")
                return None, None

        return None, None

    def load_from_sql(self):
        """
        Load data from SQL database.

        Returns:
            tuple: (DataFrame, dish_list) or (None, None) if failed
        """
        st.subheader("🗄️ SQL Database Data Source")

        # Database type selection
        available_dbs = ["SQLite"]
        if MYSQL_AVAILABLE:
            available_dbs.append("MySQL")
        if POSTGRESQL_AVAILABLE:
            available_dbs.append("PostgreSQL")

        db_type = st.selectbox(
            "Database Type",
            available_dbs,
            help="Select your database type"
        )

        if db_type == "SQLite":
            return self._load_from_sqlite()
        elif db_type == "MySQL":
            return self._load_from_mysql()
        elif db_type == "PostgreSQL":
            return self._load_from_postgresql()

    def _load_from_sqlite(self):
        """Load from SQLite database."""
        col1, col2 = st.columns(2)

        with col1:
            db_file = st.file_uploader(
                "📁 Upload SQLite Database",
                type=["db", "sqlite", "sqlite3"],
                help="Upload your SQLite database file"
            )

        with col2:
            if db_file:
                try:
                    # Save uploaded file temporarily
                    with open("temp_db.sqlite", "wb") as f:
                        f.write(db_file.getbuffer())

                    # Connect and show tables
                    conn = sqlite3.connect("temp_db.sqlite")
                    tables = pd.read_sql_query(
                        "SELECT name FROM sqlite_master WHERE type='table'", 
                        conn
                    ).values.flatten()

                    if len(tables) == 0:
                        st.error("❌ No tables found in database")
                        conn.close()
                        return None, None

                    main_table = st.selectbox("📊 Select Main Data Table", tables)
                    dish_table = st.selectbox("🍽️ Select Dishes Table", tables)

                    if st.button("🔗 Connect to SQLite", type="primary"):
                        with st.spinner("🔄 Loading data from SQLite..."):
                            # Load data
                            df = pd.read_sql_query(f"SELECT * FROM {main_table}", conn)
                            dish_df = pd.read_sql_query(f"SELECT * FROM {dish_table}", conn)
                            conn.close()

                            # Validate and process
                            if self._validate_main_dataset(df):
                                df['deliverydate'] = pd.to_datetime(df['deliverydate'], errors='coerce')
                                dish_list = dish_df.iloc[:, 0].dropna().astype(str).tolist()
                                dish_list = [dish.strip() for dish in dish_list if dish.strip()]

                                self.connection_status = {"connected": True, "source": "SQLite"}

                                st.success(f"✅ Connected successfully! Loaded {len(df):,} records and {len(dish_list):,} dishes")

                                return df, dish_list

                    if conn:
                        conn.close()

                except Exception as e:
                    st.error(f"❌ SQLite connection error: {str(e)}")

        return None, None

    def _load_from_mysql(self):
        """Load from MySQL database."""
        if not MYSQL_AVAILABLE:
            st.error("❌ MySQL not available. Install with: pip install pymysql")
            return None, None

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**🔗 Connection Settings**")
            host = st.text_input("🌐 Host", value="localhost", help="MySQL server hostname")
            port = st.number_input("🔌 Port", value=3306, help="MySQL server port")
            database = st.text_input("💾 Database Name", help="Name of the database")

        with col2:
            st.markdown("**🔐 Credentials**")
            username = st.text_input("👤 Username", help="MySQL username")
            password = st.text_input("🔐 Password", type="password", help="MySQL password")

        st.markdown("**📊 Table Configuration**")
        col1, col2 = st.columns(2)
        with col1:
            main_table = st.text_input("📊 Main Data Table Name", help="Table containing the main dataset")
        with col2:
            dish_table = st.text_input("🍽️ Dishes Table Name", help="Table containing the dishes list")

        if st.button("🔗 Connect to MySQL", type="primary"):
            if all([host, database, username, password, main_table, dish_table]):
                try:
                    with st.spinner("🔄 Connecting to MySQL..."):
                        conn = pymysql.connect(
                            host=host,
                            port=port,
                            user=username,
                            password=password,
                            database=database
                        )

                        # Load data
                        df = pd.read_sql_query(f"SELECT * FROM {main_table}", conn)
                        dish_df = pd.read_sql_query(f"SELECT * FROM {dish_table}", conn)
                        conn.close()

                        # Validate and process
                        if self._validate_main_dataset(df):
                            df['deliverydate'] = pd.to_datetime(df['deliverydate'], errors='coerce')
                            dish_list = dish_df.iloc[:, 0].dropna().astype(str).tolist()
                            dish_list = [dish.strip() for dish in dish_list if dish.strip()]

                            self.connection_status = {"connected": True, "source": "MySQL"}

                            st.success(f"✅ Connected successfully! Loaded {len(df):,} records and {len(dish_list):,} dishes")

                            return df, dish_list

                except Exception as e:
                    st.error(f"❌ MySQL connection error: {str(e)}")
            else:
                st.warning("⚠️ Please fill all connection fields")

        return None, None

    def _load_from_postgresql(self):
        """Load from PostgreSQL database."""
        if not POSTGRESQL_AVAILABLE:
            st.error("❌ PostgreSQL not available. Install with: pip install psycopg2-binary")
            return None, None

        # Similar implementation as MySQL
        st.info("PostgreSQL connection - Similar to MySQL implementation")
        return None, None

    def _validate_main_dataset(self, df):
        """
        Validate main dataset structure and data quality.

        Args:
            df: DataFrame to validate

        Returns:
            bool: True if valid, False otherwise
        """
        required_columns = ['deliverydate', 'variantname', 'Quantity', 'kitchenName', 'Week_num']

        # Check required columns
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            st.error(f"❌ Missing required columns: {', '.join(missing_columns)}")

            with st.expander("📋 Required Dataset Structure", expanded=True):
                st.markdown("""
                **Required Columns:**
                - `deliverydate`: Date of delivery (DD/MM/YYYY format)
                - `variantname`: Dish/item name (string)
                - `Quantity`: Number of items delivered (numeric)
                - `kitchenName`: Location/kitchen identifier (string)
                - `Week_num`: Week number for grouping (numeric)
                """)

            return False

        # Data type validation
        try:
            # Convert Quantity to numeric
            if df['Quantity'].dtype not in ['int64', 'float64']:
                df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')

            # Check for missing quantities
            if df['Quantity'].isna().any():
                missing_qty_count = df['Quantity'].isna().sum()
                st.warning(f"⚠️ Found {missing_qty_count:,} missing Quantity values - they will be filled with 0")
                df['Quantity'].fillna(0, inplace=True)

            # Check for negative quantities
            if (df['Quantity'] < 0).any():
                negative_count = (df['Quantity'] < 0).sum()
                st.warning(f"⚠️ Found {negative_count:,} negative Quantity values - they will be set to 0")
                df.loc[df['Quantity'] < 0, 'Quantity'] = 0

        except Exception as e:
            st.error(f"❌ Error processing Quantity column: {str(e)}")
            return False

        return True
