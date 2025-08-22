"""
Setup configuration for Dish Forecasting Pipeline (FIXED VERSION)

Author: Lalith Thomala
Version: 1.0.1 (Fixed)
Date: August 2025
Email: sai@bellabona.com
"""

from setuptools import setup, find_packages

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="dish-forecasting-pipeline",
    version="1.0.1",
    author="Lalith Thomala",
    author_email="sai@bellabona.com",
    description="End-to-end machine learning pipeline for dish demand forecasting (Fixed Version)",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/lalith-thomala/dish-forecasting-pipeline",

    packages=find_packages(where="src"),
    package_dir={"": "src"},

    install_requires=requirements,

    extras_require={
        "statistical": ["statsmodels>=0.14.0", "scipy>=1.10.0"],
        "deep_learning": ["tensorflow>=2.13.0", "keras>=2.13.0"],
        "full": ["statsmodels>=0.14.0", "scipy>=1.10.0", "tensorflow>=2.13.0", "keras>=2.13.0"]
    },

    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],

    python_requires=">=3.8",

    keywords=["machine-learning", "time-series", "forecasting", "streamlit", "restaurant", "demand-planning"],

    license="MIT",
    zip_safe=False,
)
