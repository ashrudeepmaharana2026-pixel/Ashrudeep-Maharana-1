# AI Travel Analyst: Flight Price Data Exploration & Visualization

## Project Overview
The **AI Travel Analyst** project is a data science solution designed to analyze dynamic flight booking prices. It explores pricing behaviors across various airlines, routes, booking windows, travel classes, and seasonal trends to provide actionable travel insights.

---

## Problem Statement
Flight prices fluctuate rapidly based on factors like seasonality, booking lead times, travel class, and carrier types. Travelers and booking platforms often struggle to identify cost drivers. This project addresses this by preprocessing raw, noisy flight booking datasets and performing Exploratory Data Analysis (EDA) to uncover primary cost drivers.

---

## Installation Instructions

1. **Clone the repository:**
   ```bash
  
#Dataset Title: flight_pricing_dataset.csv

Source Attributes:

Identifiers: Flight_ID, Airline, Aircraft_Type

Route Details: Source, Destination, Distance_km

Timing Features: Departure_Date, Departure_Time, Arrival_Time, Duration, Weekday, Season

Transaction Info: Travel_Class, Total_Stops, Days_Before_Departure, Booking_Channel, Passenger_Count, Price

# Methodology
Data Preprocessing & Standardisation:

Price Cleaning: Stripped currency strings (e.g., "Rs.") and commas to cast prices into continuous numeric floats.

Duration Conversion: Converted mixed string formats (e.g., "1h 45m", "177 min", "1.67") into continuous total minutes.

Layover Formatting: Mapped layover descriptions ("non-stop", "1 stop", "2 stops") into standard integer codes (0, 1, 2).

Categorical Normalization: Cleaned whitespace and normalized airline names to title casing.

Exploratory Data Analysis (EDA):

Generated univariate distributions for ticket prices across travel classes.

Computed multivariate correlation matrices to inspect relationships between distance, lead time, duration, and price.

Analyzed price trends based on booking advance windows (Days_Before_Departure).

🛠️ Technologies Used
Programming Language: Python 3.8+

Data Processing & Manipulation: Pandas, NumPy

Data Visualization: Matplotlib, Seaborn

Pattern Matching: Regular Expressions (re)

# Results
Class Margin: Travel class is the primary price driver. Business and First Class tickets carry a 3x to 5x markup over standard Economy fares.

Lead Time Impact: Flight prices spike significantly when booked within 14 days of departure, whereas booking 30+ days in advance stabilizes baseline fares.

Carrier Variance: Premium international carriers (e.g., Qatar Airways, Emirates) show higher price variance compared to low-cost domestic carriers (e.g., IndiGo, SpiceJet).

# Challenges Faced
Mixed Duration String Formats: The dataset logged durations inconsistently (decimal hours vs. string minutes vs. hours/minutes). This was solved by creating a regex-based parsing function to convert all entries into total minutes.

Inconsistent Price Formats: Several entries contained raw string currency labels (e.g., "Rs. 151,632.89"), which were systematically stripped and coerced to floats.

# Future Improvements
Machine Learning Models: Build predictive regression models (e.g., Random Forest, XGBoost) to forecast ticket prices based on booking parameters.

Interactive Dashboard: Build a Web UI using Streamlit to let users filter route prices and view real-time fare trends.
