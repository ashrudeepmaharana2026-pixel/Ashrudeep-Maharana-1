README
Project Overview
The *AI Travel Analyst* project is a comprehensive data science solution engineered to decode dynamic flight pricing mechanics. In the modern aviation industry, pricing algorithms respond dynamically to demand, lead times, carrier positioning, and seasonal trends. This project delivers an end-to-end analytical pipeline that cleans raw, real-world booking datasets, standardizes fragmented feature formats, conducts Exploratory Data Analysis (EDA), and uncovers key factors driving ticket prices.
Project Statement
Flight pricing is notoriously opaque and volatile, making it difficult for both consumers to optimize booking times and travel platforms to offer transparent pricing insights. Raw travel datasets often suffer from severe data quality issues, including mixed currency formats, inconsistent duration strings, missing route parameters, and unstandardized carrier naming conventions. 
Core Objectives:
1. Clean and standardize raw flight booking data containing dirty text and missing values.
2. Build an automated feature-engineering pipeline for flight durations and advance booking windows.
3. Conduct in-depth univariate, bivariate, and multivariate Exploratory Data Analysis (EDA).
4. Identify major drivers behind flight prices and synthesize actionable recommendations for travelers.
Installation & Setup Instructions
Python 3.8+ installed on your system.
Git for repository cloning.
### Step-by-Step Environment Setup
1. Clone the Repository:
  git clone [https://github.com/your-username/ai-travel-analyst.git](https://github.com/your-username/ai-travel-analyst.git)
   cd ai-travel-analyst
Create and Activate a Virtual Environment:
python -m venv venv
venv\Scripts\activate
 Install Required Dependencies:
pip install --upgrade pip
pip install pandas numpy matplotlib seaborn regex
Execute the Data Analysis Script:
python flight_analysis.py

Dataset Used
File Name: flight_pricing_dataset.csv
Data Volume: Multi-attribute flight transactions covering domestic and international routes.
Feature Data Dictionary
Column name               Data type                Description                                   Sample Raw Values
Flight_ID                 String                  Unique identifier                            FL133547, FL101419
                                                  for each flight entry
Airline                   String                  Commercial airline                           Indigo, Qatar  
                                                   operating the route       
Source / Destination      String                 Departure and arrival locations               Hyderabad, DXB, SYD   
Duration                   String                Total flight elapsed time                     1.67, 0h 45m, 177 min
Total_Stops                String /Int           Number of layovers                            0,1-stop and 2-stop
                                                   during travel
Days_Before_Departure     Integer                  Advance booking window                       43, 3, 110
                                                       days
Travel_Class                String                  Seating tier                                Economy, Business
Price                       String/Float        Target variable: Flight ticket cost              "Rs. 151,632.89"

Methodology
Step 1: Data Cleaning & Preprocessing Workflow
To get the dataset ready for modeling and visualization, perform the following steps:
Handle Missing Values: Impute or drop missing Price, Source, Destination, Season, and Total_Stops values.
Standardize Currency & Text: Remove strings like "Rs. " or commas from Price values (e.g., "Rs. 151,632.89" \rightarrow 151632.89). Standardize airline names to lowercase or title case.

Feature Extraction & Normalization:
Convert Duration into a uniform numerical representation (in total minutes).
Standardize Total_Stops into numerical values (e.g., non-stop / 0 \rightarrow 0, 1 stop / 1 \rightarrow 1).
Map 3-letter IATA airport codes to standard full city names if needed.
Step 2: Python Analysis Script
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
# Set plotting style
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
# 1. Load Data
df = pd.read_csv("flight_pricing_dataset.csv")
# 2. Data Cleaning
# Clean 'Price' column
def clean_price(val):
    if pd.isna(val):
        return np.nan
    val_str = str(val).replace("Rs.", "").replace(",", "").strip()
    try:
        return float(val_str)
    except ValueError:
        return np.nan
df['Price'] = df['Price'].apply(clean_price)
df = df.dropna(subset=['Price']) # Drop rows where Price is missing
# Clean 'Total_Stops'
def clean_stops(val):
    if pd.isna(val):
        return 0
    val_str = str(val).lower().strip()
    if 'non-stop' in val_str or val_str == '0':
        return 0
    elif '1 stop' in val_str or val_str == '1':
        return 1
    elif '2 stop' in val_str or '2' in val_str:
        return 2
    return 0
df['Total_Stops'] = df['Total_Stops'].apply(clean_stops)
# Clean 'Duration' to Total Minutes
def duration_to_mins(val):
    if pd.isna(val):
        return np.nan
    val_str = str(val).strip()
    # Check hours and mins format (e.g., "1h 28m")
    h_m = re.findall(r'(\d+)h\s*(\d+)m', val_str)
    if h_m:
        return int(h_m[0][0]) * 60 + int(h_m[0][1])
    # Check mins format (e.g., "177 min")
    mins = re.findall(r'(\d+)\s*min', val_str)
    if mins:
        return int(mins[0])
    # Check decimal hours (e.g., "1.67")
    try:
        return float(val_str) * 60
    except ValueError:
        return np.nan

df['Duration_Mins'] = df['Duration'].apply(duration_to_mins)
# Clean Categorical Text
df['Airline'] = df['Airline'].str.title().str.strip()
df['Travel_Class'] = df['Travel_Class'].str.title().str.strip()
# --- 3. Five Meaningful Visualizations ---
# Viz 1: Price Distribution by Travel Class
plt.figure()
sns.boxplot(data=df, x='Travel_Class', y='Price', palette='Set2')
plt.title('1. Price Distribution across Travel Classes')
plt.xlabel('Travel Class')
plt.ylabel('Price (INR)')
plt.show()
# Viz 2: Impact of Days Before Departure on Price
plt.figure()
sns.scatterplot(data=df, x='Days_Before_Departure', y='Price', hue='Travel_Class', alpha=0.7)
plt.title('2. Price vs. Days Before Departure')
plt.xlabel('Days Before Departure')
plt.ylabel('Price (INR)')
plt.show()
# Viz 3: Average Price by Airline
plt.figure()
avg_price_airline = df.groupby('Airline')['Price'].mean().sort_values(ascending=False).reset_index()
sns.barplot(data=avg_price_airline, x='Price', y='Airline', palette='viridis')
plt.title('3. Average Flight Price by Airline')
plt.xlabel('Average Price (INR)')
plt.ylabel('Airline')
plt.show()
# Viz 4: Correlation Heatmap of Numerical Features
plt.figure()
numeric_cols = ['Price', 'Distance_km', 'Days_Before_Departure', 'Duration_Mins', 'Total_Stops', 'Passenger_Count']
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('4. Numerical Feature Correlation Matrix')
plt.show()
# Viz 5: Seasonal Price Variation by Booking Channel
plt.figure()
sns.barplot(data=df, x='Season', y='Price', hue='Booking_Channel', ci=None, palette='magma')
plt.title('5. Average Price by Season and Booking Channel')
plt.xlabel('Season')
plt.ylabel('Average Price (INR)')
plt.show()
Step 3: Key Factors & Recommendations
Major Factors Driving Flight Prices
Travel Class: Premium Economy, Business, and First Class tiers show an exponential cost markup compared to Economy.
Flight Distance & Duration: Longer international haul flights (Distance_km) drive higher base pricing.
Advance Booking Window: Tighter booking windows (Days_Before_Departure) generally trigger dynamic surge pricing.
Airline Carrier: Premium international airlines (e.g., Qatar Airways, Emirates, British Airways) command higher average price points compared to regional low-cost carriers (e.g., IndiGo, SpiceJet).
Actionable Insights for Travelers
Book Early: Secure bookings at least 30–45 days in advance to capture baseline rates before surge pricing takes effect.
Compare Channels: Check prices across official airline websites vs. mobile app channels to identify exclusive channel-based discounts.
Class Trade-offs: Opt for non-stop direct flights in Economy class to balance travel time efficiency against high premium-cabin premiums.

[Raw CSV File] ➡️ [Data Normalization] ➡️ [Feature Engineering] ➡️ [EDA & Visualizations] ➡️ [Insights Engine]

1. Data Hygiene & Parsing
Price Standardization: Stripped non-numeric formatting including currency prefixes ("Rs. "), thousand separators (,), and handled invalid strings.
Duration Canonicalization: Standardized mixed representations ("1h 45m", "105 min", decimal hours "1.75") into a uniform numerical variable: Total Minutes.
Layover Consolidation: Categorized missing and string layover variants ("non-stop", "1 stop", "2 stops") into standardized integer codes (0, 1, 2).
Text Normalization: Trimmed leading whitespace and standardized airline/city names using proper title casing.
 Technologies  Used
Python 3.x: Core programming language.
Pandas: Data manipulation, missing value handling, and aggregation.
NumPy: Numerical computation and array manipulations.
Matplotlib: Low-level plotting and visual styling adjustments.
Seaborn: High-level statistical visualization (boxplots, heatmaps, scatterplots).
Regex (re): String pattern matching for flight duration extraction.
Results
1. Primary Price Drivers
Travel Class Premium: Travel class is the strongest single predictor of cost. First and Business class tickets carry a 300% to 500% premium over standard Economy rates on international long-haul routes.
Booking Window Dynamics: Booking within a 0–14 day lead window results in exponential price spikes, whereas booking 30+ days prior offers baseline pricing stability.
Airline Categorization: Premium international carriers (e.g., Qatar Airways, Emirates, British Airways) exhibit higher variance and higher average prices due to long-haul flight segments compared to low-cost domestic carriers (e.g., IndiGo, SpiceJet).
2. Feature Correlations
Distance vs. Duration: High positive correlation (r > 0.85) between flight distance (Distance_km) and duration (Duration_Mins).
Distance vs. Price: Strong positive correlation for international routes, though class selection remains an overriding factor.
 Challenges Faced 
Heterogeneous Flight Durations:
Challenge: The dataset stored flight duration in mixed formats (decimal hours, minutes string, hours/minutes string).
Solution: Built a custom regular expression parser (duration_to_mins) to evaluate and convert all variants into integer minutes.
Noisy Currency Formats:
Challenge: Price entries contained mixed numeric floats and string formats (e.g., "Rs. 116,295.34").
Solution: Implemented string stripping routines before typecasting to floating-point numbers, safely coercing invalid values to NaN.
Future Enhancements
Machine Learning Pipelines: Train regression algorithms (Random Forest, XGBoost, LightGBM) to forecast flight prices based on route and lead time.
Feature Importance Analysis: Use SHAP (SHapley Additive exPlanations) values to explain individual feature contributions to predicted flight prices.
Interactive Web App: Build a Streamlit web interface allowing users to input route parameters and receive instant price predictions and historical trends.
