# Hong Kong Weather Art Visualization Project

This project fetches and visualizes the latest 10 days of Hong Kong weather data, generating a stylized weather poster image that shows trends in temperature, humidity, precipitation, and wind speed.

## Features

- **Data Fetching**: Retrieve Hong Kong weather data via API, including temperature, humidity, precipitation, and wind speed.
- **Data Processing**: Clean and smooth the data using pandas (rolling average).
- **Artistic Visualization**: Generate a dark-themed weather poster featuring:
  - Temperature gradient line (with color bar)
  - Humidity ribbon
  - Precipitation bar chart
  - Neon wind speed line
- **Auto Save**: Output image to `art/hk_weather_poster.png`.

## Visualization Details

- **Temperature**: Top section with a gradient line and color bar, filled area highlights the temperature range.
- **Humidity**: Middle ribbon showing relative humidity changes.
- **Precipitation**: Bottom bar chart for hourly precipitation.
- **Wind Speed**: Neon line overlay in the bottom section for wind speed.
- **Time Axis**: X-axis shows date and hour, Y-axis shows corresponding weather variables.
- **Font Adaptation**: Automatically switches to Chinese fonts for bilingual display.

## Usage

1. Install dependencies (recommended to use a virtual environment):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Generate data (run the data fetching script to create `data/hk_weather.csv`).
3. Run the visualization script:
   ```bash
   python src/visualize_weather.py
   ```
   Output image path: `art/hk_weather_poster.png`

## Dependencies

- pandas
- numpy
- matplotlib
- seaborn

## Data Source

- OpenWeatherMap or Open-Meteo API for real-time and historical weather data.

## Extensibility

- Supports custom time ranges, variable selection, and style adjustments.
- Can be adapted for other cities or weather datasets for artistic visualization.
