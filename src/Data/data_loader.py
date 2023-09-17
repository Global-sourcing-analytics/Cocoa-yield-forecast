import pandas as pd
import argparse
import requests
import sys
import re
import os

from BC_solutions import Databases
from datetime import datetime

from pathlib import Path

def find_upwards(cwd: Path, filename: str) -> "Path | None":
    """
    Looks for specific filename upwards in hierarchy starting from specified path. The function
    resolves relative import problem which is described under the following link:
    https://stackoverflow.com/questions/14132789/relative-imports-for-the-billionth-time
    
    Parameters
    ----------
    cwd : Path
        Specific path to start searching from.
    filename : str
        Filename which needs to be found.
    
    Returns
    -------
    Path
        A full path to the searched file.
    """

    if cwd == Path(cwd.root) or cwd == cwd.parent:
        return None
    
    fullpath = cwd / filename
    
    return fullpath if fullpath.exists() else find_upwards(cwd.parent, filename)

# Find path to "BC_solutions" folder
path = find_upwards(Path(os.path.dirname(os.path.realpath(__file__))), "BC_solutions")

# Assign the path to system paths temporarly in order to make it possible to import BC_solutions package
sys.path.append(os.path.dirname(path))

from BC_solutions import Databases

os.chdir(Path(os.path.dirname(os.path.realpath(__file__))))

def get_credentials():
    """
    Sets flags so that any keys or additional information can be passed as arguments by the script execution command.
    
    Returns
    -------
    Namespace
        The namespace of the flags specified in the parser.
    """

    parser=argparse.ArgumentParser(
        prog='api_loader_palette_.py',
        description='''Script responsible for ingesting the data sourced from  https://www.hpe.de/preisindex.html.
        In the final form data lies in the following structure: |
        Server: CorpPolDB |
        Database: DW |
        Schema: data |
        Table: tbl_Prices |
        ''',
        epilog="""Good luck and have fun.""")
    
    parser.add_argument('-s', '--server', type=str, required=True, help='Server name.')
    parser.add_argument('-d', '--database', type=str, required=True, help='Database name.')
    parser.add_argument('-uid', '--username_id', type=str, required=False, help='Server username id.')
    parser.add_argument('-pwd', '--password', type=str, required=False, help='Server password.')

    return parser.parse_args()

def convert_gps_to_lat_long(gps):
    """Function responsible for converting specific type of GPS coordinates to the latitude and longitude format.
    REMEMBER: In order for the function to work properly, the input string must not consist of any whitespaces.

    Parameters
    ----------
    gps : str
        GPS to convert.

    Returns
    -------
    float, float
        Latitude and longitude extracted from GPS.
    """
    
    pattern = r'([NS])(\d+)°([\d.]+)\'([WE])(\d+)°([\d.]+)\''
    match = re.match(pattern, gps)

    lat_dir = match.group(1)
    lat_deg = int(match.group(2))
    lat_min = float(match.group(3))
    lon_dir = match.group(4)
    lon_deg = int(match.group(5))
    lon_min = float(match.group(6))

    latitude = lat_deg + lat_min / 60.0
    longitude = lon_deg + lon_min / 60.0

    if lat_dir == 'S':
        latitude = -latitude
    if lon_dir == 'W':
        longitude = -longitude

    return latitude, longitude

def extract_agronomic_data(dbmanager):
    """Load data from the database assigned to the manager passed through argument.

    Parameters
    ----------
    dbmanager : DatabaseManager
        The manager of connection between script and database.

    Returns
    -------
    DataFrame
        Tabular object prepared for further save.
    """

    df = dbmanager.get_data('SELECT * FROM [DW].[pod].[v_POD]')
    df.to_csv('POD_raw.csv', index=False)
    return df

def extract_geo_data(path):
    """Load and pre-process the geographic data from specified path.

    Parameters
    ----------
    path : String
        Path to the file (.xlsx) with the specific data.

    Returns
    -------
    DataFrame
        Tabular object prepared for further save.
    """

    df = pd.read_excel(path)
    df.loc[df["Latitude"].isnull(), "Complete GPS"] = df[df["Latitude"].isnull()]["Complete GPS"].str.replace(" ", "")
    for index, _ in pd.DataFrame(df.loc[df["Latitude"].isnull(), "Latitude"]).iterrows():
        df.iloc[index, 2] = convert_gps_to_lat_long(df.iloc[index, 1])[0]
        df.iloc[index, 3] = convert_gps_to_lat_long(df.iloc[index, 1])[1]
    df.drop(columns=["Complete GPS"], inplace=True)
    return df

def extract_weather_data(df_geo):
    """Load and pre-process the weather data from specified path.

    Parameters
    ----------
    df_geo : DataFrame
        The tabular data with the farms GPS coordinates. Crucial to extract weather
        from each specific place.

    Returns
    -------
    DataFrame
        Tabular object prepared for further save.
    """

    df_tbl_weather_daily = pd.DataFrame(columns=["Date", "FarmID", "temperature_2m_max", "temperature_2m_min", "apparent_temperature_max", "apparent_temperature_min", 
                                              "precipitation_sum", "rain_sum", "precipitation_hours", "windspeed_10m_max", "windgusts_10m_max", "shortwave_radiation_sum",
                                              "et0_fao_evapotranspiration"])
    df_tbl_weather_hourly = pd.DataFrame(columns=["Date", "FarmID", "soil_temperature_0_to_7cm", "soil_temperature_7_to_28cm", "soil_temperature_28_to_100cm", "soil_temperature_100_to_255cm", 
                                                "soil_moisture_0_to_7cm", "soil_moisture_7_to_28cm", "soil_moisture_28_to_100cm", "soil_moisture_100_to_255cm"])
    
    for farmid in df_geo[df_geo['Latitude'].isna() == False]["FarmID"]:
        latitude, longitude = df_geo[df_geo["FarmID"] == farmid]["Latitude"].values[0], df_geo[df_geo["FarmID"] == farmid]["Longitude"].values[0]
        url = f"https://archive-api.open-meteo.com/v1/archive?latitude={latitude}&longitude={longitude}&start_date=1998-01-01&end_date={datetime.today().strftime('%Y-%m-%d')}&daily=temperature_2m_max,temperature_2m_min,apparent_temperature_max,apparent_temperature_min,precipitation_sum,rain_sum,precipitation_hours,windspeed_10m_max,windgusts_10m_max,shortwave_radiation_sum,et0_fao_evapotranspiration&hourly=soil_temperature_0_to_7cm,soil_temperature_7_to_28cm,soil_temperature_28_to_100cm,soil_temperature_100_to_255cm,soil_moisture_0_to_7cm,soil_moisture_7_to_28cm,soil_moisture_28_to_100cm,soil_moisture_100_to_255cm&timezone=GMT"
        data = requests.get(url, verify=False)
        json_data= data.json()
        df_daily, df_hourly = pd.DataFrame.from_records(json_data['daily']), pd.DataFrame.from_records(json_data['hourly'])
        df_daily["FarmID"], df_hourly["FarmID"] = farmid, farmid
        df_daily.rename(columns={'time':'Date'}, inplace=True)
        df_hourly.rename(columns={'time':'Date'}, inplace=True)
        df_daily.dropna(inplace=True)
        df_hourly.dropna(inplace=True)
        df_tbl_weather_daily = pd.concat([df_tbl_weather_daily, df_daily], ignore_index=True)
        df_tbl_weather_hourly = pd.concat([df_tbl_weather_hourly, df_hourly], ignore_index=True)
    
    return df_tbl_weather_daily, df_tbl_weather_hourly

if __name__ == '__main__':
    credentials = get_credentials()

    # Database credentials
    server = credentials.server
    database = credentials.database
    UID = credentials.username_id
    PWD = credentials.password

    database_manager = Databases.DatabaseManager(server, database, UID, PWD)
    path = "FarmGPS.xlsx"
    
    df_tbl_pods = extract_agronomic_data(database_manager)
    df_tbl_farms_location = extract_geo_data(path)
    df_tbl_weather_daily, df_tbl_weather_hourly = extract_weather_data(df_tbl_farms_location)

    df_tbl_pods.to_csv("tbl_pods_raw.csv", index=False)
    df_tbl_farms_location.to_csv("tbl_farms_location.csv", index=False)
    df_tbl_weather_daily.to_csv("tbl_weather_daily.csv", index=False)
    df_tbl_weather_hourly.to_csv("tbl_weather_hourly.csv", index=False)