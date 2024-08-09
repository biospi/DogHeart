from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from matplotlib.dates import DateFormatter, HourLocator
from plotly.subplots import make_subplots


def get_stage(meta_datafile, df, p_id):
    df_meta = pd.read_csv(meta_datafile)
    df_meta["timestamp"] = pd.to_datetime(df_meta["Visit Date"], format="%d/%m/%Y")
    pd.to_datetime(df_meta["Visit Date"], format="%d/%m/%Y")
    reference_timestamp = df['Timestamp'].values[0]
    start_window = reference_timestamp - pd.Timedelta(days=7)
    end_window = reference_timestamp + pd.Timedelta(days=7)
    df_filtered = df_meta[
        (df_meta['timestamp'] >= start_window) &
        (df_meta['timestamp'] <= end_window) &
        (df_meta['Participant No.'] == p_id)
        ]
    stage = df_filtered["Stage"].values[0]
    breed = df_filtered["Breed"].values[0]
    return stage, breed


# def visu(input_file):
#     print(input_file)
#     df = pd.read_csv(input_file)
#     print(df)
#
#     # fig = px.bar(df, x='Timestamp', y="Counter", title='Counter Data Over Time')
#     # fig.update_xaxes(title_text='Time')
#     # fig.update_yaxes(title_text='Counter')
#     # fig.show()
#
#     fig = px.line(df, x='Timestamp', y=['AccX', 'AccY', 'AccZ'], title='Acceleration Data Over Time')
#     fig.update_xaxes(title_text='Time')
#     fig.update_yaxes(title_text='Acceleration (m/s^2)')
#     width_in_pixels = 1280
#     height_in_pixels = 720
#     filepath = Path(__file__).parent / "acceleration_data.png"
#     print(filepath)
#     fig.write_image(filepath, width=width_in_pixels, height=height_in_pixels, scale=2)
#     fig.show()
#     print("done")
#
#     fig = px.line(df, x='Timestamp', y=['MagX', 'MagY', 'MagZ'], title='Magnetic Field Data Over Time')
#     fig.update_xaxes(title_text='Time')
#     fig.update_yaxes(title_text='Magnetic Field (uT)')
#     fig.write_image("magnetic_field_data.png")
#     fig.show()
#
#     fig = px.line(df, x='Timestamp', y=['GyrX', 'GyrY', 'GyrZ'], title='Gyroscope Data Over Time')
#     fig.update_xaxes(title_text='Time')
#     fig.update_yaxes(title_text='Angular Velocity (rad/s)')
#     fig.show()
#
#     fig = px.line(df, x='Timestamp', y='SpO2', title='SpO2 Over Time')
#     fig.update_xaxes(title_text='Time')
#     fig.update_yaxes(title_text='SpO2 (%)')
#     fig.write_image("gyroscope_data.png")
#     fig.write_image("spo2_over_time.png")
#     fig.show()
#
#     fig = px.line(df, x='Timestamp', y='Pulse', title='Pulse Over Time')
#     fig.update_xaxes(title_text='Time')
#     fig.update_yaxes(title_text='Pulse (BPM)')
#     fig.write_image("spo2_and_pulse_over_time.png")
#     fig.write_image("pulse_over_time.png")
#     fig.show()
#
#     fig = go.Figure()
#
#     fig.add_trace(go.Scatter(x=df['Timestamp'], y=df['SpO2'], mode='lines', name='SpO2'))
#     fig.add_trace(go.Scatter(x=df['Timestamp'], y=df['Pulse'], mode='lines', name='Pulse'))
#
#     fig.update_layout(title='SpO2 and Pulse Over Time',
#                       xaxis_title='Time',
#                       yaxis_title='Value',
#                       legend_title='Metrics')
#
#     fig.show()
#
#     # fig = px.line(df, x='Timestamp', y='DeviceTemperature', title='Device Temperature Over Time')
#     # fig.update_xaxes(title_text='Time')
#     # fig.update_yaxes(title_text='Temperature (°C)')
#     # fig.show()


def plot_heatmap_plotly(
    X,
    timestamps,
    animal_ids,
    out_dir,
    title="Heatmap",
    filename="heatmap.html",
    yaxis="Data",
    xaxis="Time",
):
    fig = make_subplots(rows=1, cols=1)
    trace = go.Heatmap(
        z=X,
        x=timestamps,
        y=animal_ids,
        colorscale="Viridis"
    )
    fig.add_trace(trace, row=1, col=1)
    fig.update_layout(title_text=title)
    fig.update_layout(xaxis_title=xaxis)
    fig.update_layout(yaxis_title=yaxis)
    # fig.show()
    # create_rec_dir(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    file_path = out_dir / filename.replace("=", "_").lower()
    print(file_path)
    fig.write_html(str(file_path))
    return trace, title


def plot_heatmap(
        X,
        timestamps,
        animal_ids,
        out_dir,
        title="Heatmap",
        filename="heatmap.png",
        yaxis="Data",
        xaxis="Time",
        figsize=(20, 20),
        dpi=300
):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    cax = ax.imshow(X, aspect='auto', cmap='viridis', interpolation='nearest')

    # Set title and labels
    ax.set_title(title)
    ax.set_xlabel(xaxis)
    ax.set_ylabel(yaxis)

    # Format x-axis to display ticks for each hour
    ax.xaxis.set_major_locator(HourLocator(interval=1))
    ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))

    # Set x and y ticks and labels
    ax.set_xticks(timestamps)
    ax.set_yticks(animal_ids)
    ax.set_xticklabels(timestamps, rotation=90)
    #ax.set_yticklabels(animal_ids)

    # Add colorbar
    fig.colorbar(cax)

    # Ensure the output directory exists
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save the plot
    file_path = out_dir / filename.replace("=", "_").lower()
    plt.savefig(file_path, bbox_inches='tight')
    plt.close(fig)

    print(file_path)
    return cax, title


def resample(df):
    # Create a complete range of timestamps for the entire day at second intervals
    start_time = df['Timestamp'].iloc[0].replace(hour=0, minute=0, second=0)
    end_time = df['Timestamp'].iloc[-1].replace(hour=23, minute=59, second=59)
    full_range = pd.date_range(start=start_time, end=end_time, freq='s')

    # Reindex the dataframe to the full range of timestamps
    df_full = pd.DataFrame(index=full_range)
    df_full['Timestamp'] = df_full.index
    df_full = df_full.reset_index(drop=True)

    # Merge the original dataframe with the full range dataframe
    df_full = df_full.merge(df[['AccX', 'AccY', 'AccZ', 'MagX', 'MagY', 'MagZ', 'Pulse', 'SpO2', 'Timestamp', 'Counts']], on='Timestamp', how='left')

    # Extract the time from the Timestamp for the Time_dt column
    df_full['Time_dt'] = df_full['Timestamp'].dt.time
    #df_full = df_full.head(86400) #86400 in a day
    # df_full['Magnitude'] = np.sqrt(df_full['AccX'] ** 2 + df_full['AccY'] ** 2 + df_full['AccZ'] ** 2)
    # df_full.index = df_full['Timestamp']
    # df_full['Counts'] = df_full['Magnitude'].rolling('60s').apply(count_above_threshold)

    grouped = df_full.groupby(df_full['Timestamp'].dt.date)
    dfs = [group for _, group in grouped]

    return df_full, dfs


def count_above_threshold(window):
    threshold = 10
    return np.nansum(window > threshold)


def anscombe(arr, sigma_sq=0, alpha=1):
    """
    Generalized Anscombe variance-stabilizing transformation
    References:
    [1] http://www.cs.tut.fi/~foi/invansc/
    [2] M. Makitalo and A. Foi, "Optimal inversion of the generalized
    Anscombe transformation for Poisson-Gaussian noise", IEEE Trans.
    Image Process, 2012
    [3] J.L. Starck, F. Murtagh, and A. Bijaoui, Image  Processing
    and Data Analysis, Cambridge University Press, Cambridge, 1998)
    :param arr: variance-stabilized signal
    :param sigma_sq: variance of the Gaussian noise component
    :param alpha: scaling factor of the Poisson noise component
    :return: variance-stabilized array
    """
    v = np.maximum((arr / alpha) + (3. / 8.) + sigma_sq / (alpha ** 2), 0)
    f = 2. * np.sqrt(v)
    return f