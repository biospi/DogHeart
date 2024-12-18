import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from matplotlib.dates import DateFormatter, HourLocator
from plotly.subplots import make_subplots
from openpyxl import load_workbook
import numpy as np
import matplotlib.pyplot as plt


def minimum_spanning_tree(X, copy_X=True):
    """X are edge weights of fully connected graph"""
    if copy_X:
        X = X.copy()

    if X.shape[0] != X.shape[1]:
        raise ValueError("X needs to be square matrix of edge weights")
    n_vertices = X.shape[0]
    spanning_edges = []

    # initialize with node 0:
    visited_vertices = [0]
    num_visited = 1
    # exclude self connections:
    diag_indices = np.arange(n_vertices)
    X[diag_indices, diag_indices] = np.inf

    while num_visited != n_vertices:
        new_edge = np.argmin(X[visited_vertices], axis=None)
        # 2d encoding of new_edge from flat, get correct indices
        new_edge = divmod(new_edge, n_vertices)
        new_edge = [visited_vertices[new_edge[0]], new_edge[1]]
        # add edge to tree
        spanning_edges.append(new_edge)
        visited_vertices.append(new_edge[1])
        # remove all edges inside current tree
        X[visited_vertices, new_edge[1]] = np.inf
        X[new_edge[1], visited_vertices] = np.inf
        num_visited += 1
    return np.vstack(spanning_edges)


def polar_to_cartesian(arr, r):
    a = np.concatenate((np.array([2 * np.pi]), arr))
    si = np.sin(a)
    si[0] = 1
    si = np.cumprod(si)
    co = np.cos(a)
    co = np.roll(co, -1)
    return si * co * r


def get_stage(meta_datafile, df, p_id):
    df = df.copy()
    df["Home Activity"] = "nan"
    df["Clinic Activity"] = "nan"
    df["Period"] = "nan"
    df["Note"] = "nan"
    df["Time_dt"] = pd.to_datetime(df["Time_dt"], format="%H:%M:%S").dt.time

    df_meta = pd.read_csv(meta_datafile)
    df_meta["timestamp"] = pd.to_datetime(df_meta["Date"])
    reference_timestamp = df["Timestamp"].values[0]
    start_window = reference_timestamp - pd.Timedelta(days=7)
    end_window = reference_timestamp + pd.Timedelta(days=7)
    df_filtered = df_meta[
        (df_meta["timestamp"] >= start_window)
        & (df_meta["timestamp"] <= end_window)
        & (df_meta["Participant No."] == p_id)
    ]
    df_filtered_day = df_meta[
        (df_meta["timestamp"] == reference_timestamp)
        & (df_meta["Participant No."] == p_id)
    ]
    stage = df_filtered["Stage"].values[0]
    breed = df_filtered["Breed"].values[0]

    for i, row in df_filtered_day.iterrows():
        start_time = pd.to_datetime(row["Start Time"], format="%H:%M:%S").time()
        if not pd.isna(row["End Time"]):
            end_time = pd.to_datetime(row["End Time"], format="%H:%M:%S").time()
        else:
            end_time = start_time

        df.loc[
            (df["Time_dt"] >= start_time) & (df["Time_dt"] <= end_time), "Home Activity"
        ] = row["Home Activity"]
        df.loc[
            (df["Time_dt"] >= start_time) & (df["Time_dt"] <= end_time),
            "Clinic Activity",
        ] = row["Clinic Activity"]
        df.loc[
            (df["Time_dt"] >= start_time) & (df["Time_dt"] <= end_time), "Period"
        ] = (str(row["Start Time"]) + " " + str(row["End Time"]))
        df.loc[(df["Time_dt"] >= start_time) & (df["Time_dt"] <= end_time), "Note"] = (
            row["Note"]
        )

    clinic = (
        df_filtered_day["Clinic Activity"].values[0]
        if len(df_filtered_day["End Time"].values) > 0
        else None
    )
    home = (
        " ".join(df_filtered_day["Home Activity"].astype(str))
        + "-"
        + " ".join(df_filtered_day["Start Time"].astype(str))
        + "-"
        + " ".join(df_filtered_day["End Time"].astype(str))
    )
    start_time = (
        df_filtered_day["Start Time"].values[0]
        if len(df_filtered_day["Start Time"]) > 0
        else None
    )
    end_time = (
        df_filtered_day["End Time"].values[0]
        if len(df_filtered_day["End Time"].values) > 0
        else None
    )
    note = " ".join(df_filtered_day["Note"].astype(str))
    return df, stage, breed, clinic, home, start_time, end_time, note


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
    trace = go.Heatmap(z=X, x=timestamps, y=animal_ids, colorscale="Viridis")
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
    dpi=300,
):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    cax = ax.imshow(X, aspect="auto", cmap="viridis", interpolation="nearest")

    # Set title and labels
    ax.set_title(title)
    ax.set_xlabel(xaxis)
    ax.set_ylabel(yaxis)

    # Format x-axis to display ticks for each hour
    ax.xaxis.set_major_locator(HourLocator(interval=1))
    ax.xaxis.set_major_formatter(DateFormatter("%H:%M"))

    # Set x and y ticks and labels
    ax.set_xticks(timestamps)
    ax.set_yticks(animal_ids)
    ax.set_xticklabels(timestamps, rotation=90)
    # ax.set_yticklabels(animal_ids)

    # Add colorbar
    fig.colorbar(cax)

    # Ensure the output directory exists
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save the plot
    file_path = out_dir / filename.replace("=", "_").lower()
    plt.savefig(file_path, bbox_inches="tight")
    plt.close(fig)

    print(file_path)
    return cax, title


def resample(df):
    # Create a complete range of timestamps for the entire day at second intervals
    start_time = df["Timestamp"].iloc[0].replace(hour=0, minute=0, second=0)
    end_time = df["Timestamp"].iloc[-1].replace(hour=23, minute=59, second=59)
    full_range = pd.date_range(start=start_time, end=end_time, freq="s")

    # Reindex the dataframe to the full range of timestamps
    df_full = pd.DataFrame(index=full_range)
    df_full["Timestamp"] = df_full.index
    df_full = df_full.reset_index(drop=True)

    # Merge the original dataframe with the full range dataframe
    df_full = df_full.merge(
        df[
            [
                "AccX",
                "AccY",
                "AccZ",
                "MagX",
                "MagY",
                "MagZ",
                "Pulse",
                "SpO2",
                "Timestamp",
                "Counts",
            ]
        ],
        on="Timestamp",
        how="left",
    )

    # Extract the time from the Timestamp for the Time_dt column
    df_full["Time_dt"] = df_full["Timestamp"].dt.time
    # df_full = df_full.head(86400) #86400 in a day
    # df_full['Magnitude'] = np.sqrt(df_full['AccX'] ** 2 + df_full['AccY'] ** 2 + df_full['AccZ'] ** 2)
    # df_full.index = df_full['Timestamp']
    # df_full['Counts'] = df_full['Magnitude'].rolling('60s').apply(count_above_threshold)

    grouped = df_full.groupby(df_full["Timestamp"].dt.date)
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
    v = np.maximum((arr / alpha) + (3.0 / 8.0) + sigma_sq / (alpha**2), 0)
    f = 2.0 * np.sqrt(v)
    return f


def format_metafile(input_file):
    sheet_name = "Timestamps"  # Replace with your sheet name
    df = pd.read_excel(input_file, sheet_name=sheet_name, skiprows=[0, 1])

    df1 = df.iloc[:, 1:8]
    df2 = df.iloc[:, 11:]
    df2.columns = df1.columns
    # Iterate through each row
    for df_ in [df1, df2]:
        df_["Participant No."] = df_["Participant No."].ffill()
        df_["Date"] = df_["Date"].ffill()
    print(df1)

    df_concat = pd.concat([df1, df2], axis=0)
    df_concat = df_concat.sort_values(["Participant No.", "Date"])

    df_cleaned = df_concat.dropna(
        subset=["Clinic Activity", "Home Activity", "Start Time", "End Time", "Note"],
        how="all",
    )
    df_cleaned.to_csv("metadata.csv", index=False)

    df_stages = pd.read_csv("meta.csv")

    df_merged = df_cleaned.merge(
        df_stages[["Participant No.", "Stage", "Breed"]],
        on="Participant No.",
        how="left",
    )
    df_merged = df_merged.sort_values(["Participant No.", "Date"])
    df_merged.to_csv("meta_data.csv", index=False)


def get_2d_histogram(list1, list2, plot=True):
    # Compute the 2D histogram
    counts, x_edges, y_edges = np.histogram2d(list1, list2, bins=10)

    if plot:
        plt.imshow(counts.T, origin='lower', extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], aspect='auto',
                   cmap='Blues')
        plt.colorbar(label='Counts')
        plt.xlabel('Sensor 1 Measurements')
        plt.ylabel('Sensor 2 Measurements')
        plt.title('2D Histogram of Sensor Data')
        plt.show()

    # Combine all data into a single matrix
    histogram_matrix = np.zeros((counts.shape[0] + 2, counts.shape[1] + 2))
    histogram_matrix[0, 1:-1] = x_edges[:-1]  # X-axis bin starts
    histogram_matrix[1:-1, 0] = y_edges[:-1]  # Y-axis bin starts
    histogram_matrix[1:-1, 1:-1] = counts  # Bin counts
    print("Histogram Matrix:")
    print(histogram_matrix)
    return histogram_matrix


if __name__ == "__main__":
    #format_metafile(Path("C:\Brooke Study Data\Stages and timestamps .xlsx"))


    # Example data
    list1 = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    list2 = [9, 8, 7, 6, 5, 4, 3, 2, 1]

    get_2d_histogram(list1, list2, plot=True)