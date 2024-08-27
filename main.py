from pathlib import Path

import numpy as np
import pandas as pd

from utils import plot_heatmap_plotly, resample, count_above_threshold, get_stage


def build_samples_day(out_dir, v, df, df_list, sep="__"):
    print(f"Building samples for {v}...")
    df_samples = df.copy()
    df_samples.columns = [f"x{i}" for i in range(len(df_samples.columns))]
    df_samples["id"] = df_samples.index.str.split(sep).str[0].values
    df_samples["label"] = df_samples.index.str.split(sep).str[1].values
    df_samples["breed"] = df_samples.index.str.split(sep).str[2].values
    df_samples["date"] = df_samples.index.str.split(sep).str[3].values
    df_samples["clinic"] = df_samples.index.str.split(sep).str[4].values
    df_samples["home"] = df_samples.index.str.split(sep).str[5].values
    df_samples["start_time"] = df_samples.index.str.split(sep).str[6].values
    df_samples["end_time"] = df_samples.index.str.split(sep).str[7].values
    df_samples["note"] = df_samples.index.str.split(sep).str[8].values
    df_samples = df_samples.reset_index(drop=True)
    build_samples_hour(out_dir, df_samples, v, df_list)
    df_cleaned = df_samples.dropna(subset=[f"x{i}" for i in range(86400)], how="all")
    print(df_cleaned)
    df_cleaned.to_csv(out_dir / f"{v}_dataset_day.csv", index=False)


def build_samples_hour(out_dir, df, v, df_list):
    data = []
    for _, row in df.iterrows():

        meta = df_list[int(row["id"])]
        print(df_list[int(row["id"])])

        hourly_segments = split_features(row, meta)
        for (
                segment_features,
                label,
                participant_id,
                breed,
                day,
                clinic,
                home,
                start_time,
                end_time,
                note,
                hour,
                pos_value_count,
                mean,
                median,
                missingness_percentage,
        ) in hourly_segments:
            if home == '' or pd.isna(home):
                if 'home' in note.lower():
                    home = "Home"
                if 'garden' in note.lower():
                    home = "Garden"
                if 'sleep' in note.lower():
                    home = "Sleep"
                if 'running' in note.lower():
                    home = "Running"
                if 'playing' in note.lower():
                    home = "Playing"
            new_row = list(segment_features) + [
                label,
                participant_id,
                breed,
                day,
                clinic,
                home,
                start_time,
                end_time,
                note,
                hour,
                pos_value_count,
                mean,
                median,
                missingness_percentage,
            ]
            data.append(new_row)
    columns = [f"x{i}" for i in range(3600)] + [
        "label",
        "participant_id",
        "breed",
        "day",
        "clinic",
        "home",
        "start_time",
        "end_time",
        "note",
        "hour",
        "pos_value_count",
        "mean",
        "median",
        "missingness_percentage",
    ]
    df_h = pd.DataFrame(data, columns=columns)
    df_cleaned = df_h.dropna(subset=[f"x{i}" for i in range(3600)], how="all")
    df_cleaned.loc[(df_cleaned["clinic"] == '') & (df_cleaned["home"] == ''), "home"] = "Home"
    print(df_cleaned)
    df_cleaned.to_csv(out_dir / f"{v}_dataset_hour.csv", index=False)


def split_features(row, meta):
    hourly_segments = []
    label = row["label"]
    participant_id = row["id"]
    breed = row["breed"]
    day = row["date"]
    clinic = 'nan'
    home = 'nan'
    start_time = 'nan'
    end_time = 'nan'
    note = 'nan'

    features = row.drop(["label", "id", "breed", "date", "clinic", "home", "start_time", "end_time", "note"]).values
    for hour in range(24):
        df_m_h = meta[meta['Time_dt'].apply(lambda x: x.hour) == hour]

        clinic = np.unique(df_m_h["Clinic Activity"].values.astype(str))
        clinic = "__".join([x for x in clinic if x != 'nan'])
        home = np.unique(df_m_h["Home Activity"].values.astype(str))
        home = "__".join([x for x in home if x != 'nan'])
        note = np.unique(df_m_h["Note"].values.astype(str))
        note = "__".join([x for x in note if x != 'nan'])
        period = np.unique(df_m_h["Period"].values.astype(str))
        if len(period) > 0:
            if period[0] != "nan":
                split = period[0].split(' ')
                start_time = split[0]
                end_time = split[1]

        segment_features = features[hour * 3600 : (hour + 1) * 3600]
        pos_value_count = len(segment_features[segment_features > 0])
        mean = np.nanmean(segment_features)
        median = np.nanmedian(segment_features)
        total_elements = segment_features.size
        nan_count = np.count_nonzero(pd.isna(segment_features))
        missingness_percentage = (nan_count / total_elements) * 100
        hourly_segments.append(
            (
                segment_features,
                label,
                participant_id,
                breed,
                day,
                clinic,
                home,
                start_time,
                end_time,
                note,
                hour,
                pos_value_count,
                mean,
                median,
                missingness_percentage,
            )
        )
        clinic = 'nan'
        home = 'nan'
        start_time = 'nan'
        end_time = 'nan'
        note = 'nan'

    return hourly_segments


def build_datasets(out_dir):
    print("building samples...")
    df_counts = pd.read_csv(out_dir / "counts_dataset_hour.csv")
    df_pulse = pd.read_csv(out_dir / "Pulse_dataset_hour.csv")
    df_accx = pd.read_csv(out_dir / "AccX_dataset_hour.csv")
    df_accy = pd.read_csv(out_dir / "AccY_dataset_hour.csv")
    df_accz = pd.read_csv(out_dir / "AccZ_dataset_hour.csv")
    df_mag = pd.read_csv(out_dir / "Magnitude_dataset_hour.csv")
    df_counts = df_counts.head(len(df_pulse))

    df_p = df_pulse
    df_c = df_counts
    df_x = df_accx
    df_y = df_accy
    df_z = df_accz
    df_m = df_mag

    # mask_miss = df_pulse["missingness_percentage"] < 100
    # df_p = df_pulse[mask_miss]
    # df_c = df_counts[mask_miss]
    # df_x = df_accx[mask_miss]
    # df_y = df_accy[mask_miss]
    # df_z = df_accz[mask_miss]
    # df_m = df_mag[mask_miss]
    #
    # mask_miss = df_c["missingness_percentage"] < 50
    # df_p = df_p[mask_miss]
    # df_c = df_c[mask_miss]
    # df_x = df_x[mask_miss]
    # df_y = df_y[mask_miss]
    # df_z = df_z[mask_miss]
    # df_m = df_m[mask_miss]
    #
    # mask_activity = df_c["mean"] > 3
    # df_p = df_p[mask_activity]
    # df_c = df_c[mask_activity]
    # df_x = df_x[mask_activity]
    # df_y = df_y[mask_activity]
    # df_z = df_z[mask_activity]
    # df_m = df_m[mask_activity]

    df_p.to_csv(out_dir / "cleaned_dataset_pulse.csv", index=False)
    df_c.to_csv(out_dir / "cleaned_dataset_counts.csv", index=False)
    df_x.to_csv(out_dir / "cleaned_dataset_acc_x.csv", index=False)
    df_y.to_csv(out_dir / "cleaned_dataset_acc_y.csv", index=False)
    df_z.to_csv(out_dir / "cleaned_dataset_acc_z.csv", index=False)
    df_m.to_csv(out_dir / "cleaned_dataset_acc_m.csv", index=False)

    meta_columns = [
        "label",
        "participant_id",
        "breed",
        "day",
        "clinic",
        "home",
        "start_time",
        "end_time",
        "note",
        "hour",
        "pos_value_count",
        "mean",
        "median",
        "missingness_percentage"
    ]

    df_p_features = df_p.drop(columns=meta_columns)
    df_c_features = df_c.drop(columns=meta_columns)
    df_combined_features = pd.concat([df_p_features, df_c_features], axis=1)
    df_meta = df_p[meta_columns]
    df_combined = pd.concat([df_combined_features, df_meta], axis=1)
    df_combined.columns = [
        f"x{i}" for i in range(len(df_combined.columns) - len(df_meta.columns))
    ] + df_meta.columns.tolist()
    df_combined.to_csv(out_dir / "cleaned_dataset_pulse_and_counts.csv", index=False)

    df_accx_features = df_x.drop(columns=meta_columns)
    df_accy_features = df_y.drop(columns=meta_columns)
    df_accz_features = df_z.drop(columns=meta_columns)
    df_combined_features_xyz = pd.concat(
        [df_accx_features, df_accy_features, df_accz_features], axis=1
    )
    df_meta = df_x[meta_columns]
    df_combined_xyz = pd.concat([df_combined_features_xyz, df_meta], axis=1)
    df_combined_xyz.columns = [
        f"x{i}" for i in range(len(df_combined_xyz.columns) - len(df_meta.columns))
    ] + df_meta.columns.tolist()
    df_combined_xyz.to_csv(out_dir / "cleaned_dataset_xyz.csv", index=False)

    df_combined_features_xyz_pulse = pd.concat(
        [df_accx_features, df_accy_features, df_accz_features, df_p_features], axis=1
    )
    df_meta = df_x[meta_columns]
    df_combined_xyz_pulse = pd.concat([df_combined_features_xyz_pulse, df_meta], axis=1)
    df_combined_xyz_pulse.columns = [
        f"x{i}"
        for i in range(len(df_combined_xyz_pulse.columns) - len(df_meta.columns))
    ] + df_meta.columns.tolist()
    df_combined_xyz_pulse.to_csv(out_dir / "cleaned_dataset_xyz_pulse.csv", index=False)

    df_m_features = df_m.drop(columns=meta_columns)
    df_combined_features_full = pd.concat(
        [
            df_accx_features,
            df_accy_features,
            df_accz_features,
            df_p_features,
            df_c_features,
            df_m_features,
        ],
        axis=1,
    )
    df_combined_full = pd.concat([df_combined_features_full, df_meta], axis=1)
    df_combined_full.columns = (
        [f"accx_{x}" for x in range(len(df_accx_features.columns))]
        + [f"accy_{x}" for x in range(len(df_accy_features.columns))]
        + [f"accz_{x}" for x in range(len(df_accz_features.columns))]
        + [f"pulse_{x}" for x in range(len(df_p_features.columns))]
        + [f"count_{x}" for x in range(len(df_c_features.columns))]
        + [f"magnitude_{x}" for x in range(len(df_m_features.columns))]
        + df_meta.columns.tolist()
    )
    df_combined_full.to_csv(out_dir / "cleaned_dataset_full.csv", index=False)


def main(meta_datafile, input_dir, out_dir):
    out_dir.mkdir(exist_ok=True, parents=True)
    print(f"input_dir={input_dir}")
    files = list(input_dir.glob("*.csv"))
    vars = ["AccX",
            "AccY",
            "AccZ",
            "MagX",
            "MagY",
            "MagZ",
            "Pulse",
            "SpO2"]
    data_dict = {
        "AccX": [],
        "AccY": [],
        "AccZ": [],
        "MagX": [],
        "MagY": [],
        "MagZ": [],
        "Pulse": [],
        "SpO2": [],
        "Counts": [],
        "Magnitude": [],
        "Home Activity": [],
        "Clinic Activity": [],
        "Period": [],
        "Note": []
    }
    time_dict = {
        "AccX": [],
        "AccY": [],
        "AccZ": [],
        "MagX": [],
        "MagY": [],
        "MagZ": [],
        "Pulse": [],
        "SpO2": [],
        "Counts": [],
        "Magnitude": [],
        "Home Activity": [],
        "Clinic Activity": [],
        "Period": [],
        "Note": []
    }
    id_dict = {
        "AccX": [],
        "AccY": [],
        "AccZ": [],
        "MagX": [],
        "MagY": [],
        "MagZ": [],
        "Pulse": [],
        "SpO2": [],
        "Counts": [],
        "Magnitude": [],
        "Home Activity": [],
        "Clinic Activity": [],
        "Period": [],
        "Note": []
    }
    df_list = {}
    for j, file in enumerate(files):
        print(f"processing {j}/{len(files)} {file}...")
        df = pd.read_csv(file, on_bad_lines="warn")
        df["Counts"] = np.nan
        df["Timestamp"] = df["Date"] + " " + df["Time"]
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%d/%m/%Y %H:%M:%S")
        _, dfs = resample(df)
        for i, df in enumerate(dfs):
            p_id = int(file.stem.split("_")[0].replace("P", ""))
            df = df.head(86400)  # 86400 in a day
            df, stage, breed, clinic, home, start_time, end_time, note = get_stage(
                meta_datafile, df, p_id
            )
            # df = df[
            #     vars + ["Timestamp", "Time_dt"]
            # ].copy()  # Ensure you are working with a copy
            day = pd.to_datetime(df["Timestamp"].values[0]).strftime("%d/%m/%Y")
            label = f"{p_id}__{stage}__{breed}__{day}__{clinic}__{home}__{start_time}__{end_time}__{note}"
            # df["id"] = label

            # label = (str(p_id) + "__" +
            #             str(stage) + "__" +
            #             str(breed) + "__" +
            #             str(day) + "__" +
            #             df["Clinic Activity"] + "__" +
            #             df["Home Activity"] + "__" +
            #             df["Period"] + "__" +
            #             df["Note"] + "__" +
            #             str(i)
            #          ).tolist()
            df["id"] = label

            df["Magnitude"] = np.sqrt(
                df["AccX"] ** 2 + df["AccY"] ** 2 + df["AccZ"] ** 2
            )
            df.index = df["Timestamp"]
            df["Counts"] = df["Magnitude"].rolling("60s").apply(count_above_threshold)
            # vars = ['AccX', 'AccY', 'AccZ', 'MagX', 'MagY', 'MagZ', 'Pulse', 'SpO2', 'Counts', 'Magnitude']
            df_list[p_id] = df

            for var in vars + ["Counts", "Magnitude"]:
                data_dict[var].append(df[var].values.tolist())
                time_dict[var].append(df["Time_dt"].values.tolist())
                id_dict[var].append(f"{label}__{i}")
                # id_dict[var].append(f"{label}__{i}")
                #id_dict[var].append(label)
        # id_list.append((str(p_id) + "__" +
        #                 str(stage) + "__" +
        #                 str(breed) + "__" +
        #                 str(day) + "__" +
        #                 df["Clinic Activity"] + "__" +
        #                 df["Home Activity"] + "__" +
        #                 df["Period"] + "__" +
        #                 df["Note"]).tolist())

    for v in vars + ["Counts", "Magnitude"]:
        data = np.array(data_dict[v])
        time_list = np.array(time_dict[v])
        id_list = np.array(id_dict[v])
        df_data = pd.DataFrame(data, columns=time_list[0], index=id_list)
        build_samples_day(out_dir, v, df_data, df_list)
        # plotly struggles with second resolution data because too many data points
        # df_data = df_data.dropna(axis=1, how="all")
        # z = df_data.values
        # x = df_data.columns.values
        # y = df_data.index.str.replace('_Merged', '').values
        # plot_heatmap(z, x, y, Path("output"), title=f"{v}", filename= f"{v}.png")
        df = df_data.T
        df.index = pd.to_datetime(df.index, format="%H:%M:%S")
        df_resampled = df.resample("min").sum(min_count=1)
        nan_intervals = df.resample("min").apply(lambda x: x.isna().all(axis=0))
        df_resampled[nan_intervals] = float("nan")

        df_resampled = df_resampled.T
        z = df_resampled.values
        x = df_resampled.columns.values
        y = df_resampled.index.values
        plot_heatmap_plotly(z, x, y, out_dir, title=f"{v}", filename=f"{v}.html")


if __name__ == "__main__":
    out_dir = Path("output/datasets5")
    meta_datafile = Path(r"C:\Brooke Study Data\meta_data.csv")
    data_dir = Path(r"C:\Brooke Study Data\Data")
    main(meta_datafile, data_dir, out_dir)
    build_datasets(out_dir)
