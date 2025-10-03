import pandas as pd
import json
import numpy as np
from datetime import datetime


# Keep only the latest record within a given time window for the same group
def drop_near_duplicates(df, time_col, group_cols, window='1H'):
    """
    For each group (e.g., same parcel_id + worktype), keep only the latest
    permit within any rolling time window (default: 1 hour). Assumes `time_col`
    is a pandas datetime dtype.
    """
    # Sort so that the newest rows come first within each group
    sort_cols = group_cols + [time_col]
    df = df.sort_values(sort_cols, ascending=[True]*len(group_cols) + [False]).copy()

    keep_idx = []
    # Iterate group by group, selecting newest first and skipping any earlier
    # record that happens within `window` from the last kept one.
    for _, g in df.groupby(group_cols, sort=False):
        last_kept_time = None
        for idx, row in g.iterrows():
            if last_kept_time is None or (last_kept_time - row[time_col]) > pd.Timedelta(window):
                keep_idx.append(idx)
                last_kept_time = row[time_col]

    # Return kept rows in chronological order (optional)
    kept = df.loc[keep_idx].sort_values(sort_cols)
    return kept


def process_demolition_data(assessment_file, permit_file):
    """
    Process Boston demolition data and generate JSON files for web dashboard.
    """
    # --- 1. Load the datasets ---
    print("Loading datasets...")
    try:
        prop_ass_df = pd.read_csv(assessment_file, low_memory=False)
        bldg_permit_df = pd.read_csv(permit_file, low_memory=False)
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        return None

    # --- 2. Prepare property assessment data ---
    print("Processing property assessment data...")
    assessment_data = prop_ass_df[['PID', 'YR_BUILT']].copy()
    assessment_data.dropna(subset=['YR_BUILT'], inplace=True)
    assessment_data = assessment_data[assessment_data['YR_BUILT'] > 0]
    build_years = assessment_data.groupby('PID')['YR_BUILT'].min().reset_index()
    build_years.rename(columns={'YR_BUILT': 'build_year'}, inplace=True)

    # --- NEW: Calculate current building age distribution (5-year and 10-year) ---
    print("Calculating current building age distribution (5yr & 10yr)...")
    current_year = 2025  # Keep consistent with the rest of the analysis
    all_buildings = prop_ass_df[['YR_BUILT']].copy()
    all_buildings = all_buildings[all_buildings['YR_BUILT'] > 0]
    all_buildings['age'] = current_year - all_buildings['YR_BUILT']

    def make_hist(df, width):
        """
        Build a right-open histogram [start, end) for the given bin width.
        Returns a list of dicts like {'range': '0-10', 'count': 123}.
        """
        max_age = int(df['age'].max())
        edges = list(range(0, max_age + width + 1, width))
        out = []
        for i in range(len(edges) - 1):
            s, e = edges[i], edges[i + 1]
            cnt = len(df[(df['age'] >= s) & (df['age'] < e)])
            if cnt > 0:  # keep output compact: drop trailing all-zero bins
                out.append({'range': f"{s}-{e}", 'count': int(cnt)})
        return out

    # Build both 5-year and 10-year distributions
    current_age_distribution_5yr = make_hist(all_buildings, 5)
    current_age_distribution_10yr = make_hist(all_buildings, 10)

    # --- 3. Process all demolition permits together (with 1-hour de-dup per parcel+type) ---
    print("Processing demolition permits with de-duplication (1-hour window per parcel+type)...")
    demolition_types = ['EXTDEM', 'INTDEM', 'RAZE']
    permit_cols = ['worktype', 'issued_date', 'parcel_id', 'y_latitude', 'x_longitude']
    if not all(col in bldg_permit_df.columns for col in permit_cols):
        print("Error: Permit file is missing required columns. Proceeding without map data.")
        permit_cols = ['worktype', 'issued_date', 'parcel_id']

    demolition_permits = bldg_permit_df[bldg_permit_df['worktype'].isin(demolition_types)][permit_cols].copy()
    if demolition_permits.empty:
        print("No demolition permits found of types EXTDEM, INTDEM, or RAZE.")
        return None

    # Parse datetime and drop rows with invalid dates
    demolition_permits['issued_date'] = pd.to_datetime(demolition_permits['issued_date'], errors='coerce')
    demolition_permits.dropna(subset=['issued_date'], inplace=True)

    # De-duplicate: same parcel_id + same worktype within 1 hour → keep the latest
    dedup_permits = drop_near_duplicates(
        demolition_permits,
        time_col='issued_date',
        group_cols=['parcel_id', 'worktype'],
        window='1H'
    ).copy()

    # Derive demolition year after de-dup
    dedup_permits['demolition_year'] = dedup_permits['issued_date'].dt.year

    # --- 4. Merge with build years and calculate lifespan (split positive vs negative) ---
    print("Merging data and calculating lifespan...")
    lifespan_df_all = pd.merge(dedup_permits, build_years, left_on='parcel_id', right_on='PID', how='inner')
    lifespan_df_all['lifespan'] = lifespan_df_all['demolition_year'] - lifespan_df_all['build_year']

    # Negative-age RAZE (to be stacked as 'demolished and replaced')
    replaced_raze_df = lifespan_df_all[(lifespan_df_all['worktype'] == 'RAZE') & (lifespan_df_all['lifespan'] <= 0)].copy()


    # Keep positive lifespans for the main analyses
    lifespan_df = lifespan_df_all[lifespan_df_all['lifespan'] > 0].copy()

    initial_record_count = len(lifespan_df_all)
    final_record_count = len(lifespan_df)

    print(f"\nRecords after merge: {initial_record_count} (all)")
    print(f"Final valid records (lifespan > 0): {final_record_count}")

    # Detect geo columns
    has_geo_data = 'y_latitude' in lifespan_df_all.columns and 'x_longitude' in lifespan_df_all.columns
    if has_geo_data:
        # Only keep geo rows within rough Boston bounds for positive-lifespan data
        lifespan_df.dropna(subset=['y_latitude', 'x_longitude'], inplace=True)
        lifespan_df = lifespan_df[lifespan_df['y_latitude'].between(42, 43)]
        lifespan_df = lifespan_df[lifespan_df['x_longitude'].between(-72, -70)]
        print(f"Final valid records with coordinates (lifespan > 0): {len(lifespan_df)}")

    # If both positive-lifespan data and replaced (<=0) RAZE are empty, stop.
    if lifespan_df.empty and replaced_raze_df.empty:
        print("No valid records after merging and cleaning.")
        return None

    # --- Yearly stacked data (RAZE <=0 goes to 'demolished_and_replaced') ---
    all_data = {}

    type_counts_pos = lifespan_df['worktype'].value_counts().to_dict()

    # Define year span from BOTH positive-lifespan data and replaced (<=0) RAZE
    pos_year_min = lifespan_df['demolition_year'].min() if not lifespan_df.empty else np.inf
    pos_year_max = lifespan_df['demolition_year'].max() if not lifespan_df.empty else -np.inf
    rep_year_min = replaced_raze_df['demolition_year'].min() if not replaced_raze_df.empty else np.inf
    rep_year_max = replaced_raze_df['demolition_year'].max() if not replaced_raze_df.empty else -np.inf

    min_year = int(min(pos_year_min, rep_year_min))
    max_year = int(max(pos_year_max, rep_year_max))
    years = list(range(min_year, max_year + 1)) if min_year <= max_year else []

    # Summary Stats (positive lifespans only, plus explicit RAZE <=0 counts)
    all_data['summary_stats'] = {
        'total_demolitions': final_record_count,
        'average_lifespan': float(lifespan_df['lifespan'].mean()) if final_record_count else 0.0,
        'median_lifespan': float(lifespan_df['lifespan'].median()) if final_record_count else 0.0,
        'min_lifespan': int(lifespan_df['lifespan'].min()) if final_record_count else 0,
        'max_lifespan': int(lifespan_df['lifespan'].max()) if final_record_count else 0,
        'extdem_count': type_counts_pos.get('EXTDEM', 0),
        'intdem_count': type_counts_pos.get('INTDEM', 0),
        'raze_count': type_counts_pos.get('RAZE', 0),  # NOTE: this is after geo-filter (positive-lifespan only)
        # clear counts for RAZE by lifespan sign (based on merged ALL data, before geo filter)
        'negative_raze_count': int(((lifespan_df_all['worktype'] == 'RAZE') & (lifespan_df_all['lifespan'] < 0)).sum()),
        'zero_raze_count': int(((lifespan_df_all['worktype'] == 'RAZE') & (lifespan_df_all['lifespan'] == 0)).sum()),
        'demolished_and_replaced_count': int(len(replaced_raze_df))  # RAZE with lifespan <= 0
    }

    # Build yearly stacked series
    yearly_data = []
    for year in years:
        # counts from positive-lifespan data
        y_pos = lifespan_df[lifespan_df['demolition_year'] == year]
        row = {
            'year': int(year),
            'RAZE': int((y_pos['worktype'] == 'RAZE').sum()),
            'EXTDEM': int((y_pos['worktype'] == 'EXTDEM').sum()),
            'INTDEM': int((y_pos['worktype'] == 'INTDEM').sum()),
            # extra stack: negative-age RAZE
            'demolished_and_replaced': int((replaced_raze_df['demolition_year'] == year).sum())
        }
        yearly_data.append(row)

    all_data['yearly_stacked'] = yearly_data

    # Lifespan distribution (5-year bins)
    bins_5 = list(range(0, int(lifespan_df['lifespan'].max()) + 6, 5))
    lifespan_bins_5 = []
    for i in range(len(bins_5) - 1):
        bin_label = f"{bins_5[i]}-{bins_5[i + 1]}"
        bin_data = {'range': bin_label}
        for demo_type in demolition_types:
            count = len(lifespan_df[(lifespan_df['worktype'] == demo_type) &
                                    (lifespan_df['lifespan'] >= bins_5[i]) &
                                    (lifespan_df['lifespan'] < bins_5[i + 1])])
            bin_data[demo_type] = count
        if any(bin_data[dt] > 0 for dt in demolition_types):
            lifespan_bins_5.append(bin_data)
    all_data['lifespan_distribution_5yr'] = lifespan_bins_5

    all_data['demolition_types'] = [{'type': demo_type, 'count': type_counts_pos.get(demo_type, 0)}
                                    for demo_type in demolition_types]

    lifespan_by_type_df = lifespan_df.groupby('worktype')['lifespan'].agg(['mean', 'median', 'count']).reset_index()
    all_data['lifespan_by_type'] = [
        {'type': row['worktype'], 'average': float(row['mean']),
         'median': float(row['median']), 'count': int(row['count'])}
        for index, row in lifespan_by_type_df.iterrows()
    ]

    # --- NEW: Add current building age distributions (5yr & 10yr) to data ---
    # Backward-compatible key (10-year bins):
    all_data['current_building_age_distribution'] = current_age_distribution_10yr
    # Explicit keys for clarity:
    all_data['current_building_age_distribution_10yr'] = current_age_distribution_10yr
    all_data['current_building_age_distribution_5yr'] = current_age_distribution_5yr

    # --- 8. Yearly Age Distribution ---
    print("Performing accurate calculation for Yearly Age Distribution chart...")
    age_bins_definition = [
        {'label': '0-5 years', 'min': 0, 'max': 5},
        {'label': '5-10 years', 'min': 5, 'max': 10},
        {'label': '10-20 years', 'min': 10, 'max': 20},
        {'label': '20-30 years', 'min': 20, 'max': 30},
        {'label': '30-50 years', 'min': 30, 'max': 50},
        {'label': '50-75 years', 'min': 50, 'max': 75},
        {'label': '75-100 years', 'min': 75, 'max': 100},
        {'label': '100-150 years', 'min': 100, 'max': 150},
        {'label': '150+ years', 'min': 150, 'max': float('inf')}
    ]

    yearly_age_distribution = {}
    lifespan_df['demolition_year'] = lifespan_df['demolition_year'].astype(int)

    for year in years:
        yearly_age_distribution[year] = {}
        year_df = lifespan_df[lifespan_df['demolition_year'] == year]
        types_to_calculate = ['All'] + demolition_types

        for demo_type in types_to_calculate:
            if demo_type == 'All':
                type_df = year_df
            else:
                type_df = year_df[year_df['worktype'] == demo_type]

            age_counts = {b['label']: 0 for b in age_bins_definition}

            if not type_df.empty:
                bin_ranges = [b['min'] for b in age_bins_definition] + [age_bins_definition[-1]['max']]
                bin_labels = [b['label'] for b in age_bins_definition]
                bin_ranges[-2] = 150
                bin_ranges[-1] = float('inf')

                lifespan_series = pd.cut(
                    type_df['lifespan'],
                    bins=bin_ranges,
                    labels=bin_labels,
                    right=False,
                    include_lowest=True
                )
                value_counts = lifespan_series.value_counts().to_dict()
                age_counts.update(value_counts)

            yearly_age_distribution[year][demo_type] = age_counts

    all_data['yearly_age_distribution'] = yearly_age_distribution

    # --- NEW: Construction Era Distribution ---
    print("Calculating construction era distribution...")
    construction_eras = [
        {'label': 'Pre-1900', 'min': 0, 'max': 1900},
        {'label': '1900-1920', 'min': 1900, 'max': 1920},
        {'label': '1920-1940', 'min': 1920, 'max': 1940},
        {'label': '1940-1960', 'min': 1940, 'max': 1960},
        {'label': '1960-1980', 'min': 1960, 'max': 1980},
        {'label': '1980-2000', 'min': 1980, 'max': 2000},
        {'label': '2000-2010', 'min': 2000, 'max': 2010},
        {'label': '2010-2020', 'min': 2010, 'max': 2020},
        {'label': '2020+', 'min': 2020, 'max': float('inf')}
    ]

    yearly_construction_era = {}
    for year in years:
        yearly_construction_era[year] = {}
        year_df = lifespan_df[lifespan_df['demolition_year'] == year]

        types_to_calculate = ['All'] + demolition_types
        for demo_type in types_to_calculate:
            if demo_type == 'All':
                type_df = year_df
            else:
                type_df = year_df[year_df['worktype'] == demo_type]

            era_counts = {era['label']: 0 for era in construction_eras}

            if not type_df.empty:
                for era in construction_eras:
                    if era['max'] == float('inf'):
                        count = len(type_df[type_df['build_year'] >= era['min']])
                    else:
                        count = len(type_df[(type_df['build_year'] >= era['min']) &
                                            (type_df['build_year'] < era['max'])])
                    era_counts[era['label']] = count

            yearly_construction_era[year][demo_type] = era_counts

    all_data['yearly_construction_era'] = yearly_construction_era

    # Box plot data - filtering by type
    print("Generating data for lifespan by year box plot...")
    lifespan_by_year_boxplot = {}

    for demo_type in ['All'] + demolition_types:
        type_data = []
        for year in sorted(lifespan_df['demolition_year'].unique()):
            if demo_type == 'All':
                year_df = lifespan_df[lifespan_df['demolition_year'] == year]
            else:
                year_df = lifespan_df[(lifespan_df['demolition_year'] == year) &
                                      (lifespan_df['worktype'] == demo_type)]

            lifespans_for_year = year_df['lifespan'].tolist()
            if lifespans_for_year:
                type_data.append({'year': int(year), 'lifespans': lifespans_for_year})

        lifespan_by_year_boxplot[demo_type] = type_data

    all_data['lifespan_by_year_boxplot'] = lifespan_by_year_boxplot

    # Map data
    if has_geo_data:
        print("Generating data for map plot...")
        map_df = lifespan_df[['y_latitude', 'x_longitude', 'worktype', 'lifespan']].copy()
        map_df.rename(columns={'y_latitude': 'lat', 'x_longitude': 'lng', 'worktype': 'type'}, inplace=True)
        all_data['map_points'] = map_df.to_dict(orient='records')
    else:
        all_data['map_points'] = []

    # Metadata
    all_data['metadata'] = {
        'generated_date': datetime.now().isoformat(),
        'total_parcels_analyzed': final_record_count,
        'year_range': f"{min_year}-{max_year}",
        'data_source': 'Boston Property Assessment & Building Permits',
        'total_buildings_in_assessment': len(build_years),
        'total_demolition_permits': len(demolition_permits),
        'matched_records': initial_record_count,
        'final_valid_records': final_record_count
    }

    return all_data


def save_json_files(data, output_prefix='boston_demolition'):
    if not data:
        print("No data to save")
        return
    main_file = f"{output_prefix}_data.json"
    with open(main_file, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\nSaved main data to {main_file}")


def main():
    assessment_file = 'fy2025-property-assessment-data_12_30_2024.csv'
    permit_file = 'tmpbtz4x7bc.csv'
    print("Starting Boston Demolition Data Processing...")
    print("=" * 50)
    processed_data = process_demolition_data(assessment_file, permit_file)
    if processed_data:
        save_json_files(processed_data)
        print("\n" + "=" * 50)
        print("PROCESSING COMPLETE!")
        print(f"  Final valid records for charting: {processed_data['metadata']['final_valid_records']}")
        print(f"  Points for map plot: {len(processed_data['map_points'])}")
        print("\n" + "=" * 50)
    else:
        print("Failed to process data. Please check your CSV files and their contents.")


if __name__ == "__main__":
    main()