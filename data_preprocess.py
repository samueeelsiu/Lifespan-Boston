import pandas as pd
import json
import numpy as np
from datetime import datetime


# Keep only the latest record within a given time window for the same group
# Keep only the latest record within a given time window for the same group,
# prioritizing a specific status.
def drop_near_duplicates(df, time_col, group_cols, status_col, preferred_status='Close', window='24h'):
    """
    For each group (e.g., same parcel_id + worktype), keep only the "best"
    permit within any rolling time window (default: 24h).
    "Best" is defined as:
    1. Any permit matching `preferred_status` (e.g., 'Close').
    2. If multiple preferred, the latest one.
    3. If no preferred, the latest one (original logic).

    Assumes `time_col` is a pandas datetime dtype.
    """

    df_copy = df.copy()

    # --- NEW: Create a priority column ---
    # Give a priority of 1 to the preferred status, 0 to all others.
    # NaNs will be treated as 0.
    if status_col in df_copy.columns:
        df_copy['priority'] = (df_copy[status_col] == preferred_status).astype(int)
    else:
        # If status column is missing, default all priorities to 0
        df_copy['priority'] = 0

    # --- MODIFIED: Sort by priority first, then time ---
    # Sort so that preferred status (priority=1) comes first,
    # and *then* the newest rows come first within that priority.
    sort_cols = group_cols + ['priority', time_col]
    ascending_order = [True] * len(group_cols) + [False, False]  # [group_cols ASC, priority DESC, time_col DESC]

    df_sorted = df_copy.sort_values(sort_cols, ascending=ascending_order)

    keep_idx = []
    # Iterate group by group, selecting "best" first and skipping any earlier
    # record that happens within `window` from the last kept one.
    for _, g in df_sorted.groupby(group_cols, sort=False):
        last_kept_time = None
        for idx, row in g.iterrows():
            if last_kept_time is None or (last_kept_time - row[time_col]) > pd.Timedelta(window):
                keep_idx.append(idx)
                last_kept_time = row[time_col]

    # Return kept rows in chronological order (optional)
    # We must sort by the *original* time_col to restore chronological order
    final_sort_cols = group_cols + [time_col]
    kept = df_copy.loc[keep_idx].sort_values(final_sort_cols)
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
    # Load more columns to identify buildings correctly
    assessment_cols = ['PID', 'CM_ID', 'YR_BUILT']
    if 'CM_ID' not in prop_ass_df.columns:
        print("Warning: 'CM_ID' not found in assessment data. Condominium logic will be skipped.")
        assessment_cols = ['PID', 'YR_BUILT']

    assessment_data = prop_ass_df[assessment_cols].copy()
    assessment_data.dropna(subset=['YR_BUILT'], inplace=True)
    assessment_data = assessment_data[assessment_data['YR_BUILT'] > 0]

    # --- NEW LOGIC: Create a unified 'building_id' ---
    # For condominiums, multiple PIDs (units) share a CM_ID (the building).
    # We use CM_ID as the building identifier if it exists, otherwise fall back to PID.
    if 'CM_ID' in assessment_data.columns:
        # Fill missing CM_ID with the PID of the same row
        assessment_data['building_id'] = assessment_data['CM_ID'].fillna(assessment_data['PID'])
    else:
        assessment_data['building_id'] = assessment_data['PID']

    print("Aggregating build years by 'building_id' (handling condos)...")
    # Group by the new building_id and find the earliest construction year for that building.
    build_years = assessment_data.groupby('building_id')['YR_BUILT'].min().reset_index()
    build_years.rename(columns={'YR_BUILT': 'build_year'}, inplace=True)

    # --- NEW: Calculate current building age distribution (5-year and 10-year) ---
    print("Calculating current building age distribution (5yr & 10yr)...")
    current_year = 2025  # Keep consistent with the rest of the analysis
    all_buildings = prop_ass_df[['YR_BUILT']].copy()
    all_buildings = all_buildings[
        (all_buildings['YR_BUILT'] > 0) &
        (all_buildings['YR_BUILT'] <= current_year)
        ]
    all_buildings['age'] = current_year - all_buildings['YR_BUILT']
    avg_current_age = float(all_buildings['age'].mean()) if not all_buildings.empty else 0.0

    def make_hist(df, width):
        """
        Build a right-open histogram [start, end) for the given bin width.
        Returns a list of dicts like {'range': '0-10', 'count': 123}.
        """
        # Ensure 'age' is numeric and handle potential NaNs
        df = df.dropna(subset=['age'])
        if df.empty:
            return []

        max_age = int(df['age'].max())
        edges = list(range(0, max_age + width + 1, width))
        out = []
        for i in range(len(edges) - 1):
            s, e = edges[i], edges[i + 1]
            cnt = len(df[(df['age'] >= s) & (df['age'] < e)])
            if cnt > 0:
                out.append({'range': f"{s}-{e}", 'count': int(cnt)})
        return out

    # Build both 5-year and 10-year distributions
    current_age_distribution_5yr = make_hist(all_buildings, 5)
    current_age_distribution_10yr = make_hist(all_buildings, 10)

    # --- 3. Process all demolition permits together (with 1-hour de-dup per parcel+type) ---
    print("Processing demolition permits with de-duplication (1-hour window per parcel+type)...")
    demolition_types = ['EXTDEM', 'INTDEM', 'RAZE']

    base_cols = ['worktype', 'issued_date', 'parcel_id']
    present_cols = [c for c in base_cols if c in bldg_permit_df.columns]
    if len(present_cols) < len(base_cols):
        print("Error: Permit file is missing required columns among", base_cols)

    geo_cols = [c for c in ['y_latitude', 'x_longitude'] if c in bldg_permit_df.columns]

    status_candidates = ['status', 'STATUS', 'permit_status', 'PERMIT_STATUS', 'current_status', 'Current Status']
    status_col = next((c for c in status_candidates if c in bldg_permit_df.columns), None)

    permit_cols = present_cols + geo_cols + ([status_col] if status_col else [])

    demolition_permits = bldg_permit_df[bldg_permit_df['worktype'].isin(demolition_types)][permit_cols].copy()

    if demolition_permits.empty:
        print("No demolition permits found of types EXTDEM, INTDEM, or RAZE.")
        return None

    demolition_permits['issued_date'] = pd.to_datetime(demolition_permits['issued_date'], errors='coerce')
    demolition_permits.dropna(subset=['issued_date'], inplace=True)

    # --- MOVED: Normalize STATUS immediately after creating lifespan_df_all ---
    print("Normalizing permit status ('Close'/'Open')...")
    # Check lifespan_df_all (the source) for the status column
    status_col_detected = status_col if status_col in demolition_permits.columns else None

    def normalize_status(val):
        if pd.isna(val):
            return None  # Keep NaN as None
        s = str(val).strip().upper()
        # Define 'Close' statuses
        if s in ('CLOSED', 'CLOSE', 'CLOSED OUT', 'COMPLETE', 'COMPLETED'):
            return 'Close'
        # Define 'Open' statuses
        if s in ('OPEN', 'ACTIVE', 'ISSUED', 'PENDING'):
            return 'Open'
        return None  # Other statuses (e.g., 'Cancelled') become None

    if status_col_detected:
        print(f"Normalizing status using column: {status_col_detected}")
        # Apply normalization to the source DataFrame
        demolition_permits['status_norm'] = demolition_permits[status_col_detected].map(normalize_status)
    else:
        print("Warning: No status column found. All 'status_norm' will be None.")
        # Add an empty column to the source DataFrame
        demolition_permits['status_norm'] = None
    # --- END OF MOVED BLOCK ---

    dedup_permits = drop_near_duplicates(
        demolition_permits,
        time_col='issued_date',
        group_cols=['parcel_id', 'worktype'],
        status_col='status_norm',  # <-- Add this
        preferred_status='Close',  # <-- Add this
        window='24h'
    ).copy()

    dedup_permits['demolition_year'] = dedup_permits['issued_date'].dt.year

    # --- 4. Merge with build years and calculate lifespan ---
    print("Merging data and calculating lifespan...")
    lifespan_df_all = pd.merge(dedup_permits, build_years, left_on='parcel_id', right_on='building_id', how='inner')
    lifespan_df_all['lifespan'] = lifespan_df_all['demolition_year'] - lifespan_df_all['build_year']

    # --- NEW: Enforce one final RAZE event per building ---
    print("Enforcing a single, definitive RAZE event per building...")
    # ... (rest of the script continues from here) ...

    # --- NEW: Enforce one final RAZE event per building ---
    print("Enforcing a single, definitive RAZE event per building...")

    raze_df = lifespan_df_all[lifespan_df_all['worktype'] == 'RAZE'].copy()

    multi_raze_records = []
    if not raze_df.empty:
        # Prepare columns for multi-raze detection
        use_cols = ['building_id', 'build_year', 'demolition_year', 'issued_date']
        if 'issued_date' not in raze_df.columns:
            raze_df['issued_date'] = pd.NaT  # Add empty column if missing
        if status_col and status_col in raze_df.columns:
            use_cols.append(status_col)
        else:
            # Add an empty status col if it doesn't exist, so the logic doesn't break
            status_col = 'status_placeholder'  # Use a placeholder name
            raze_df[status_col] = None
            use_cols.append(status_col)

        tmp = raze_df[use_cols].dropna(subset=['building_id', 'demolition_year']).copy()

        # Ensure correct dtypes
        tmp['demolition_year'] = tmp['demolition_year'].astype(int, errors='ignore')
        if 'build_year' in tmp.columns:
            tmp['build_year'] = pd.to_numeric(tmp['build_year'], errors='coerce')

        # Find parcels with multiple RAZE permits
        for bid, g in tmp.groupby('building_id', sort=False):
            # Group by year and take the last permit of that year
            g = g.sort_values(['demolition_year', 'issued_date']).copy()
            per_year = g.groupby('demolition_year', as_index=False).last()

            # Format permit info
            permits = []
            for _, r in per_year.iterrows():
                permits.append({
                    'year': int(r['demolition_year']),
                    'status': (str(r[status_col]) if (status_col and pd.notna(r[status_col])) else None)
                })

            # If 2+ permits found, add to our list
            if len(permits) >= 2:
                by = pd.to_numeric(g['build_year'], errors='coerce').dropna()
                build_year_value = int(by.iloc[0]) if not by.empty else None

                multi_raze_records.append({
                    'building_id': str(bid),
                    'build_year': build_year_value,
                    'raze_permits': sorted(permits, key=lambda x: x['year'])
                })

    # Separate non-RAZE permits
    other_permits_df = lifespan_df_all[lifespan_df_all['worktype'] != 'RAZE'].copy()

    # Apply the "one final RAZE" logic
    if not raze_df.empty:
        # Sort to find the "best" permit: positive lifespan first, then latest demo year
        raze_df_sorted = raze_df.sort_values(
            by=['building_id', 'lifespan', 'demolition_year'],
            ascending=[True, False, False]
        )
        # Keep only the top-ranked permit for each building_id
        final_raze_df = raze_df_sorted.drop_duplicates(subset=['building_id'], keep='first')
    else:
        final_raze_df = pd.DataFrame()  # Create empty DF if no RAZE permits

    # Recombine the definitive RAZE permits with all other permit types
    lifespan_df_all_cleaned = pd.concat([final_raze_df, other_permits_df], ignore_index=True)

    print(f"RAZE permits reduced from {len(raze_df)} to {len(final_raze_df)} after applying logic.")


    # --- 5. Split data based on the CLEANED dataframe ---

    # --- "All" Data (Default) ---
    # Negative-age RAZE (to be stacked as 'demolished and replaced')
    replaced_raze_df = lifespan_df_all_cleaned[
        (lifespan_df_all_cleaned['worktype'] == 'RAZE') & (lifespan_df_all_cleaned['lifespan'] <= 0)
        ].copy()
    # Keep positive lifespans for the main analyses
    lifespan_df = lifespan_df_all_cleaned[lifespan_df_all_cleaned['lifespan'] > 0].copy()

    # --- NEW: Create "Closed Only" DataFrames ---
    print("Creating 'Closed' only data subsets...")
    replaced_raze_df_closed = replaced_raze_df[
        replaced_raze_df['status_norm'] == 'Close'
        ].copy()
    lifespan_df_closed = lifespan_df[
        lifespan_df['status_norm'] == 'Close'
        ].copy()

    # --- Get counts for logging ---
    initial_record_count = len(lifespan_df_all)
    final_record_count = len(lifespan_df)
    final_record_count_closed = len(lifespan_df_closed)  # NEW

    print(f"\nRecords after merge: {initial_record_count} (all)")
    print(f"Final valid records (lifespan > 0): {final_record_count}")
    print(f"Final 'Closed' valid records (lifespan > 0): {final_record_count_closed}")  # NEW

    # Drop records with no coordinates for mapping
    has_geo_data = 'y_latitude' in lifespan_df_all.columns and 'x_longitude' in lifespan_df_all.columns
    if has_geo_data:
        lifespan_df = lifespan_df.dropna(subset=['y_latitude', 'x_longitude']).copy()
        lifespan_df_closed = lifespan_df_closed.dropna(subset=['y_latitude', 'x_longitude']).copy()  # NEW
        print(f"Final valid records with coordinates (lifespan > 0): {len(lifespan_df)}")
        print(f"Final 'Closed' valid records with coordinates (lifespan > 0): {len(lifespan_df_closed)}")  # NEW

    if lifespan_df.empty and replaced_raze_df.empty:
        print("No valid records after merging and cleaning.")
        return None

    # --- 6. Start Aggregating Data for JSON ---
    print("Starting data aggregations for JSON export...")
    all_data = {}  # This will be our final JSON object
    all_data['multi_raze_parcels'] = multi_raze_records  # This chart is not filtered by status

    # --- Define year span from ALL data (positive and replaced) ---
    pos_year_min = lifespan_df['demolition_year'].min() if not lifespan_df.empty else np.inf
    pos_year_max = lifespan_df['demolition_year'].max() if not lifespan_df.empty else -np.inf
    rep_year_min = replaced_raze_df['demolition_year'].min() if not replaced_raze_df.empty else np.inf
    rep_year_max = replaced_raze_df['demolition_year'].max() if not replaced_raze_df.empty else -np.inf

    min_year = int(min(pos_year_min, rep_year_min)) if min(pos_year_min,
                                                           rep_year_min) != np.inf else datetime.now().year
    max_year = int(max(pos_year_max, rep_year_max)) if max(pos_year_max,
                                                           rep_year_max) != -np.inf else datetime.now().year
    years = list(range(min_year, max_year + 1)) if min_year <= max_year else []

    # --- Get counts for 'All' and 'Closed' (positive lifespan only) ---
    type_counts_pos = lifespan_df['worktype'].value_counts().to_dict()
    type_counts_pos_closed = lifespan_df_closed['worktype'].value_counts().to_dict()  # NEW

    # --- Summary Stats (All Data) ---
    # Get counts for RAZE <= 0 (from the *original* uncleaned RAZE df)
    zero_mask_all = (lifespan_df_all_cleaned['worktype'] == 'RAZE') & (lifespan_df_all_cleaned['lifespan'] == 0)
    neg_mask_all = (lifespan_df_all_cleaned['worktype'] == 'RAZE') & (lifespan_df_all_cleaned['lifespan'] < 0)

    # Helper for status counts
    def count_co(df):
        if df.empty:
            return {'close': 0, 'open': 0}
        vc = df['status_norm'].value_counts(dropna=True).to_dict()
        return {
            'close': int(vc.get('Close', 0)),
            'open': int(vc.get('Open', 0))
        }

    # This summary block is NOT filtered and is used for the RAZE Lifespan Summary chart
    sb_positive = count_co(lifespan_df_all_cleaned[
                               (lifespan_df_all_cleaned['worktype'] == 'RAZE') &
                               (lifespan_df_all_cleaned['lifespan'] > 0)
                               ])
    sb_zero = count_co(lifespan_df_all_cleaned[zero_mask_all])
    sb_negative = count_co(lifespan_df_all_cleaned[neg_mask_all])
    sb_total = {
        'close': sb_positive['close'] + sb_zero['close'] + sb_negative['close'],
        'open': sb_positive['open'] + sb_zero['open'] + sb_negative['open'],
    }

    all_data['summary_stats'] = {
        'total_demolitions': final_record_count,
        'average_lifespan': float(lifespan_df['lifespan'].mean()) if final_record_count else 0.0,
        'median_lifespan': float(lifespan_df['lifespan'].median()) if final_record_count else 0.0,
        'min_lifespan': int(lifespan_df['lifespan'].min()) if final_record_count else 0,
        'max_lifespan': int(lifespan_df['lifespan'].max()) if final_record_count else 0,
        'extdem_count': type_counts_pos.get('EXTDEM', 0),
        'intdem_count': type_counts_pos.get('INTDEM', 0),
        'raze_count': type_counts_pos.get('RAZE', 0),  # positive-lifespan RAZE
        'negative_raze_count': int(neg_mask_all.sum()),
        'zero_raze_count': int(zero_mask_all.sum()),
        'demolished_and_replaced_count': int(len(replaced_raze_df)),
        'avg_current_building_age': avg_current_age,
        'raze_status_by_lifespan': {  # This specific sub-object is used by the RAZE summary chart
            'positive': sb_positive,
            'zero': sb_zero,
            'negative': sb_negative,
            'total': sb_total
        }
    }

    # --- NEW: Summary Stats (Closed Only) ---
    all_data['summary_stats_closed'] = {
        'total_demolitions': final_record_count_closed,
        'average_lifespan': float(lifespan_df_closed['lifespan'].mean()) if final_record_count_closed else 0.0,
        'median_lifespan': float(lifespan_df_closed['lifespan'].median()) if final_record_count_closed else 0.0,
        'min_lifespan': int(lifespan_df_closed['lifespan'].min()) if final_record_count_closed else 0,
        'max_lifespan': int(lifespan_df_closed['lifespan'].max()) if final_record_count_closed else 0,
        'extdem_count': type_counts_pos_closed.get('EXTDEM', 0),
        'intdem_count': type_counts_pos_closed.get('INTDEM', 0),
        'raze_count': type_counts_pos_closed.get('RAZE', 0),
        'demolished_and_replaced_count': int(len(replaced_raze_df_closed)),
        # Note: avg_current_building_age is not included as it's independent of permit status
    }

    # --- MODIFIED: Yearly stacked data (now includes 'All' and 'Closed' keys) ---
    print("Aggregating yearly stacked data...")
    yearly_data = []
    for year in years:
        # 'All' data for this year
        y_pos = lifespan_df[lifespan_df['demolition_year'] == year]
        y_rep = replaced_raze_df[replaced_raze_df['demolition_year'] == year]
        # 'Closed' data for this year
        y_pos_closed = lifespan_df_closed[lifespan_df_closed['demolition_year'] == year]
        y_rep_closed = replaced_raze_df_closed[replaced_raze_df_closed['demolition_year'] == year]

        row = {
            'year': int(year),
            # 'All' keys
            'RAZE': int((y_pos['worktype'] == 'RAZE').sum()),
            'EXTDEM': int((y_pos['worktype'] == 'EXTDEM').sum()),
            'INTDEM': int((y_pos['worktype'] == 'INTDEM').sum()),
            'demolished_and_replaced': int(len(y_rep)),
            # 'Closed' keys
            'RAZE_closed': int((y_pos_closed['worktype'] == 'RAZE').sum()),
            'EXTDEM_closed': int((y_pos_closed['worktype'] == 'EXTDEM').sum()),
            'INTDEM_closed': int((y_pos_closed['worktype'] == 'INTDEM').sum()),
            'demolished_and_replaced_closed': int(len(y_rep_closed))
        }
        yearly_data.append(row)
    all_data['yearly_stacked'] = yearly_data

    # --- MODIFIED: Lifespan Distribution (5-year and 10-year bins) ---
    print("Aggregating lifespan distributions (5yr and 10yr)...")

    def get_lifespan_distribution(bin_width):
        """Helper to generate lifespan distribution for 'All' and 'Closed'."""
        if lifespan_df.empty:
            return []

        max_lifespan = int(lifespan_df['lifespan'].max())
        bins = list(range(0, max_lifespan + bin_width + 1, bin_width))
        lifespan_bins = []

        for i in range(len(bins) - 1):
            bin_label = f"{bins[i]}-{bins[i + 1]}"
            bin_data = {'range': bin_label}

            # Filter data for this bin once
            bin_df = lifespan_df[(lifespan_df['lifespan'] >= bins[i]) & (lifespan_df['lifespan'] < bins[i + 1])]
            bin_df_closed = lifespan_df_closed[
                (lifespan_df_closed['lifespan'] >= bins[i]) & (lifespan_df_closed['lifespan'] < bins[i + 1])]

            has_data = False
            for demo_type in demolition_types:
                # 'All' data counts
                count = int((bin_df['worktype'] == demo_type).sum())
                bin_data[demo_type] = count
                if count > 0: has_data = True

                # 'Closed' data counts
                count_closed = int((bin_df_closed['worktype'] == demo_type).sum())
                bin_data[f"{demo_type}_closed"] = count_closed
                if count_closed > 0: has_data = True

            if has_data:  # Only append if there's data in this bin
                lifespan_bins.append(bin_data)
        return lifespan_bins

    # Generate both 5-year and 10-year distributions
    all_data['lifespan_distribution_5yr'] = get_lifespan_distribution(5)
    all_data['lifespan_distribution'] = get_lifespan_distribution(10)  # JS expects 'lifespan_distribution' for 10yr

    # --- Demolition Types (All) ---
    all_data['demolition_types'] = [{'type': demo_type, 'count': type_counts_pos.get(demo_type, 0)}
                                    for demo_type in demolition_types]
    # --- NEW: Demolition Types (Closed) ---
    all_data['demolition_types_closed'] = [{'type': demo_type, 'count': type_counts_pos_closed.get(demo_type, 0)}
                                           for demo_type in demolition_types]

    # --- Lifespan by Type (All) ---
    lifespan_by_type_df = lifespan_df.groupby('worktype')['lifespan'].agg(['mean', 'median', 'count']).reset_index()
    all_data['lifespan_by_type'] = [
        {'type': row['worktype'], 'average': float(row['mean']),
         'median': float(row['median']), 'count': int(row['count'])}
        for index, row in lifespan_by_type_df.iterrows()
    ]

    # --- NEW: Lifespan by Type (Closed) ---
    lifespan_by_type_df_closed = lifespan_df_closed.groupby('worktype')['lifespan'].agg(
        ['mean', 'median', 'count']).reset_index()
    all_data['lifespan_by_type_closed'] = [
        {'type': row['worktype'], 'average': float(row['mean']),
         'median': float(row['median']), 'count': int(row['count'])}
        for index, row in lifespan_by_type_df_closed.iterrows()
    ]

    # --- Current Building Age (Not Filtered) ---
    all_data['current_building_age_distribution'] = current_age_distribution_10yr
    all_data['current_building_age_distribution_10yr'] = current_age_distribution_10yr
    all_data['current_building_age_distribution_5yr'] = current_age_distribution_5yr

    # --- MODIFIED: Yearly Age Distribution (All and Closed) ---
    print("Aggregating yearly age distribution...")
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
    age_bin_ranges = [b['min'] for b in age_bins_definition] + [age_bins_definition[-1]['max']]
    age_bin_labels = [b['label'] for b in age_bins_definition]
    age_bin_ranges[-2] = 150  # Adjust for pd.cut
    age_bin_ranges[-1] = float('inf')

    yearly_age_distribution = {}
    if not lifespan_df.empty:
        lifespan_df['demolition_year'] = lifespan_df['demolition_year'].astype(int)
        lifespan_df_closed['demolition_year'] = lifespan_df_closed['demolition_year'].astype(int)

    for year in years:
        yearly_age_distribution[year] = {}
        year_df = lifespan_df[lifespan_df['demolition_year'] == year]
        year_df_closed = lifespan_df_closed[lifespan_df_closed['demolition_year'] == year]  # NEW

        types_to_calculate = ['All'] + demolition_types

        for demo_type in types_to_calculate:
            # --- Process 'All' data ---
            type_key = demo_type  # e.g., 'RAZE'
            if demo_type == 'All':
                type_df = year_df
            else:
                type_df = year_df[year_df['worktype'] == demo_type]

            age_counts = {b['label']: 0 for b in age_bins_definition}
            if not type_df.empty:
                lifespan_series = pd.cut(type_df['lifespan'], bins=age_bin_ranges, labels=age_bin_labels, right=False,
                                         include_lowest=True)
                value_counts = lifespan_series.value_counts().to_dict()
                age_counts.update(value_counts)
            yearly_age_distribution[year][type_key] = age_counts

            # --- NEW: Process 'Closed' data ---
            type_key_closed = f"{demo_type}_closed"  # e.g., 'RAZE_closed'
            if demo_type == 'All':
                type_df_closed = year_df_closed
            else:
                type_df_closed = year_df_closed[year_df_closed['worktype'] == demo_type]

            age_counts_closed = {b['label']: 0 for b in age_bins_definition}
            if not type_df_closed.empty:
                lifespan_series_closed = pd.cut(type_df_closed['lifespan'], bins=age_bin_ranges, labels=age_bin_labels,
                                                right=False, include_lowest=True)
                value_counts_closed = lifespan_series_closed.value_counts().to_dict()
                age_counts_closed.update(value_counts_closed)
            yearly_age_distribution[year][type_key_closed] = age_counts_closed

    all_data['yearly_age_distribution'] = yearly_age_distribution

    # --- MODIFIED: Construction Era Distribution (All and Closed) ---
    print("Aggregating yearly construction era distribution...")
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
        year_df_closed = lifespan_df_closed[lifespan_df_closed['demolition_year'] == year]  # NEW

        types_to_calculate = ['All'] + demolition_types

        for demo_type in types_to_calculate:
            # --- Process 'All' data ---
            type_key = demo_type  # e.g., 'RAZE'
            if demo_type == 'All':
                type_df = year_df
            else:
                type_df = year_df[year_df['worktype'] == demo_type]

            era_counts = {era['label']: 0 for era in construction_eras}
            if not type_df.empty:
                for era in construction_eras:
                    count = len(type_df[(type_df['build_year'] >= era['min']) & (type_df['build_year'] < era['max'])])
                    era_counts[era['label']] = count
            yearly_construction_era[year][type_key] = era_counts

            # --- NEW: Process 'Closed' data ---
            type_key_closed = f"{demo_type}_closed"  # e.g., 'RAZE_closed'
            if demo_type == 'All':
                type_df_closed = year_df_closed
            else:
                type_df_closed = year_df_closed[year_df_closed['worktype'] == demo_type]

            era_counts_closed = {era['label']: 0 for era in construction_eras}
            if not type_df_closed.empty:
                for era in construction_eras:
                    count_closed = len(type_df_closed[(type_df_closed['build_year'] >= era['min']) & (
                                type_df_closed['build_year'] < era['max'])])
                    era_counts_closed[era['label']] = count_closed
            yearly_construction_era[year][type_key_closed] = era_counts_closed

    all_data['yearly_construction_era'] = yearly_construction_era

    # --- MODIFIED: Lifespan by Year Boxplot (All and Closed) ---
    print("Aggregating lifespan-by-year boxplot data...")
    lifespan_by_year_boxplot = {}
    all_years_sorted = sorted(lifespan_df['demolition_year'].unique().astype(int))

    for demo_type in ['All'] + demolition_types:
        # --- Process 'All' data ---
        type_key = demo_type  # e.g., 'RAZE'
        type_data = []
        for year in all_years_sorted:
            if demo_type == 'All':
                year_df = lifespan_df[lifespan_df['demolition_year'] == year]
            else:
                year_df = lifespan_df[(lifespan_df['demolition_year'] == year) & (lifespan_df['worktype'] == demo_type)]

            lifespans_for_year = year_df['lifespan'].tolist()
            if lifespans_for_year:
                type_data.append({'year': int(year), 'lifespans': lifespans_for_year})
        lifespan_by_year_boxplot[type_key] = type_data

        # --- NEW: Process 'Closed' data ---
        type_key_closed = f"{demo_type}_closed"  # e.g., 'RAZE_closed'
        type_data_closed = []
        for year in all_years_sorted:
            if demo_type == 'All':
                year_df_closed = lifespan_df_closed[lifespan_df_closed['demolition_year'] == year]
            else:
                year_df_closed = lifespan_df_closed[
                    (lifespan_df_closed['demolition_year'] == year) & (lifespan_df_closed['worktype'] == demo_type)]

            lifespans_for_year_closed = year_df_closed['lifespan'].tolist()
            if lifespans_for_year_closed:
                type_data_closed.append({'year': int(year), 'lifespans': lifespans_for_year_closed})
        lifespan_by_year_boxplot[type_key_closed] = type_data_closed

    all_data['lifespan_by_year_boxplot'] = lifespan_by_year_boxplot

    # --- MODIFIED: Map Points (now includes 'status') ---
    if has_geo_data:
        print("Generating data for map plot (including status)...")
        # Select 'status_norm' and rename it to 'status' for the JSON
        map_df = lifespan_df[['y_latitude', 'x_longitude', 'worktype', 'lifespan', 'status_norm']].copy()
        map_df.rename(columns={
            'y_latitude': 'lat',
            'x_longitude': 'lng',
            'worktype': 'type',
            'status_norm': 'status'  # NEW
        }, inplace=True)
        all_data['map_points'] = map_df.to_dict(orient='records')
    else:
        all_data['map_points'] = []

    # --- Metadata (unchanged, refers to 'All' data) ---
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

    print("All aggregations complete.")
    return all_data


def save_json_files(data, output_prefix='boston_demolition'):
    """Saves the final data dictionary to a JSON file."""
    if not data:
        print("No data to save")
        return
    main_file = f"{output_prefix}_data.json"
    try:
        with open(main_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"\nSaved main data to {main_file}")
    except Exception as e:
        print(f"Error saving JSON file: {e}")


def main():
    """Main execution function."""
    # Define your input files here
    assessment_file = 'fy2025-property-assessment-data_12_30_2024.csv'
    permit_file = 'tmpbtz4x7bc.csv'

    print("Starting Boston Demolition Data Processing...")
    print("=" * 50)

    processed_data = process_demolition_data(assessment_file, permit_file)

    if processed_data:
        save_json_files(processed_data, output_prefix='boston_demolition')
        print("\n" + "=" * 50)
        print("PROCESSING COMPLETE!")
        print(f"  Final valid records for charting: {processed_data['metadata']['final_valid_records']}")
        print(f"  Points for map plot: {len(processed_data['map_points'])}")
        print("\n" + "=" * 50)
    else:
        print("\n" + "=" * 50)
        print("PROCESSING FAILED.")
        print("Failed to process data. Please check your CSV files and their contents.")
        print("=" * 50)


if __name__ == "__main__":
    main()