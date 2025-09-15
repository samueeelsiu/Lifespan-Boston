import pandas as pd
import json
import numpy as np
from datetime import datetime


def process_demolition_data(assessment_file, permit_file):
    """
    Process Boston demolition data and generate JSON files for web dashboard.

    Note: While buildings in the assessment data may date back to the 1900s,
    the demolition permit records typically cover recent years (2009-2025),
    reflecting modern record-keeping practices and recent urban development.
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

    # --- 3. Process all demolition permits together (MODIFIED LOGIC) ---
    print("Processing demolition permits with de-duplication...")
    demolition_types = ['EXTDEM', 'INTDEM', 'RAZE']

    # **MODIFICATION**: Use correct column names 'y_latitude' and 'x_longitude'
    permit_cols = ['worktype', 'issued_date', 'parcel_id', 'y_latitude', 'x_longitude']
    if not all(col in bldg_permit_df.columns for col in permit_cols):
        print(
            "Error: Permit file is missing required columns (worktype, issued_date, parcel_id, y_latitude, x_longitude).")
        # Fallback to not include geo data if columns are missing
        permit_cols = ['worktype', 'issued_date', 'parcel_id']
        print("Warning: Proceeding without generating map data.")

    demolition_permits = bldg_permit_df[bldg_permit_df['worktype'].isin(demolition_types)][permit_cols].copy()

    if demolition_permits.empty:
        print("No demolition permits found of types EXTDEM, INTDEM, or RAZE.")
        return None

    # Convert dates and extract demolition year
    demolition_permits['issued_date'] = pd.to_datetime(demolition_permits['issued_date'], errors='coerce')
    demolition_permits.dropna(subset=['issued_date'], inplace=True)
    demolition_permits['demolition_year'] = demolition_permits['issued_date'].dt.year

    # Get the single, most recent demolition permit for each parcel to avoid double counting
    final_demolitions = demolition_permits.loc[
        demolition_permits.groupby('parcel_id')['demolition_year'].idxmax()
    ]

    # --- 4. Merge with build years and calculate lifespan ---
    print("Merging data and calculating lifespan...")
    lifespan_df = pd.merge(
        final_demolitions,
        build_years,
        left_on='parcel_id',
        right_on='PID',
        how='inner'
    )

    # Calculate lifespan and clean data (lifespan > 0)
    lifespan_df['lifespan'] = lifespan_df['demolition_year'] - lifespan_df['build_year']

    # Report records removed during cleaning
    initial_record_count = len(lifespan_df)
    lifespan_df = lifespan_df[lifespan_df['lifespan'] > 0]
    final_record_count = len(lifespan_df)
    records_removed = initial_record_count - final_record_count

    print(f"\nTotal matched records before cleaning: {initial_record_count}")
    print(f"Records with lifespan < 1 year removed: {records_removed}")
    print(f"Final valid records: {final_record_count}")

    # **NEW**: Clean up coordinate data only if columns exist
    has_geo_data = 'y_latitude' in lifespan_df.columns and 'x_longitude' in lifespan_df.columns
    if has_geo_data:
        lifespan_df.dropna(subset=['y_latitude', 'x_longitude'], inplace=True)
        # Ensure coordinates are valid for Boston area
        lifespan_df = lifespan_df[lifespan_df['y_latitude'].between(42, 43)]
        lifespan_df = lifespan_df[lifespan_df['x_longitude'].between(-72, -70)]
        print(f"Final valid records with coordinates: {len(lifespan_df)}")

    if lifespan_df.empty:
        print("No valid records after merging and cleaning.")
        return None

    # --- 5. Generate all data structures for JSON output ---
    all_data = {}

    # Calculate demolition type counts from the final, cleaned data
    type_counts = lifespan_df['worktype'].value_counts().to_dict()

    # Calculate summary statistics
    all_data['summary_stats'] = {
        'total_demolitions': final_record_count,
        'average_lifespan': float(lifespan_df['lifespan'].mean()),
        'median_lifespan': float(lifespan_df['lifespan'].median()),
        'min_lifespan': int(lifespan_df['lifespan'].min()),
        'max_lifespan': int(lifespan_df['lifespan'].max()),
        'extdem_count': type_counts.get('EXTDEM', 0),
        'intdem_count': type_counts.get('INTDEM', 0),
        'raze_count': type_counts.get('RAZE', 0)
    }

    # Create yearly demolitions data (for stacked bar chart)
    min_year = int(lifespan_df['demolition_year'].min())
    max_year = int(lifespan_df['demolition_year'].max())
    years = list(range(min_year, max_year + 1))

    yearly_data = []
    for year in years:
        year_data = {'year': year}
        for demo_type in demolition_types:
            count = len(lifespan_df[
                            (lifespan_df['demolition_year'] == year) &
                            (lifespan_df['worktype'] == demo_type)
                            ])
            year_data[demo_type] = count
        yearly_data.append(year_data)

    all_data['yearly_stacked'] = yearly_data

    # Create lifespan distribution (10-year bins)
    bins = list(range(0, int(lifespan_df['lifespan'].max()) + 11, 10))
    lifespan_bins = []
    for i in range(len(bins) - 1):
        bin_label = f"{bins[i]}-{bins[i + 1]}"
        bin_data = {'range': bin_label}
        for demo_type in demolition_types:
            count = len(lifespan_df[
                            (lifespan_df['worktype'] == demo_type) &
                            (lifespan_df['lifespan'] >= bins[i]) &
                            (lifespan_df['lifespan'] < bins[i + 1])
                            ])
            bin_data[demo_type] = count
        if any(bin_data[dt] > 0 for dt in demolition_types):
            lifespan_bins.append(bin_data)
    all_data['lifespan_distribution'] = lifespan_bins

    # Create lifespan distribution (5-year bins)
    bins_5 = list(range(0, int(lifespan_df['lifespan'].max()) + 6, 5))
    lifespan_bins_5 = []
    for i in range(len(bins_5) - 1):
        bin_label = f"{bins_5[i]}-{bins_5[i + 1]}"
        bin_data = {'range': bin_label}
        for demo_type in demolition_types:
            count = len(lifespan_df[
                            (lifespan_df['worktype'] == demo_type) &
                            (lifespan_df['lifespan'] >= bins_5[i]) &
                            (lifespan_df['lifespan'] < bins_5[i + 1])
                            ])
            bin_data[demo_type] = count
        if any(bin_data[dt] > 0 for dt in demolition_types):
            lifespan_bins_5.append(bin_data)
    all_data['lifespan_distribution_5yr'] = lifespan_bins_5

    # Demolition type pie chart data
    all_data['demolition_types'] = [
        {'type': demo_type, 'count': type_counts.get(demo_type, 0)}
        for demo_type in demolition_types
    ]

    # Average lifespan by type
    lifespan_by_type_df = lifespan_df.groupby('worktype')['lifespan'].agg(['mean', 'median', 'count']).reset_index()
    all_data['lifespan_by_type'] = [
        {
            'type': row['worktype'],
            'average': float(row['mean']),
            'median': float(row['median']),
            'count': int(row['count'])
        }
        for index, row in lifespan_by_type_df.iterrows()
    ]

    # --- 6. Generate data for Lifespan by Year Box Plot ---
    print("Generating data for lifespan by year box plot...")
    lifespan_by_year_boxplot = []
    # Ensure demolition_year is integer for consistent sorting and JSON serialization
    lifespan_df['demolition_year'] = lifespan_df['demolition_year'].astype(int)
    for year in sorted(lifespan_df['demolition_year'].unique()):
        lifespans_for_year = lifespan_df[lifespan_df['demolition_year'] == year]['lifespan'].tolist()
        if lifespans_for_year:
            lifespan_by_year_boxplot.append({
                'year': int(year),
                'lifespans': lifespans_for_year
            })
    all_data['lifespan_by_year_boxplot'] = lifespan_by_year_boxplot

    # --- 7. **NEW**: Generate data for the map plot ---
    if has_geo_data:
        print("Generating data for map plot...")
        map_df = lifespan_df[['y_latitude', 'x_longitude', 'worktype', 'lifespan']].copy()
        map_df.rename(columns={
            'y_latitude': 'lat',
            'x_longitude': 'lng',
            'worktype': 'type'
        }, inplace=True)
        all_data['map_points'] = map_df.to_dict(orient='records')
    else:
        all_data['map_points'] = []  # Add empty list if no geo data

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
    """
    Save processed data to JSON files
    """
    if not data:
        print("No data to save")
        return

    # Save main data file
    main_file = f"{output_prefix}_data.json"
    with open(main_file, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\nSaved main data to {main_file}")


def main():
    """
    Main function to process data and generate JSON files
    """
    # File paths - update these to match your file names
    assessment_file = 'fy2025-property-assessment-data_12_30_2024.csv'
    permit_file = 'tmpbtz4x7bc.csv'

    print("Starting Boston Demolition Data Processing...")
    print("=" * 50)

    # Process the data
    processed_data = process_demolition_data(assessment_file, permit_file)

    if processed_data:
        # Save to JSON files
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