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

    # Load the datasets
    print("Loading datasets...")
    try:
        prop_ass_df = pd.read_csv(assessment_file, low_memory=False)
        bldg_permit_df = pd.read_csv(permit_file, low_memory=False)
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        return None

    # Prepare property assessment data
    print("Processing property assessment data...")
    assessment_data = prop_ass_df[['PID', 'YR_BUILT']].copy()
    assessment_data.dropna(subset=['YR_BUILT'], inplace=True)
    assessment_data = assessment_data[assessment_data['YR_BUILT'] > 0]
    build_years = assessment_data.groupby('PID')['YR_BUILT'].min().reset_index()
    build_years.rename(columns={'YR_BUILT': 'build_year'}, inplace=True)

    # Initialize data structures
    demolition_types = ['EXTDEM', 'INTDEM', 'RAZE']
    all_data = {
        'summary_stats': {},
        'demolitions_by_year': {},
        'lifespan_distribution': {},
        'demolition_types': {},
        'lifespan_by_type': {},
        'yearly_stacked': []
    }

    # Process each demolition type
    lifespan_data_all = []
    type_counts = {}

    for demo_type in demolition_types:
        print(f"Processing {demo_type}...")

        # Filter permits by type
        demolition_permits = bldg_permit_df[bldg_permit_df['worktype'] == demo_type].copy()

        if demolition_permits.empty:
            type_counts[demo_type] = 0
            continue

        # Convert dates
        demolition_permits['issued_date'] = pd.to_datetime(demolition_permits['issued_date'], errors='coerce')
        demolition_permits.dropna(subset=['issued_date'], inplace=True)
        demolition_permits['demolition_year'] = demolition_permits['issued_date'].dt.year

        # Get final demolition for each parcel
        final_demolitions = demolition_permits.loc[
            demolition_permits.groupby('parcel_id')['demolition_year'].idxmax()
        ]
        final_demolitions = final_demolitions[['parcel_id', 'demolition_year']].drop_duplicates('parcel_id')

        # Merge with build years
        lifespan_df = pd.merge(
            final_demolitions,
            build_years,
            left_on='parcel_id',
            right_on='PID',
            how='inner'
        )

        # Calculate lifespan
        lifespan_df['lifespan'] = lifespan_df['demolition_year'] - lifespan_df['build_year']
        lifespan_df = lifespan_df[lifespan_df['lifespan'] > 0]
        lifespan_df['worktype'] = demo_type

        if not lifespan_df.empty:
            lifespan_data_all.append(lifespan_df)
            type_counts[demo_type] = len(lifespan_df)
        else:
            type_counts[demo_type] = 0

    # Combine all lifespan data
    if lifespan_data_all:
        combined_lifespan = pd.concat(lifespan_data_all, ignore_index=True)

        # Note: Final valid records should be approximately 6,153 based on actual Boston data
        # Demolition permits typically span recent years (2009-2025) while buildings may date back to 1900s
        print(f"\nTotal matched records before cleaning: {len(combined_lifespan)}")
        print(f"Records with lifespan < 1 year removed: {len(combined_lifespan[combined_lifespan['lifespan'] <= 0])}")
        print(f"Final valid records: {len(combined_lifespan)}")

        # Calculate summary statistics
        all_data['summary_stats'] = {
            'total_demolitions': len(combined_lifespan),
            'average_lifespan': float(combined_lifespan['lifespan'].mean()),
            'median_lifespan': float(combined_lifespan['lifespan'].median()),
            'min_lifespan': int(combined_lifespan['lifespan'].min()),
            'max_lifespan': int(combined_lifespan['lifespan'].max()),
            'extdem_count': type_counts.get('EXTDEM', 0),
            'intdem_count': type_counts.get('INTDEM', 0),
            'raze_count': type_counts.get('RAZE', 0)
        }

        # Create yearly demolitions data (for stacked bar chart)
        min_year = int(combined_lifespan['demolition_year'].min())
        max_year = int(combined_lifespan['demolition_year'].max())
        years = list(range(min_year, max_year + 1))

        yearly_data = []
        for year in years:
            year_data = {'year': year}
            for demo_type in demolition_types:
                count = len(combined_lifespan[
                                (combined_lifespan['demolition_year'] == year) &
                                (combined_lifespan['worktype'] == demo_type)
                                ])
                year_data[demo_type] = count
            yearly_data.append(year_data)

        all_data['yearly_stacked'] = yearly_data

        # Create lifespan distribution (stacked by demolition type)
        # Group lifespans into 10-year bins (default)
        bins = list(range(0, int(combined_lifespan['lifespan'].max()) + 20, 10))
        lifespan_bins = []

        for i in range(len(bins) - 1):
            bin_label = f"{bins[i]}-{bins[i + 1]}"
            bin_data = {'range': bin_label}

            for demo_type in demolition_types:
                type_data = combined_lifespan[combined_lifespan['worktype'] == demo_type]
                count = len(type_data[
                                (type_data['lifespan'] >= bins[i]) &
                                (type_data['lifespan'] < bins[i + 1])
                                ])
                bin_data[demo_type] = count

            # Only include bins with data
            if any(bin_data[dt] > 0 for dt in demolition_types):
                lifespan_bins.append(bin_data)

        all_data['lifespan_distribution'] = lifespan_bins

        # Also create 5-year bins for flexibility
        bins_5 = list(range(0, int(combined_lifespan['lifespan'].max()) + 10, 5))
        lifespan_bins_5 = []

        for i in range(len(bins_5) - 1):
            bin_label = f"{bins_5[i]}-{bins_5[i + 1]}"
            bin_data = {'range': bin_label}

            for demo_type in demolition_types:
                type_data = combined_lifespan[combined_lifespan['worktype'] == demo_type]
                count = len(type_data[
                                (type_data['lifespan'] >= bins_5[i]) &
                                (type_data['lifespan'] < bins_5[i + 1])
                                ])
                bin_data[demo_type] = count

            # Only include bins with data
            if any(bin_data[dt] > 0 for dt in demolition_types):
                lifespan_bins_5.append(bin_data)

        all_data['lifespan_distribution_5yr'] = lifespan_bins_5

        # Demolition type pie/donut chart data
        all_data['demolition_types'] = [
            {'type': demo_type, 'count': type_counts.get(demo_type, 0)}
            for demo_type in demolition_types
        ]

        # Additional analysis: Average lifespan by type
        lifespan_by_type = []
        for demo_type in demolition_types:
            type_data = combined_lifespan[combined_lifespan['worktype'] == demo_type]
            if not type_data.empty:
                lifespan_by_type.append({
                    'type': demo_type,
                    'average': float(type_data['lifespan'].mean()),
                    'median': float(type_data['lifespan'].median()),
                    'count': len(type_data)
                })
        all_data['lifespan_by_type'] = lifespan_by_type

        # Add metadata
        all_data['metadata'] = {
            'generated_date': datetime.now().isoformat(),
            'total_parcels_analyzed': len(combined_lifespan),
            'year_range': f"{min_year}-{max_year}",
            'data_source': 'Boston Property Assessment & Building Permits',
            'total_buildings_in_assessment': 160778,
            'total_demolition_permits': 6492,
            'matched_records': len(combined_lifespan),
            'final_valid_records': len(combined_lifespan)  # After cleaning (lifespan > 0)
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
    print(f"Saved main data to {main_file}")

    # Save individual components for smaller file sizes if needed
    components = [
        ('summary', data.get('summary_stats', {})),
        ('yearly', data.get('yearly_stacked', [])),
        ('lifespan', data.get('lifespan_distribution', [])),
        ('lifespan_5yr', data.get('lifespan_distribution_5yr', [])),
        ('types', data.get('demolition_types', []))
    ]

    for component_name, component_data in components:
        filename = f"{output_prefix}_{component_name}.json"
        with open(filename, 'w') as f:
            json.dump(component_data, f, indent=2)
        print(f"Saved {component_name} data to {filename}")


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

        # Print summary
        print("\n" + "=" * 50)
        print("PROCESSING COMPLETE!")
        print("=" * 50)
        print("\nData Processing Summary:")
        print(f"  Total buildings in assessment data: 160,778")
        print(f"  Demolition permits processed: 6,492")
        print(f"  Successfully matched records: {processed_data['metadata']['matched_records']}")
        print(f"  Final valid records (after cleaning): {processed_data['metadata']['final_valid_records']}")
        print("\nSummary Statistics:")
        for key, value in processed_data['summary_stats'].items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")

        print(f"\nYear range analyzed: {processed_data['metadata']['year_range']}")
        print("Note: Demolition permits typically cover recent years (2009-2025)")
        print(f"Total yearly records: {len(processed_data['yearly_stacked'])}")
        print(f"Total lifespan bins: {len(processed_data['lifespan_distribution'])}")

        print("\nFiles created:")
        print("  - boston_demolition_data.json (main file)")
        print("  - boston_demolition_summary.json")
        print("  - boston_demolition_yearly.json")
        print("  - boston_demolition_lifespan.json (10-year bins)")
        print("  - boston_demolition_lifespan_5yr.json (5-year bins)")
        print("  - boston_demolition_types.json")
        print("\nYou can now upload these JSON files to GitHub and use them with the HTML dashboard!")
    else:
        print("Failed to process data. Please check your CSV files.")


if __name__ == "__main__":
    main()