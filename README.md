# Boston Building Demolition Analysis Dashboard

An interactive web-based visualization dashboard for analyzing building demolition patterns in Boston, combining property assessment data with building permit records to reveal urban development trends and building lifecycles.

## Live Demo

[View Live Dashboard](https://samueeelsiu.github.io/Lifespan-Boston/)


## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Data Pipeline](#data-pipeline)
- [Installation](#installation)
- [Usage Guide](#usage-guide)
- [Data Structure](#data-structure)
- [Methodology](#methodology)
- [Technologies](#technologies)
- [Support](#support)

## Overview

This dashboard visualizes and analyzes building demolition patterns in Boston, integrating data from:
- **Boston Property Assessment Dataset**: Current building inventory and construction years
- **Boston Approved Building Permits**: Historical demolition records from 2009-2025
- **Permit Types Analyzed**: RAZE (full demolition), EXTDEM (exterior demolition), INTDEM (interior demolition)

## Key Features

### Interactive Visualizations

1. **Demolition Locations Map**
   - Interactive Leaflet map with demolition points
   - Color-coded by demolition type
   - Hover tooltips showing building lifespan
   - Filter by permit status (Open/Close)

2. **Multi-RAZE Parcels Analysis**
   - Dot plot visualization of buildings with multiple RAZE permits
   - Build year indicators (green squares)
   - Latest RAZE events (red triangles)
   - Historical RAZE permits (circles)

3. **Temporal Analysis**
   - Demolitions by year (2009-2025)
   - Multiple chart types (Bar/Area/Line)
   - Absolute and percentage view modes
   - Stacked visualization for permit types

4. **Age Distribution Analysis**
   - Building age at demolition
   - Current building age distribution
   - Configurable bin sizes (5/10/20 years)
   - Comparative analysis

5. **Construction Era Tracking**
   - Pre-1900 through 2020+ era analysis
   - Year-by-year demolition patterns
   - Interactive legend for data isolation

6. **Statistical Summaries**
   - RAZE lifespan summary with status badges
   - Average building lifespan calculations
   - Permit status breakdown (Open/Close)
   - Key performance indicators

## Data Pipeline

### Processing Methodology

```
Stage 1: Building Identification
├── Input: Property Assessment + Building Permits
├── Process: Unified building_id creation
│   ├── Condominiums: Use CM_ID for entire building
│   └── Other properties: Use PID
└── Output: Building-level analysis

Stage 2: Duplicate Handling
├── Input: Raw permit records
├── Process: 24-hour window consolidation
│   ├── Priority: Closed status permits
│   └── Remove administrative duplicates
└── Output: De-duplicated permit set

Stage 3: Lifespan Calculation
├── Input: Matched building-permit pairs
├── Process: Demolition year - Build year
│   ├── Positive lifespan: Complete lifecycle
│   └── Non-positive: Demolished and replaced
└── Output: Categorized demolition events
```

## Installation

### Prerequisites

- Modern web browser (Chrome, Firefox, Safari, Edge)
- Local web server (for local deployment)
- Python 3.x (for data preprocessing)

### Setup

1. **Clone or Download Files**
```bash
# Required files:
├── index.html                        # Main dashboard
├── boston_demolition_data.json       # Processed demolition data
├── data_preprocess.py               # Data processing script
└── README.md                        # This file
```

2. **Data Files Required (for reprocessing)**
```bash
# Input CSV files needed:
├── fy2025-property-assessment-data_*.csv   # Property assessment
└── tmpbtz4x7bc.csv                         # Building permits
```

3. **Launch Dashboard**

Option A: Direct website opening
```bash
# Open https://samueeelsiu.github.io/Lifespan-Boston/ in your browser
```

Option B: Local server (recommended)
```bash
# Python 3
python -m http.server 8000

# Then navigate to http://localhost:8000
```

## Usage Guide

### Controls

- **Status Filter**: Checkbox to show only 'Closed' status permits
- **Demolition Type**: Filter map by RAZE/EXTDEM/INTDEM/All
- **Chart Type**: Switch between Bar/Area/Line visualizations
- **View Mode**: Toggle between absolute numbers and percentages
- **Bin Size**: Adjust age distribution bins (5/10/20 years)

### Key Interactions

- **Map**: Click and drag to pan, scroll to zoom
- **Charts**: Hover for detailed tooltips
- **Legends**: Click to isolate/show datasets
- **Multi-RAZE Plot**: View permit history per building

## Data Structure

### JSON Schema

```javascript
{
  "metadata": {
    "generated_date": ...,
    "total_parcels_analyzed": ...,
    "year_range": "2009-2025",
    "data_source": "Boston Property Assessment & Approved Building Permits"
  },
  "summary_stats": {
    "total_demolitions": ...,
    "average_lifespan": 91.3,
    "raze_count": 357,
    "extdem_count": 1,022,
    "intdem_count": 6,395,
    "raze_status_by_lifespan": {...}
  },
  "summary_stats_closed": {...},      // Filtered for closed permits
  "yearly_stacked": [...],           // Annual demolition data
  "lifespan_distribution": [...],     // Age bins
  "map_points": [...],               // Geospatial data
  "multi_raze_parcels": [...]        // Multiple RAZE analysis
}
```

## Methodology

### Key Concepts

1. **Building vs. Unit Identification**
   - Condominiums use CM_ID to represent entire buildings
   - Prevents double-counting of multi-unit demolitions

2. **Permit De-duplication**
   - 24-hour window consolidation per parcel+type
   - Prioritizes 'Closed' status permits
   - Maintains data integrity

3. **Lifespan Interpretation**
   - **Positive Lifespan**: Observable complete lifecycle
   - **Non-positive (≤0)**: Urban renewal indicator
   - "Demolished and Replaced" category for redevelopment

4. **Status Normalization**
   - Closed: CLOSED, CLOSE, CLOSED OUT, COMPLETE
   - Open: OPEN, ACTIVE, ISSUED, PENDING

## Technologies

### Frontend
- **HTML5/CSS3**: Responsive design
- **JavaScript**: Dynamic interactions
- **Chart.js 4.4.0**: Data visualizations
- **Leaflet 1.9.4**: Interactive mapping

### Backend Processing
- **Python 3.x**
  - pandas: Data manipulation
  - numpy: Numerical operations
  - datetime: Temporal processing

### Data Formats
- **JSON**: Primary data format
- **CSV**: Source data files

## Performance Notes

- Handles 7000+ demolition records efficiently
- Map points filtered dynamically
- Pre-aggregated statistics for responsiveness
- Optimized for modern browsers

## Known Limitations

- Permit data available from 2009 onwards
- Some buildings lack coordinate data
- Status information may be incomplete for older permits

## Credit

### Development Team
- **Developer**: Lang (Samuel) Shao
- **Supervisor**: Prof. Demi Fang
- **Institution**: [Northeastern University](https://www.northeastern.edu/)
- **Lab**: [Structural Futures Lab](https://structural-futures.org/)

## Support

For issues, questions, or suggestions regarding this dashboard, please contact: shao.la@northeastern.edu
