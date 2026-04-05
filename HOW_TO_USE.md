# How to Use - Learning Analytics Platform

## Starting the Application

```bash
cd learning_analytics_platform
pip install -r requirements.txt
python app/main.py
```

Open **http://localhost:8050** in your browser.

---

## Step-by-Step Workflow

### Step 1: Upload Data (Upload Data page)

1. Click **"Upload Data"** in the left sidebar
2. **Option A - Load Sample Data:** Click the **"Load Sample Data"** button to load pre-generated datasets (Coursera activity, LMS sessions, Academic records)
3. **Option B - Upload Your Own:**
   - Select the **Source Type** from the dropdown (Coursera, LMS, Academic Records, or Custom)
   - Drag and drop your CSV or Excel file onto the upload zone, or click to browse
   - Supported formats: `.csv`, `.xlsx`
4. The **Uploaded Datasets** table below shows all loaded files with row/column counts

### Step 2: Clean & Process Data (Data Management page)

1. Click **"Data Management"** in the sidebar

2. **Run Cleaning:**
   - Select a missing value strategy (Mean, Median, Drop, or Zero)
   - Set the missing threshold (columns with more missing values than this % will be dropped)
   - Select duplicate handling (Keep First, Keep Last, or Remove All)
   - Click **"Run Cleaning"**
   - A green alert confirms how many rows were removed and missing values handled

3. **Run Integration:**
   - Select the Join Key (default: `student_id`)
   - Select Conflict Resolution strategy (Latest or Source Priority)
   - Click **"Run Integration"**
   - The system automatically loads the student mapping table and merges all sources

4. **Generate Features:**
   - Check the features you want: Engagement Score, Session Frequency, Video Completion, Assessment Improvement
   - Click **"Generate Features"**
   - New feature columns are added to the dataset

5. Use the **Raw Data / Cleaned Data / Integrated Data** tabs to preview data at each stage
6. The quality KPI cards show: Total Rows, Missing Values, Duplicates, and Columns

### Step 3: Run Analytics (Analytics page)

1. Click **"Analytics"** in the sidebar

2. **Exploratory Data Analysis (EDA):**
   - Select correlation method (Pearson, Spearman, or Kendall)
   - Click **"Run EDA"**
   - View distribution statistics table and correlation heatmap

3. **Hypothesis Testing:**
   - Select test type: T-Test or ANOVA
   - Select the target column (numeric variable to test)
   - Select the group column (categorical variable to split by)
   - Set significance level (default 0.05)
   - Click **"Run Test"**
   - View test statistic, p-value, and significance result

4. **Machine Learning:**
   - Select task: Regression, Classification, or Clustering
   - For Regression/Classification:
     - Select the target column
     - Select model type (Linear, Ridge, Random Forest, etc.)
     - Set test/train split size
     - Click **"Train Model"**
   - For Clustering:
     - Select model (K-Means, DBSCAN, Hierarchical)
     - Set number of clusters
     - Click **"Train Model"**
   - View metrics (R2/RMSE for regression, Accuracy/F1 for classification, Silhouette for clustering)
   - Feature importance charts are shown when available

### Step 4: Explore Visualizations (Visualization page)

1. Click **"Visualization"** in the sidebar
2. Four tabs are available:
   - **Distribution:** Select a column and number of bins to see its histogram
   - **Correlation:** Select method and columns to see the correlation heatmap
   - **Clustering:** Select X/Y axes to see cluster scatter plot (after running clustering in Analytics)
   - **Time Series:** Select time column and value columns for trend lines

### Step 5: Generate Reports (Reports page)

1. Click **"Reports"** in the sidebar
2. **Select report sections** to include:
   - Summary Statistics
   - Distribution Analysis
   - Correlation Analysis
   - Hypothesis Test Results
   - ML Model Results
3. Select **export format**: PDF or CSV
4. Set decimal places for rounding
5. Click **"Generate Report"** to download
6. The **Report Preview** panel shows what will be included
7. Use the **Export Data** buttons at the bottom to download:
   - Integrated Dataset (CSV)
   - Feature Dataset (CSV)
   - Summary Statistics (CSV)

### Dashboard (Home page)

The home page shows:
- **KPI Cards:** Total Students, Datasets Loaded, Avg Engagement Score, Dropout Rate
- **Pipeline Status:** Green dots indicate completed steps
- **Quick Actions:** Buttons to jump to key pages
- **Activity Log:** Shows pipeline progress

---

## Important Notes

- **Data must flow in order:** Upload -> Clean -> Integrate -> Features -> Analytics -> Reports
- Each step depends on the previous step being completed
- The pipeline status on the Dashboard tracks your progress
- All data is stored in browser memory (refreshing the page clears data)
- For production deployment, use: `waitress-serve --port=8050 app.main:server`

## Configuration

Edit `config/pipeline_config.yaml` to change:
- Schema validation rules per source
- Default cleaning strategies
- Feature engineering weights
- ML model parameters
- Visualization themes
- Report formatting options

## Generating Sample Data

```bash
python data/sample/generate_sample_data.py
```

This creates 4 files in `data/sample/`:
- `coursera_activity.csv` (17,697 records)
- `lms_sessions.csv` (14,511 records)
- `academic_records.csv` (1,606 records)
- `student_mapping.csv` (200 student ID mappings)
