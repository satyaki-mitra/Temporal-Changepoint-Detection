# PHQ-9 Exploratory Data Analysis Module

## Overview

Comprehensive exploratory data analysis for PHQ-9 temporal data with advanced clustering and visualization capabilities.

### Key Features

✅ **Multiple Clustering Algorithms** - KMeans, Agglomerative, Temporal-aware  
✅ **Optimal Cluster Selection** - Elbow, Silhouette, Gap Statistic  
✅ **Proper Missing Data Handling** - Mean/median imputation (NO -1 filling!)  
✅ **Rich Visualizations** - Scatter plots, cluster boundaries, temporal trends  
✅ **Statistical Analysis** - Summary statistics, cluster characteristics  

---

## Quick Start

### Basic Usage

```python
from config.eda_config import EDAConfig
from phq9_analysis.eda import PHQ9DataAnalyzer

# Create configuration
config = EDAConfig()

# Run full analysis
analyzer = PHQ9DataAnalyzer(config)
analyzer.load_data("data/raw/synthetic_phq9_data.csv")
results = analyzer.run_full_analysis()

# Results contain:
# - Optimal cluster count
# - Cluster labels and characteristics
# - All visualizations saved
```

### Command-Line Usage

```bash
# Full analysis with defaults
python scripts/run_eda.py

# Custom data file
python scripts/run_eda.py --data path/to/data.csv

# Test more clusters
python scripts/run_eda.py --max-clusters 20

# Use specific number of clusters (skip optimization)
python scripts/run_eda.py --n-clusters 5

# Temporal-aware clustering
python scripts/run_eda.py --temporal

# Custom output directory
python scripts/run_eda.py --output-dir results/my_analysis
```

---

## Clustering Methods

### 1. KMeans Clustering (Default)

Standard KMeans with proper imputation:

```python
from phq9_analysis.eda import ClusteringEngine

engine = ClusteringEngine(imputation_method='mean')
labels, inertia, silhouette = engine.fit_kmeans(data, n_clusters=3)
```

**When to use:**
- Fast and scalable
- Works well for spherical clusters
- Good for initial exploration

### 2. Agglomerative Clustering

Hierarchical clustering with linkage options:

```python
labels, silhouette = engine.fit_agglomerative(
    data,
    n_clusters=3,
    linkage='ward'  # 'ward', 'complete', 'average'
)
```

**When to use:**
- Need cluster hierarchy
- Non-spherical clusters
- Small datasets (<1000 days)

### 3. Temporal-Aware Clustering

Penalizes clustering distant days together:

```python
from phq9_analysis.eda import TemporalClustering

temporal = TemporalClustering(temporal_weight=0.3)
labels = temporal.fit(data, n_clusters=3)
```

**When to use:**
- Want temporally contiguous clusters
- Studying treatment phases
- Detecting regime changes

**Parameters:**
- `temporal_weight=0.0`: Pure score-based (ignore time)
- `temporal_weight=0.3`: Balanced (recommended)
- `temporal_weight=1.0`: Pure temporal (only proximity)

---

## Optimal Cluster Selection

### Elbow Method

Finds "elbow" in inertia curve using angle detection:

```python
from phq9_analysis.eda import OptimalClusterSelector

selector = OptimalClusterSelector(engine)
elbow_k, inertias = selector.elbow_method(data, max_clusters=15)
print(f"Elbow suggests: {elbow_k} clusters")
```

**Note:** Uses proper angle-based detection, NOT minimum percentage change!

### Silhouette Method

Maximizes average silhouette score:

```python
silhouette_k, silhouettes = selector.silhouette_method(data, max_clusters=15)
print(f"Silhouette suggests: {silhouette_k} clusters")
```

**Interpretation:**
- Score > 0.7: Strong clustering
- Score 0.5-0.7: Reasonable clustering
- Score < 0.5: Weak clustering

### Gap Statistic

Compares clustering to random uniform data:

```python
gap_k, gaps = selector.gap_statistic(data, max_clusters=10, n_refs=10)
print(f"Gap statistic suggests: {gap_k} clusters")
```

**When to use:** More rigorous but computationally expensive

---

## Missing Data Handling

### ⚠️ CRITICAL: Never Use fillna(-1)!

**WRONG:**
```python
# ❌ BAD - Distorts distance calculations
data_filled = data.fillna(-1)
kmeans.fit(data_filled)
```

**CORRECT:**
```python
# ✅ GOOD - Proper imputation
engine = ClusteringEngine(imputation_method='mean')
labels = engine.fit_kmeans(data, n_clusters=3)
```

### Imputation Methods

1. **Mean Imputation** (Default)
   - Fills missing with column mean
   - Best for: Random missingness (MCAR)

2. **Median Imputation**
   - Fills missing with column median
   - Best for: Outlier-prone data

3. **Forward Fill**
   - Uses last observation carried forward
   - Best for: Temporally ordered data

```python
# Configure imputation method
config = EDAConfig(imputation_method='median')
analyzer = PHQ9DataAnalyzer(config)
```

---

## Visualizations

### 1. Scatter Plot

Shows all individual PHQ-9 scores across days:

```python
from phq9_analysis.eda import plot_scatter

plot_scatter(data, save_path="scatter.png")
```

**Use for:** Data distribution, outlier detection

### 2. Daily Averages

Plots mean PHQ-9 score per day with trend line:

```python
from phq9_analysis.eda import plot_daily_averages

plot_daily_averages(data, save_path="daily_avg.png")
```

**Shows:**
- Overall treatment trajectory
- Severity level thresholds
- Linear trend (slope)

### 3. Cluster Results

Visualizes clustering with colored segments:

```python
from phq9_analysis.eda import plot_clusters

plot_clusters(data, labels, n_clusters=3, save_path="clusters.png")
```

**Shows:**
- Cluster membership per day
- Cluster boundaries
- Within-cluster patterns

### 4. Cluster Optimization

Elbow and silhouette plots side-by-side:

```python
visualizer.plot_cluster_optimization(
    inertias=[...],
    silhouettes=[...],
    elbow_k=4,
    silhouette_k=3,
    save_path="optimization.png"
)
```

**Use for:** Selecting optimal K

---

## Output Files

### Generated by Full Analysis

```
results/eda/
├── summary_statistics.csv           # Descriptive statistics per day
├── cluster_characteristics.csv      # Cluster properties
├── analysis_summary.json            # Overall results
├── clustering/
│   └── cluster_optimization.png     # Elbow + silhouette plots
└── visualizations/
    ├── scatter_plot.png             # All scores
    ├── daily_averages.png           # Temporal trend
    └── cluster_results.png          # Clustering results
```

### Summary Statistics CSV

```csv
,count,mean,std,min,25%,50%,75%,max,missing_count,missing_pct
Day_1,85.0,16.2,3.8,8.1,13.5,16.0,18.9,24.3,15,15.0
Day_7,82.0,15.8,3.9,7.2,13.1,15.6,18.4,23.8,18,18.0
...
```

### Cluster Characteristics CSV

```csv
cluster_id,n_days,avg_score,std_score,min_score,max_score,day_range,severity
0,15,17.2,1.3,15.1,19.8,0-14,Moderately Severe
1,20,11.5,1.8,8.3,14.9,15-34,Moderate
2,15,6.3,1.5,3.2,9.1,35-49,Mild
```

---

## Configuration

### Full Configuration Example

```python
from config.eda_config import EDAConfig

config = EDAConfig(
    # Data
    data_file_path=Path("data/raw/synthetic_phq9_data.csv"),
    min_observations_per_patient=3,
    
    # Clustering
    max_clusters_to_test=20,
    clustering_algorithm='kmeans',  # 'kmeans', 'agglomerative', 'both'
    imputation_method='mean',  # 'mean', 'median', 'forward_fill'
    use_temporal_clustering=False,
    temporal_weight=0.3,
    
    # Visualization
    figure_size=(15, 12),
    dpi=300,
    plot_style='seaborn',
    color_palette='husl',
    
    # Output
    results_base_directory=Path("results/eda"),
    save_intermediate_results=True,
)
```

---

## Advanced Usage

### Custom Clustering Pipeline

```python
from phq9_analysis.eda import ClusteringEngine, OptimalClusterSelector

# Initialize
engine = ClusteringEngine(imputation_method='median', standardize=False)
selector = OptimalClusterSelector(engine)

# Find optimal K
elbow_k, _ = selector.elbow_method(data, max_clusters=15)
silhouette_k, _ = selector.silhouette_method(data, max_clusters=15)
gap_k, _ = selector.gap_statistic(data, max_clusters=10)

print(f"Elbow: {elbow_k}, Silhouette: {silhouette_k}, Gap: {gap_k}")

# Use consensus
optimal_k = int(np.median([elbow_k, silhouette_k, gap_k]))

# Fit final model
labels, inertia, silhouette = engine.fit_kmeans(data, optimal_k)
```

### Comparing Multiple Methods

```python
methods = {
    'KMeans': lambda k: engine.fit_kmeans(data, k),
    'Agglomerative': lambda k: engine.fit_agglomerative(data, k),
    'Temporal': lambda k: temporal.fit(data, k)
}

results = {}
for name, method in methods.items():
    labels = method(n_clusters=3)
    # Analyze and compare...
```

---

## Troubleshooting

### Issue: "Scores outside valid range"

**Cause:** Data contains values < 0 or > 27  
**Solution:** Check data source, clean outliers

### Issue: "Could not calculate silhouette"

**Cause:** Too few observations after imputation  
**Solution:** 
- Reduce `min_observations_per_patient`
- Check data quality
- Use different imputation method

### Issue: "All clusters same size"

**Cause:** Data doesn't have clear structure  
**Solution:**
- Try different K values
- Check if temporal clustering helps
- Verify data has temporal patterns

### Issue: Elbow not clear

**Cause:** Gradual inertia decrease (no sharp elbow)  
**Solution:**
- Use silhouette or gap statistic
- Try K in range suggested by different methods
- Consider domain knowledge (e.g., treatment phases)

---

## Testing

Run module tests:

```bash
# Test clustering
python -m phq9_analysis.eda.clustering

# Test visualizations
python -m phq9_analysis.eda.visualizations

# Test analyzer
python -c "from phq9_analysis.eda import PHQ9DataAnalyzer; \
           from config.eda_config import EDAConfig; \
           config = EDAConfig(); \
           analyzer = PHQ9DataAnalyzer(config); \
           print('✅ EDA module OK')"
```

---

## References

1. **Cluster Analysis**  
   Kaufman, L., & Rousseeuw, P. J. (1990). *Finding groups in data: an introduction to cluster analysis.*

2. **Silhouette Method**  
   Rousseeuw, P. J. (1987). *Silhouettes: a graphical aid to the interpretation and validation of cluster analysis.*

3. **Gap Statistic**  
   Tibshirani, R., Walther, G., & Hastie, T. (2001). *Estimating the number of clusters in a data set via the gap statistic.*

---

## License

MIT License - See project root LICENSE file

---

**Last Updated:** January 2025