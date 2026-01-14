# Dependencies
import numpy as np
import pandas as pd
from typing import List
from typing import Tuple
from typing import Literal
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from config.eda_constants import eda_constants_instance
from sklearn.metrics.pairwise import euclidean_distances



class ClusteringEngine:
    """
    Main clustering engine with multiple algorithms and validation
    """
    def __init__(self, imputation_method: Literal['mean', 'median', 'forward_fill'] = 'mean', standardize: bool = False, random_seed: int = 1234):
        """
        Initialize clustering engine
        
        Arguments:
        ----------
            imputation_method { Literal } : How to handle missing values

            standardize         { bool }  : Whether to standardize features
            
            random_seed         { int }   : Random seed for reproducibility
        """
        self.imputation_method = imputation_method
        self.standardize       = standardize
        self.random_seed       = random_seed


    def extract_day_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract clinically meaningful daily features for clustering
        
        Returns:
        --------
            { pd.DataFrame }    : A pandas DataFrame with rows and columns as: Days × Features
        """
        features                = pd.DataFrame(index = data.index)

        features['mean']        = data.mean(axis   = 1, 
                                            skipna = True,
                                           )

        features['std']         = data.std(axis   = 1, 
                                           skipna = True,
                                          )

        features['cv']          = np.clip(a     = features['std'] / (features['mean'] + 1e-6), 
                                          a_min = 0, 
                                          a_max = eda_constants_instance.CV_CLIP_UPPER_BOUND,
                                         )

        features['n_obs']       = data.notna().sum(axis = 1)
        features['pct_severe']  = (data >= eda_constants_instance.SEVERE_THRESHOLD).sum(axis = 1) / features['n_obs'].replace(0, np.nan)

        return features.fillna(0)

    
    def prepare_data(self, data: pd.DataFrame) -> np.ndarray:
        """
        Prepare data for clustering with proper missing data handling

        NOTE:
        -----
        Clustering is performed on extracted day-level features, not on raw patient-by-day matrices
        
        Arguments:
        ----------
            data { pd.DataFrame } : PHQ-9 DataFrame (Days x Patients)
        
        Returns:
        --------
            { np.ndarray }        : Prepared feature matrix (Days x Features)
        """
        # Extract day-level features
        features = self.extract_day_features(data = data)
        X        = features.values

        # Handle missing values (feature-wise)
        if (self.imputation_method in ['mean', 'median']):
            imputer = SimpleImputer(strategy = self.imputation_method)
            X       = imputer.fit_transform(X)

        # Standardize if requested
        if self.standardize:
            scaler = StandardScaler()
            X      = scaler.fit_transform(X)

        return X

    
    def fit_kmeans(self, data: pd.DataFrame, n_clusters: int) -> Tuple[np.ndarray, float, float]:
        """
        Fit KMeans clustering
        
        Arguments:
        ----------
            data       { pd.DataFrame } : PHQ-9 DataFrame

            n_clusters      { int }     : Number of clusters
        
        Returns:
        --------
                    { tuple }           : A python tuple containing:
                                          - cluster labels
                                          - inertia
                                          - silhouette_score
        """
        # Prepare data
        X          = self.prepare_data(data = data)
        
        # Fit KMeans
        kmeans     = KMeans(n_clusters   = n_clusters,
                            init         = 'k-means++',
                            random_state = self.random_seed,
                            n_init       = 10,
                            max_iter     = 500,
                           ) 

        labels     = kmeans.fit_predict(X)
        
        # Calculate metrics
        inertia    = kmeans.inertia_

        if len(np.unique(labels)) < 2:
            silhouette = -1.0

        else:
            silhouette = silhouette_score(X, labels)

        return labels, inertia, silhouette

    
    def fit_agglomerative(self, data: pd.DataFrame, n_clusters: int, linkage: str = 'ward') -> Tuple[np.ndarray, float]:
        """
        Fit Agglomerative (hierarchical) clustering
        
        Arguments:
        ----------
            data       { pd.DataFrame } : PHQ-9 DataFrame

            n_clusters      { int }     : Number of clusters
            
            linkage         { str }     : Linkage criterion ('ward', 'complete', 'average')
        
        Returns:
        --------
                   { tuple }            : A python tuple containing:
                                          - cluster labels
                                          - silhouette_score
        """
        # Prepare data
        X          = self.prepare_data(data = data)
        
        # Fit Agglomerative
        if (linkage == 'ward'):
            agg = AgglomerativeClustering(n_clusters = n_clusters,
                                          linkage    = 'ward',
                                         )
        else:
            agg = AgglomerativeClustering(n_clusters = n_clusters,
                                          linkage    = linkage,
                                          metric     = 'euclidean',
                                         )

        labels     = agg.fit_predict(X)
        
        # Calculate silhouette
        if (len(np.unique(labels)) < 2):
            silhouette = -1.0

        else:
            silhouette = silhouette_score(X, labels)
        
        return labels, silhouette


class OptimalClusterSelector:
    """
    Determine optimal number of clusters using multiple methods
    """
    def __init__(self, clustering_engine: ClusteringEngine):
        """
        Initialize selector
        
        Arguments:
        ----------
            clustering_engine { ClusteringEngine } : ClusteringEngine instance
        """
        self.engine = clustering_engine
    

    def elbow_method(self, data: pd.DataFrame, max_clusters: int = None) -> Tuple[int, List[float]]:
        """
        Find optimal clusters using elbow method:
        - Uses angle-based detection (maximum distance from line)
        - NOT minimum percentage change (which is incorrect)
        
        Arguments:
        ----------
            data         { pd.DataFrame } : PHQ-9 DataFrame

            max_clusters      { int }     : Maximum clusters to test
        
        Returns:
        --------
                     { tuple }            : (optimal_k, inertia_list)
        """
        if max_clusters is None:
            max_clusters = eda_constants_instance.MAX_CLUSTERS_DEFAULT
        
        inertias = list()
        
        for k in range(eda_constants_instance.MIN_CLUSTERS, max_clusters + 1):
            _, inertia, _ = self.engine.fit_kmeans(data, k)
            inertias.append(inertia)
        
        # Find elbow using angle method : +MIN_CLUSTERS because we start from MIN_CLUSTERS
        optimal_k = self._find_elbow_angle(inertias) + eda_constants_instance.MIN_CLUSTERS
        
        return optimal_k, inertias
    

    def _find_elbow_angle(self, inertias: List[float]) -> int:
        """
        Find elbow point using angle-based method: Elbow is point with maximum distance from line connecting first and last points
        
        Arguments:
        ----------
            inertias { list } : List of inertia values
        
        Returns:
        --------
               { int }        : Index of elbow point
        """
        # Normalize coordinates to [0, 1]
        x         = np.arange(len(inertias))
        y         = np.array(inertias)
        
        # Handle edge case: constant inertias (no elbow exists)
        if (np.std(y) < 1e-10):
            # Return first point as default
            return 0  
        
        x_norm      = (x - x.min()) / (x.max() - x.min())
        y_norm      = (y - y.min()) / (y.max() - y.min())
        
        # Line from first to last point: ax + by + c = 0
        x1, y1      = x_norm[0], y_norm[0]
        x2, y2      = x_norm[-1], y_norm[-1]
        
        a           = y2 - y1
        b           = x1 - x2
        c           = x2 * y1 - x1 * y2
        
        # Check for degenerate line (denominator near zero)
        denominator = np.sqrt(a**2 + b**2)
        
        if (denominator < 1e-10):
            # Fallback: use derivative method when angle method fails
            return self._find_elbow_derivative(inertias = inertias)
        
        # Distance from each point to line
        distances = np.abs(a * x_norm + b * y_norm + c) / denominator
        
        # Elbow is point with maximum distance
        elbow_idx = np.argmax(distances)
        
        return elbow_idx


    def _find_elbow_derivative(self, inertias: List[float]) -> int:
        """
        Fallback elbow detection using second derivative method
        
        Strategy:
        ---------
        - Compute first derivative (rate of change)
        - Compute second derivative (acceleration)
        - Elbow is where second derivative is maximum (sharpest turn)
        
        Arguments:
        ----------
            inertias { list } : List of inertia values
        
        Returns:
        --------
               { int }        : Index of elbow point
        """
        if (len(inertias) < 3):
            # Not enough points for derivatives
            return 0  
        
        y         = np.array(inertias)
        
        # First derivative (discrete)
        dy        = np.diff(y)
        
        # Second derivative (discrete)
        d2y       = np.diff(dy)
        
        # Elbow is where curvature is maximum (most negative second derivative for decreasing function): for inertia curves (decreasing), most negative d2y is needed
        elbow_idx = np.argmin(d2y)
        
        # Add 2 because diff reduces length by 2
        return elbow_idx + 2

        
    def silhouette_method(self, data: pd.DataFrame, max_clusters: int = None) -> Tuple[int, List[float]]:
        """
        Find optimal clusters using silhouette analysis
        
        Arguments:
        ----------
            data         { pd.DataFrame } : PHQ-9 DataFrame

            max_clusters      { int }     : Maximum clusters to test
        
        Returns:
        --------
                    { tuple }             : A python tuple containing:
                                            - optimal_k
                                            - silhouette_scores
        """
        if max_clusters is None:
            max_clusters = eda_constants_instance.MAX_CLUSTERS_DEFAULT
        
        silhouettes = list()
        
        for k in range(eda_constants_instance.MIN_CLUSTERS, max_clusters + 1):
            _, _, silhouette = self.engine.fit_kmeans(data, k)

            silhouettes.append(silhouette)
        
        # Optimal is maximum silhouette
        optimal_k = np.argmax(silhouettes) + eda_constants_instance.MIN_CLUSTERS
        
        return optimal_k, silhouettes
    

    def gap_statistic(self, data: pd.DataFrame, max_clusters: int = 10, n_refs: int = 10) -> Tuple[int, List[float]]:
        """
        Find optimal clusters using Gap Statistic.

        Gap(k) = E[log(W_k_ref)] - log(W_k_real)

        Arguments:
        ----------
            data         { pd.DataFrame } : PHQ-9 DataFrame

            max_clusters      { int }     : Maximum clusters to test
            
            n_refs            { int }     : Number of reference datasets
        
        Returns:
        --------
                        { tuple }         : (optimal_k, gap_values)
        """
        # Prepare real data
        X    = self.engine.prepare_data(data = data)

        gaps = list()

        for k in range(1, max_clusters + 1):
            # Cluster real data
            if (k == 1):
                real_wk = np.sum((X - X.mean(axis=0)) ** 2)
            
            else:
                _, inertia, _ = self.engine.fit_kmeans(data, k)
                real_wk       = inertia

            # Reference datasets
            ref_wks = list()

            for _ in range(n_refs):
                rng   = np.random.default_rng(self.engine.random_seed)
                X_ref = rng.uniform(X.min(axis = 0), X.max(axis = 0), size = X.shape)

                if (k == 1):
                    ref_wk = np.sum((X_ref - X_ref.mean(axis = 0)) ** 2)
                
                else:
                    kmeans_ref = KMeans(n_clusters = k, 
                                        n_init     = 10,
                                       )

                    kmeans_ref.fit(X_ref)

                    ref_wk     = kmeans_ref.inertia_

                ref_wks.append(np.log(ref_wk))

            gap = np.mean(ref_wks) - np.log(real_wk)
            gaps.append(gap)

        optimal_k = (np.argmax(gaps) + 1)

        return optimal_k, gaps



class TemporalClustering:
    """
    Clustering with temporal proximity constraints: Days that are close in time are encouraged to cluster together
    """
    def __init__(self, temporal_weight: float = 0.3, random_seed: int = 1234):
        """
        Initialize temporal clustering
        
        Arguments:
        ----------
            temporal_weight { float } : Weight for temporal proximity (0-1)
                                        - 0 = ignore time
                                        - 1 = only time

            random_seed      { int  } : Random seed
        """
        self.temporal_weight = temporal_weight
        self.random_seed     = random_seed
        
        if not (0 <= temporal_weight <= 1):
            raise ValueError(f"temporal_weight must be in [0, 1], got {temporal_weight}")
    

    def fit(self, data: pd.DataFrame, n_clusters: int, imputation_method: str = 'mean') -> np.ndarray:
        """
        Fit clustering with temporal constraints: 
        - Combined distance = (1-w)*score_distance + w*temporal_distance
        
        Arguments:
        ----------
            data              { pd.DataFrame } : PHQ-9 DataFrame (Days x Patients)

            n_clusters             { int }     : Number of clusters
            
            imputation_method      { str }     : How to handle missing values
        
        Returns:
        --------
                     { np.ndarray }            : Cluster labels for each day
        """
        # Prepare data
        engine             = ClusteringEngine(imputation_method = imputation_method,
                                              random_seed       = self.random_seed,
                                             )

        X                  = engine.prepare_data(data = data)
        
        # Calculate score-based distances
        score_distances    = euclidean_distances(X)
        
        # Calculate temporal distances
        n_days             = X.shape[0]
        
        temporal_distances = np.abs(np.arange(n_days)[:, None] - np.arange(n_days))
        
        # Normalize temporal distances to [0, 1]
        temporal_distances = temporal_distances / temporal_distances.max()
        
        # Combine distances
        combined_distances = ((1 - self.temporal_weight) * score_distances + self.temporal_weight * temporal_distances)
        
        # Use agglomerative clustering with precomputed distances
        agg                = AgglomerativeClustering(n_clusters = n_clusters,
                                                     metric     = 'precomputed',
                                                     linkage    = 'average',
                                                    )
        
        labels             = agg.fit_predict(combined_distances)
        
        return labels