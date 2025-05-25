"""Implements the signature verification system described in the paper:

    "Online Signature Verification Using Locally
    Weighted Dynamic Time Warping via
    Multiple Fusion Strategies"

    by:
    - MANABU OKAWA

    Last run test results:
        SVC2004 Task1 EER: 0.0350
        SVC2004 Task2 EER: 0.0287
"""
import os, random, re, warnings, time, zipfile
from collections import namedtuple, Counter
import requests
import numpy as np
from tslearn.barycenters import dtw_barycenter_averaging
from tslearn.metrics import dtw_path
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve
from joblib import Parallel, delayed
from numba import njit, prange
from sklearn.model_selection import GridSearchCV

# Add fixed random seed for reproducibility
random.seed(42)
np.random.seed(42)

# -- Data structure --
SignaturePoint = namedtuple('SignaturePoint', ['x', 'y', 'pressure', 'timestamp', 'strokeId'])

# -- Preprocessing & derivatives --
@njit
def compute_derivatives(x, T):
    dx = np.zeros(T, dtype=np.float32)
    if T >= 5:
        for t in range(2, T-2):
            dx[t] = (1*(x[t+1] - x[t-1]) + 2*(x[t+2] - x[t-2])) / 10
    elif T >= 3:
        for t in range(1, T-1):
            dx[t] = (x[t+1] - x[t-1]) / 2
    if T >= 2:
        dx[0] = x[1] - x[0]
        dx[-1] = x[-1] - x[-2]
    return dx

def preprocess_signature(signature, include_pressure=True):
    if not signature:
        raise ValueError("Empty signature provided.")
    x = np.array([p.x for p in signature], dtype=np.float32)
    y = np.array([p.y for p in signature], dtype=np.float32)
    p = np.array([p.pressure for p in signature], dtype=np.float32) if include_pressure else None

    # Normalize
    x_g, y_g = np.mean(x), np.mean(y)
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    x_range = max(1e-5, x_max - x_min)
    y_range = max(1e-5, y_max - y_min)
    x_hat = (x - x_g)/x_range
    y_hat = (y - y_g)/y_range

    T = len(signature)
    dx = compute_derivatives(x_hat, T)
    dy = compute_derivatives(y_hat, T)
    eps = 1e-9
    theta = np.arctan2(dy+eps, dx+eps)
    v = np.sqrt(dx*dx + dy*dy)
    dtheta = compute_derivatives(theta, T)
    dv = compute_derivatives(v, T)
    rho = np.log(np.maximum(v, eps)/np.maximum(np.abs(dtheta), eps))
    alpha = np.sqrt(dv*dv + (v*dtheta)**2)

    feats = [x_hat, y_hat]
    if include_pressure:
        feats.append(p)
    feats.extend([theta, v, rho, alpha])

    # Standardize & clip
    std_feats = []
    for f in feats:
        m, s = f.mean(), f.std()
        s = max(s, 1e-5)
        sf = (f-m)/s
        std_feats.append(np.clip(sf, -1e6, 1e6))
    return std_feats

# -- Improved mean-template: per-feature init lengths --
def compute_mean_template(ref_signatures, include_pressure=True):
    ref_feats = [preprocess_signature(sig, include_pressure) for sig in ref_signatures]
    K = len(ref_feats[0])

    # Compute initial length per feature
    initial = []
    for k in range(K):
        lengths = [len(f[k]) for f in ref_feats]
        init_len_k = int(round(np.mean(lengths)))
        seqs = [np.interp(np.linspace(0,len(f[k])-1,init_len_k), np.arange(len(f[k])), f[k]).reshape(-1,1) for f in ref_feats]
        init_bary = np.mean(np.stack(seqs,axis=2), axis=2)
        initial.append(init_bary)

    # DTW-Barycenter
    mean_templates = []
    for k in range(K):
        seqs = [np.array(f[k]).reshape(-1,1) for f in ref_feats if len(f[k])>0]
        tpl = dtw_barycenter_averaging(seqs, init_barycenter=initial[k])
        mean_templates.append(tpl.flatten())

    # Pad to max_len
    max_len = max(len(t) for t in mean_templates)
    return [np.pad(t,(0,max_len-len(t)),mode='edge') for t in mean_templates]

# -- Force custom DTW implementations --
@njit
def compute_weighted_dtw_1d(A, Q, w):
    n,m = len(A), len(Q)
    D = np.full((n+1,m+1), np.inf)
    D[0,0] = 0.0
    for i in range(1,n+1):
        for j in range(1,m+1):
            cost = w[i-1]*(A[i-1]-Q[j-1])**2
            D[i,j] = cost + min(D[i-1,j],D[i,j-1],D[i-1,j-1])
    return D[n,m]

@njit(parallel=True)
def compute_weighted_dtw_2d(A, Q, w, window_radius=None):
    n,K = A.shape
    m,_ = Q.shape
    D = np.full((n+1,m+1), np.inf)
    D[0,0] = 0.0
    for i in range(1,n+1):
        j0 = 1
        j1 = m+1
        for j in prange(j0, j1):
            cost = 0.0
            for k in range(K): cost += (A[i-1,k]-Q[j-1,k])**2
            D[i,j] = w[i-1]*cost + min(D[i-1,j],D[i,j-1],D[i-1,j-1])
    return D[n,m]


def compute_weighted_dtw(A, Q, w, window_radius=None):
    A, Q, w = map(lambda x: np.asarray(x, dtype=np.float32), (A, Q, w))
    if A.ndim==1:
        return compute_weighted_dtw_1d(A, Q, w)
    return compute_weighted_dtw_2d(A, Q, w, window_radius)

# -- Local weights with no window constraint for weight estimation --
def compute_local_weights(mean_templates, ref_features):
    K = len(mean_templates)
    I = len(mean_templates[0]) if K>0 else 0
    if I==0 or not ref_features:
        return [np.ones(I,dtype=np.float32) for _ in range(K)], *([np.ones(I,dtype=np.float32) for _ in range(2)])

    LM_I = [np.zeros(I,dtype=np.float32) for _ in range(K)]
    LD_I = [np.zeros(I,dtype=np.float32) for _ in range(K)]
    LM_D = np.zeros(I,dtype=np.float32)
    LD_D = np.zeros(I,dtype=np.float32)
    num_refs = len(ref_features)

    # Independent weights
    for k in range(K):
        for feats in ref_features:
            path,_ = dtw_path(mean_templates[k], feats[k], global_constraint=None)
            path = np.array(path)
            i_vals = path[:,0]
            for idx,count in Counter(i_vals).items():
                LM_I[k][idx] += count/num_refs
            for i_idx,j_idx in path:
                if Counter(i_vals)[i_idx]==1 and Counter(path[:,1])[j_idx]==1:
                    LD_I[k][i_idx] += 1/num_refs
        LM_I[k] = 1.0/np.where(LM_I[k]<1e-6,1,LM_I[k])

    # Dependent weights
    A = np.stack(mean_templates,axis=1)
    for feats in ref_features:
        lens = [len(f) for f in feats]
        min_len = min(lens)
        B = np.stack([f[:min_len] for f in feats], axis=1)
        path,_ = dtw_path(A, B, global_constraint=None)
        path = np.array(path)
        for idx,count in Counter(path[:,0]).items():
            LM_D[idx] += count/num_refs
        for i_idx,j_idx in path:
            if Counter(path[:,0])[i_idx]==1 and Counter(path[:,1])[j_idx]==1:
                LD_D[i_idx] += 1/num_refs
    LM_D = 1.0/np.where(LM_D<1e-6,1,LM_D)
    return LM_I, LD_I, LM_D, LD_D

# -- F-DTW computation unchanged --
def compute_F_DTW(mean_templates, query_features, LM_I, LD_I, LM_D, LD_D):
    K = len(mean_templates)
    FI=[]; FD_I=[]
    for k in range(K):
        A = mean_templates[k].reshape(-1,1)
        Q = np.array(query_features[k]).reshape(-1,1)
        FI.append(compute_weighted_dtw(A,Q,LM_I[k]))
        FD_I.append(compute_weighted_dtw(A,Q,LD_I[k]))
    F_I = np.array(FI+FD_I)
    # dependent
    A_all = np.stack(mean_templates,axis=1)
    Q_all = np.stack(query_features,axis=1)
    fd = [compute_weighted_dtw(A_all,Q_all,LM_D), compute_weighted_dtw(A_all,Q_all,LD_D)]
    return F_I, np.array(fd)

# -- Verifier class (uses calibrate_threshold) --
class SignatureVerifier:
    def __init__(self, include_pressure=True):
        self.include_pressure=include_pressure
        self.scaler_I = StandardScaler(); self.scaler_D = StandardScaler()
        self.C_I = 0.01  # Default C value based on grid search results
        self.C_D = 0.01  # Default C value based on grid search results
        # Add timing metrics
        self.training_time = 0
        self.prediction_times = []

    def train(self, ref_signatures, neg_features=None, grid_search=False):
        start_time = time.time()
        timings = {}  # Dictionary to collect timing information
        
        ref_feats = [preprocess_signature(sig,self.include_pressure) for sig in ref_signatures]
        timings['preprocessing'] = time.time() - start_time
        
        self.mean_templates = compute_mean_template(ref_signatures,self.include_pressure)
        timings['template_computation'] = time.time() - (start_time + timings['preprocessing'])
        
        self.LM_I, self.LD_I, self.LM_D, self.LD_D = compute_local_weights(self.mean_templates,ref_feats)
        timings['weight_computation'] = time.time() - (start_time + timings['preprocessing'] + timings['template_computation'])
        
        # positives
        pos_I=[]; pos_D=[]
        for f in ref_feats:
            fi,fd = compute_F_DTW(self.mean_templates,f,self.LM_I,self.LD_I,self.LM_D,self.LD_D)
            pos_I.append(fi); pos_D.append(fd)
        
        # negatives
        neg_feats = neg_features if neg_features is not None else []
        
        # If no negative features provided, generate synthetic ones
        if len(neg_feats) == 0:
            print("  No negative samples provided, generating synthetic negatives...")
            neg_feats = self._generate_synthetic_negatives(ref_signatures, ref_feats)
        
        neg_I=[]; neg_D=[]
        for f in neg_feats:
            fi,fd = compute_F_DTW(self.mean_templates,f,self.LM_I,self.LD_I,self.LM_D,self.LD_D)
            neg_I.append(fi); neg_D.append(fd)
        
        timings['dtw_feature_extraction'] = time.time() - (start_time + timings['preprocessing'] + 
                                                 timings['template_computation'] + timings['weight_computation'])
        
        # stack
        X_I = np.vstack([pos_I,neg_I])
        X_D = np.vstack([pos_D,neg_D])
        y = np.array([1]*len(pos_I)+[0]*len(neg_I))
        
        # Apply scaling
        X_I_s = self.scaler_I.fit_transform(X_I)
        X_D_s = self.scaler_D.fit_transform(X_D)
        
        # Grid search for optimal C values if enabled
        if grid_search and len(neg_I) > 0:
            print("  Starting grid search for SVM hyperparameters...")
            C_values = [0.01, 0.1, 1.0, 10.0]
            best_eer_I = float('inf')
            best_eer_D = float('inf')
            
            # Use a small validation set for tuning
            val_split = 0.3
            n_val = max(1, int(len(pos_I) * val_split))
            val_indices = np.random.choice(len(pos_I), n_val, replace=False)
            train_indices = np.array([i for i in range(len(pos_I)) if i not in val_indices])
            
            # Combined indices for the full dataset
            all_val_indices = np.concatenate([val_indices, len(pos_I) + np.arange(len(neg_I))])
            all_train_indices = np.concatenate([train_indices, len(pos_I) + np.arange(len(neg_I))])
            
            # Grid search for C_I
            print("  Grid search for independent features (I):")
            for C in C_values:
                svm = LinearSVC(C=C, class_weight={1:len(neg_I)/len(pos_I),0:1.0}, random_state=42)
                svm.fit(X_I_s[all_train_indices], y[all_train_indices])
                scores = svm.decision_function(X_I_s[all_val_indices])
                val_y = y[all_val_indices]
                # Calculate EER on validation set
                fpr, tpr, _ = roc_curve(val_y, scores)
                fnr = 1 - tpr
                eer = fpr[np.argmin(np.abs(fpr - fnr))]
                print(f"    C={C}: EER={eer:.4f}" + (" ✓" if eer < best_eer_I else ""))
                if eer < best_eer_I:
                    best_eer_I = eer
                    self.C_I = C
            
            # Grid search for C_D
            print("  Grid search for dependent features (D):")
            for C in C_values:
                svm = LinearSVC(C=C, class_weight={1:len(neg_I)/len(pos_I),0:1.0}, random_state=42)
                svm.fit(X_D_s[all_train_indices], y[all_train_indices])
                scores = svm.decision_function(X_D_s[all_val_indices])
                val_y = y[all_val_indices]
                fpr, tpr, _ = roc_curve(val_y, scores)
                fnr = 1 - tpr
                eer = fpr[np.argmin(np.abs(fpr - fnr))]
                print(f"    C={C}: EER={eer:.4f}" + (" ✓" if eer < best_eer_D else ""))
                if eer < best_eer_D:
                    best_eer_D = eer
                    self.C_D = C
            
            print(f"  Selected hyperparameters: C_I={self.C_I}, C_D={self.C_D}")
            timings['grid_search'] = time.time() - (start_time + sum(timings.values()))
        
        # Train final SVMs with optimal C values
        cw={1:len(neg_I)/len(pos_I),0:1.0}
        self.svm_I = LinearSVC(C=self.C_I, class_weight=cw, random_state=42).fit(X_I_s, y)
        self.svm_D = LinearSVC(C=self.C_D, class_weight=cw, random_state=42).fit(X_D_s, y)
        timings['svm_training'] = time.time() - (start_time + sum(timings.values()))
        
        self.training_time = time.time() - start_time
        
        # Print a single summarized timing report
        print(f"  Training summary ({len(ref_signatures)} ref signatures, {len(neg_feats)} negative samples):")
        print(f"    - Preprocessing: {timings['preprocessing']:.4f}s")
        print(f"    - Template computation: {timings['template_computation']:.4f}s")
        print(f"    - Weight computation: {timings['weight_computation']:.4f}s")
        print(f"    - DTW feature extraction: {timings['dtw_feature_extraction']:.4f}s")
        if 'grid_search' in timings:
            print(f"    - Grid search: {timings['grid_search']:.4f}s")
        print(f"    - SVM training: {timings['svm_training']:.4f}s")
        print(f"    - Total: {self.training_time:.4f}s")

        # Set default threshold since we don't have real forgeries for calibration
        self._set_default_threshold(pos_I, pos_D)

    def _generate_synthetic_negatives(self, ref_signatures, ref_feats, num_synthetic=20, return_raw=False):
        """Generate synthetic negative samples using various distortion techniques."""
        synthetic_negatives = []
        raw_signatures = []  # Keep track of raw signatures when return_raw=True
        
        # 1. Get base signature features to manipulate
        for ref_sig, ref_feat in zip(ref_signatures, ref_feats):
            if not ref_sig or not ref_feat:
                continue
            
            # Generate several types of synthetic negatives
            
            # Type 1: Global transformations (rotation, scaling, translation)
            for angle in [15, 30, 45]:
                # Create rotated version
                rotated_sig = self._apply_rotation(ref_sig, angle)
                if return_raw:
                    raw_signatures.append(rotated_sig)
                else:
                    rotated_feat = preprocess_signature(rotated_sig, self.include_pressure)
                    synthetic_negatives.append(rotated_feat)
            
            # Type 2: Add jitter to points
            for noise_level in [0.1, 0.2]:
                noisy_sig = self._add_noise(ref_sig, noise_level)
                if return_raw:
                    raw_signatures.append(noisy_sig)
                else:
                    noisy_feat = preprocess_signature(noisy_sig, self.include_pressure)
                    synthetic_negatives.append(noisy_feat)
            
            # Type 3: Change stroke timing
            time_warped_sig = self._time_warp(ref_sig)
            if return_raw:
                raw_signatures.append(time_warped_sig)
            else:
                time_warped_feat = preprocess_signature(time_warped_sig, self.include_pressure)
                synthetic_negatives.append(time_warped_feat)
            
            # Type 4: Change pressure patterns
            if self.include_pressure:
                pressure_mod_sig = self._modify_pressure(ref_sig)
                if return_raw:
                    raw_signatures.append(pressure_mod_sig)
                else:
                    pressure_mod_feat = preprocess_signature(pressure_mod_sig, self.include_pressure)
                    synthetic_negatives.append(pressure_mod_feat)
        
        # Type 5: Generate completely random signatures that follow signature-like statistics
        for _ in range(5):
            random_sig = self._generate_random_signature(ref_signatures)
            if return_raw:
                raw_signatures.append(random_sig)
            else:
                random_feat = preprocess_signature(random_sig, self.include_pressure)
                synthetic_negatives.append(random_feat)
        
        # Return the appropriate type based on return_raw parameter
        if return_raw:
            return raw_signatures[:num_synthetic]
        else:
            return synthetic_negatives[:num_synthetic]

    def _apply_rotation(self, signature, angle_degrees):
        """Apply rotation to a signature."""
        angle_rad = np.radians(angle_degrees)
        rot_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)]
        ])
        
        # Extract coordinates
        coords = np.array([(p.x, p.y) for p in signature])
        
        # Find center point
        center = np.mean(coords, axis=0)
        
        # Center, rotate, and translate back
        centered = coords - center
        rotated = np.dot(centered, rot_matrix.T)
        transformed = rotated + center
        
        # Create new signature points
        new_sig = []
        for i, (x, y) in enumerate(transformed):
            new_sig.append(SignaturePoint(
                x=x,
                y=y,
                pressure=signature[i].pressure,
                timestamp=signature[i].timestamp,
                strokeId=signature[i].strokeId
            ))
        return new_sig

    def _add_noise(self, signature, noise_level):
        """Add random jitter to signature points."""
        # Calculate range of coordinates for scaling noise
        x_coords = [p.x for p in signature]
        y_coords = [p.y for p in signature]
        x_range = max(x_coords) - min(x_coords)
        y_range = max(y_coords) - min(y_coords)
        
        new_sig = []
        for p in signature:
            # Add scaled noise
            noise_x = np.random.normal(0, noise_level * x_range)
            noise_y = np.random.normal(0, noise_level * y_range)
            
            new_sig.append(SignaturePoint(
                x=p.x + noise_x,
                y=p.y + noise_y,
                pressure=p.pressure,
                timestamp=p.timestamp,
                strokeId=p.strokeId
            ))
        return new_sig

    def _time_warp(self, signature):
        """Modify the timing pattern of a signature."""
        # Get time range
        times = [p.timestamp for p in signature]
        t_min, t_max = min(times), max(times)
        duration = t_max - t_min
        
        # Create warped timestamps
        if len(signature) > 10:
            # Add a few random pauses
            pause_points = np.random.choice(range(1, len(signature)-1), 
                                            size=min(3, len(signature)//10), 
                                            replace=False)
            
            new_sig = []
            extra_time = 0
            for i, p in enumerate(signature):
                if i in pause_points:
                    # Add a pause (20-40% of total duration)
                    pause_time = np.random.uniform(0.2, 0.4) * duration
                    extra_time += pause_time
                
                new_sig.append(SignaturePoint(
                    x=p.x,
                    y=p.y,
                    pressure=p.pressure,
                    timestamp=p.timestamp + int(extra_time),
                    strokeId=p.strokeId
                ))
            return new_sig
        return signature  # Too short to warp meaningfully

    def _modify_pressure(self, signature):
        """Change pressure patterns in the signature."""
        if not self.include_pressure:
            return signature
        
        pressures = [p.pressure for p in signature]
        p_min, p_max = min(pressures), max(pressures)
        p_range = p_max - p_min
        
        # Invert pressure pattern
        new_sig = []
        for p in signature:
            # Invert pressure (high becomes low, low becomes high)
            new_pressure = p_max - (p.pressure - p_min)
            
            new_sig.append(SignaturePoint(
                x=p.x,
                y=p.y,
                pressure=new_pressure,
                timestamp=p.timestamp,
                strokeId=p.strokeId
            ))
        return new_sig

    def _generate_random_signature(self, ref_signatures):
        """Generate a completely new signature with similar statistics to reference ones."""
        # Analyze reference signatures for statistics
        all_strokes = []
        for sig in ref_signatures:
            strokes = {}
            for p in sig:
                if p.strokeId not in strokes:
                    strokes[p.strokeId] = []
                strokes[p.strokeId].append(p)
            all_strokes.extend(list(strokes.values()))
        
        # Get average number of strokes and points
        stroke_counts = [len(set(p.strokeId for p in sig)) for sig in ref_signatures]
        avg_strokes = max(3, int(np.mean(stroke_counts)) if stroke_counts else 3)
        avg_points_per_stroke = max(5, int(np.mean([len(stroke) for stroke in all_strokes])) if all_strokes else 5)
        
        # Generate random signature
        random_sig = []
        
        # Get coordinate and pressure ranges from reference signatures
        x_coords = [p.x for sig in ref_signatures for p in sig]
        y_coords = [p.y for sig in ref_signatures for p in sig]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Generate random strokes
        timestamp = 0
        for stroke_id in range(np.random.randint(avg_strokes-1, avg_strokes+2)):
            # Random starting point
            x = np.random.uniform(x_min, x_max)
            y = np.random.uniform(y_min, y_max)
            
            # Random direction
            angle = np.random.uniform(0, 2*np.pi)
            dx = np.cos(angle)
            dy = np.sin(angle)
            
            # Generate stroke points
            points_in_stroke = np.random.randint(avg_points_per_stroke-3, avg_points_per_stroke+4)
            for i in range(points_in_stroke):
                # Move in the general direction with some noise
                x += dx + np.random.normal(0, 0.5)
                y += dy + np.random.normal(0, 0.5)
                
                # Ensure we stay in bounds
                x = max(x_min, min(x_max, x))
                y = max(y_min, min(y_max, y))
                
                # Random pressure if included
                pressure = 0
                if self.include_pressure:
                    pressures = [p.pressure for sig in ref_signatures for p in sig]
                    p_min, p_max = min(pressures), max(pressures)
                    pressure = np.random.uniform(p_min, p_max)
                
                # Add point
                timestamp += np.random.randint(10, 30)  # Random time increment
                random_sig.append(SignaturePoint(
                    x=x,
                    y=y,
                    pressure=pressure,
                    timestamp=timestamp,
                    strokeId=stroke_id
                ))
                
                # Occasionally change direction
                if np.random.random() < 0.2:
                    angle += np.random.uniform(-np.pi/4, np.pi/4)
                    dx = np.cos(angle)
                    dy = np.sin(angle)
        
        return random_sig

    def _set_default_threshold(self, pos_I, pos_D):
        """Set a reasonable threshold based on variation in genuine samples."""
        # Compute within-class variation
        if len(pos_I) >= 2:
            # Use SVMs to get scores for positive samples
            I_scores = self.svm_I.decision_function(self.scaler_I.transform(pos_I))
            D_scores = self.svm_D.decision_function(self.scaler_D.transform(pos_D))
            
            # Combine scores
            combined_scores = I_scores + D_scores
            
            # Get statistics
            mean_score = np.mean(combined_scores)
            std_score = np.std(combined_scores)
            
            # Set threshold conservatively below the minimum
            # The factor here should be tuned
            min_score = mean_score - 2.5 * std_score
            self.threshold_ = min_score
        else:
            # If only one sample, set a default conservative threshold
            self.threshold_ = -0.5
        
        print(f"  Default threshold set to: {self.threshold_:.3f}")

    def calibrate_threshold(self, genuine_sigs, forged_sigs):
        genu = [self.predict(sig) for sig in genuine_sigs]
        forg = [self.predict(sig) for sig in forged_sigs]
        fpr,t = calculate_eer(genu,forg)
        self.threshold_ = t
        return t

    def predict(self, signature):
        start_time = time.time()
        
        # Initialize diagnostic info dictionary
        diagnostics = {
            'timing': {},
            'feature_stats': {},
            'svm_scores': {},
            'final_score': None
        }
        
        try:
            # Preprocessing
            f = preprocess_signature(signature, self.include_pressure)
            preprocess_time = time.time() - start_time
            diagnostics['timing']['preprocess'] = preprocess_time
            
            # Log feature statistics
            diagnostics['feature_stats'] = {
                'num_points': len(signature),
                'num_strokes': len(set(p.strokeId for p in signature)),
                'duration': signature[-1].timestamp - signature[0].timestamp,
                'feature_lengths': [len(feat) for feat in f]
            }
            
            # DTW computation
            dtw_start = time.time()
            FI, FD = compute_F_DTW(self.mean_templates, f, self.LM_I, self.LD_I, self.LM_D, self.LD_D)
            dtw_time = time.time() - dtw_start
            diagnostics['timing']['dtw'] = dtw_time
            
            # Store DTW feature statistics
            diagnostics['feature_stats'].update({
                'independent_features': FI.tolist(),
                'dependent_features': FD.tolist()
            })
            
            # SVM scoring
            svm_start = time.time()
            si = self.svm_I.decision_function(self.scaler_I.transform([FI]))[0]
            sd = self.svm_D.decision_function(self.scaler_D.transform([FD]))[0]
            svm_time = time.time() - svm_start
            diagnostics['timing']['svm'] = svm_time
            
            # Store SVM scores
            diagnostics['svm_scores'] = {
                'independent_score': float(si),
                'dependent_score': float(sd)
            }
            
            # Final result
            result = si + sd
            diagnostics['final_score'] = float(result)
            diagnostics['threshold'] = float(getattr(self, 'threshold_', 0))
            diagnostics['match_result'] = result >= getattr(self, 'threshold_', 0)
            
            total_time = time.time() - start_time
            diagnostics['timing']['total'] = total_time
            
            # Store timing information
            self.prediction_times.append({
                'preprocess': preprocess_time,
                'dtw': dtw_time,
                'svm': svm_time,
                'total': total_time
            })
            
            # Log detailed diagnostics
            print("\nSignature Verification Diagnostics:")
            print(f"├── Input Statistics:")
            print(f"│   ├── Points: {diagnostics['feature_stats']['num_points']}")
            print(f"│   ├── Strokes: {diagnostics['feature_stats']['num_strokes']}")
            print(f"│   └── Duration: {diagnostics['feature_stats']['duration']}ms")
            print(f"├── Processing Times:")
            print(f"│   ├── Preprocessing: {diagnostics['timing']['preprocess']:.4f}s")
            print(f"│   ├── DTW Computation: {diagnostics['timing']['dtw']:.4f}s")
            print(f"│   ├── SVM Scoring: {diagnostics['timing']['svm']:.4f}s")
            print(f"│   └── Total: {diagnostics['timing']['total']:.4f}s")
            print(f"├── Verification Scores:")
            print(f"│   ├── Independent Features: {diagnostics['svm_scores']['independent_score']:.4f}")
            print(f"│   ├── Dependent Features: {diagnostics['svm_scores']['dependent_score']:.4f}")
            print(f"│   ├── Combined Score: {diagnostics['final_score']:.4f}")
            print(f"│   └── Threshold: {diagnostics['threshold']:.4f}")
            print(f"└── Result: {'MATCH' if diagnostics['match_result'] else 'NO MATCH'}")
            
            # Store diagnostics in instance for later reference
            if not hasattr(self, 'prediction_diagnostics'):
                self.prediction_diagnostics = []
            self.prediction_diagnostics.append(diagnostics)
            
            return result
            
        except Exception as e:
            print("\nError during signature verification:")
            print(f"├── Error Type: {type(e).__name__}")
            print(f"├── Error Message: {str(e)}")
            print(f"└── Verification Failed")
            raise

    def verify(self, sig):
        scr = self.predict(sig)
        return scr>=getattr(self,'threshold_',0)

def calculate_eer(genu,forg):
    scores = genu+forg
    labels = [1]*len(genu)+[0]*len(forg)
    fpr,tpr,th = roc_curve(labels,scores)
    fnr = 1-tpr
    i = np.argmin(np.abs(fpr-fnr))
    return fpr[i], th[i]

# -- Evaluation: micro-average EER, no neg-sampling cap --
def evaluate_dataset(name, data_dir, n_jobs=-1, num_trials=5, grid_search=True):
    print(f"Evaluating {name}")
    phase_start_time = time.time()
    
    if name=='SVC2004_Task1':
        users = load_svc2004_task1(data_dir); include_pressure=False
    else:
        users = load_svc2004_task2(data_dir); include_pressure=True

    total_users = len(users)
    print(f"Dataset contains {total_users} users")
    
    all_trials_eers = []
    
    # Pre-compute user templates and weights
    print("Pre-computing user templates and weights...")
    precomputation_start = time.time()
    
    user_templates = {}
    user_weights = {}
    user_ref_sigs = {}
    
    for uid in users:
        genuine = users[uid]['genuine']
        if len(genuine) < 5:
            continue
            
        # Sample reference signatures for each user (same ones used across trials)
        ref = random.sample(genuine, 5)
        user_ref_sigs[uid] = ref
        
        # Pre-compute templates and weights
        templates = compute_mean_template(ref, include_pressure)
        user_templates[uid] = templates
        
        ref_feats = [preprocess_signature(s, include_pressure) for s in ref]
        LM_I, LD_I, LM_D, LD_D = compute_local_weights(templates, ref_feats)
        user_weights[uid] = (LM_I, LD_I, LM_D, LD_D)
    
    # REVERTING CHANGE: Use forgeries of the user as negatives, not genuine from other users
    print("Pre-computing negative features from user's forgeries...")
    neg_features_cache = {}
    for uid in user_templates:
        neg_features_cache[uid] = []
        target_templates = user_templates[uid]
        LM_I, LD_I, LM_D, LD_D = user_weights[uid]
        
        # Use forgeries of the current user as negatives
        forgeries = users[uid]['forged']
        if forgeries:
            # Use all available forgeries (up to a reasonable limit)
            for sig in forgeries[:50]:  # Limit to 50 forgeries max
                sig_feats = preprocess_signature(sig, include_pressure)
                fi, fd = compute_F_DTW(target_templates, sig_feats, LM_I, LD_I, LM_D, LD_D)
                neg_features_cache[uid].append((sig_feats, fi, fd))

    precomputation_time = time.time() - precomputation_start
    print(f"Pre-computation completed in {precomputation_time:.2f}s")
    
    # For each trial
    for trial in range(num_trials):
        trial_start = time.time()
        print(f"Running trial {trial+1}/{num_trials}")
        per_user_eers = []
        
        def process_user(uid):
            if uid not in user_templates:
                return None
                
            genuine = users[uid]['genuine']
            forged = users[uid]['forged']
            if len(genuine) < 5 or not forged:
                return None
            
            # Get pre-computed reference signatures
            ref = user_ref_sigs[uid]
            test_g = [s for s in genuine if s not in ref]
            test_f = forged
            
            # Get pre-computed negative features
            neg_features = [templates for templates, _, _ in neg_features_cache[uid]]
            
            # Train verifier
            v = SignatureVerifier(include_pressure)
            v.mean_templates = user_templates[uid]  # Use pre-computed templates
            v.LM_I, v.LD_I, v.LM_D, v.LD_D = user_weights[uid]  # Use pre-computed weights
            
            # Compute features for reference signatures
            ref_feats = [preprocess_signature(s, include_pressure) for s in ref]
            pos_I = []; pos_D = []
            for f in ref_feats:
                fi, fd = compute_F_DTW(v.mean_templates, f, v.LM_I, v.LD_I, v.LM_D, v.LD_D)
                pos_I.append(fi); pos_D.append(fd)
            
            # Get features from pre-computed negative features cache
            neg_I = []; neg_D = []
            for _, fi, fd in neg_features_cache[uid]:
                neg_I.append(fi); neg_D.append(fd)
            
            # Stack and train SVMs
            X_I = np.vstack([pos_I, neg_I]); X_D = np.vstack([pos_D, neg_D])
            y = np.array([1]*len(pos_I)+[0]*len(neg_I))
            
            # Apply scaling
            v.scaler_I = StandardScaler()
            v.scaler_D = StandardScaler()
            X_I_s = v.scaler_I.fit_transform(X_I)
            X_D_s = v.scaler_D.fit_transform(X_D)
            
            # Use GridSearchCV for hyperparameter tuning if grid_search is enabled
            if grid_search and len(neg_I) > 0:
                # Create parameter grid
                param_grid = {'C': [0.01, 0.1, 1.0, 10.0]}
                cw = {1: len(neg_I)/len(pos_I), 0: 1.0}
                
                # Grid search for independent features
                grid_I = GridSearchCV(
                    LinearSVC(class_weight=cw, random_state=42),
                    param_grid,
                    cv=3,
                    scoring='neg_mean_squared_error'
                )
                grid_I.fit(X_I_s, y)
                v.C_I = grid_I.best_params_['C']
                v.svm_I = grid_I.best_estimator_
                
                # Grid search for dependent features
                grid_D = GridSearchCV(
                    LinearSVC(class_weight=cw, random_state=42),
                    param_grid,
                    cv=3,
                    scoring='neg_mean_squared_error'
                )
                grid_D.fit(X_D_s, y)
                v.C_D = grid_D.best_params_['C']
                v.svm_D = grid_D.best_estimator_
            else:
                # Use default hyperparameters
                cw = {1: len(neg_I)/len(pos_I), 0: 1.0}
                v.svm_I = LinearSVC(C=v.C_I, class_weight=cw, random_state=42).fit(X_I_s, y)
                v.svm_D = LinearSVC(C=v.C_D, class_weight=cw, random_state=42).fit(X_D_s, y)
            
            # CHANGE 1: Use exactly 5 forgeries for threshold calibration, not 15
            calib_f = random.sample(forged, min(len(forged), 5))  # Changed from 15 to 5
            v.calibrate_threshold(ref, calib_f)
            
            # Test on held-out genuines and forgeries
            genu_scores = [v.predict(s) for s in test_g]
            forg_scores = [v.predict(s) for s in test_f]
            
            # Calculate per-user EER
            if genu_scores and forg_scores:
                user_eer, _ = calculate_eer(genu_scores, forg_scores)
                return user_eer, genu_scores, forg_scores
            return None
        
        # Process each user and collect per-user EERs
        print(f"Processing {total_users} users in parallel with {n_jobs} jobs...")
        
        # Add verbose=10 to show progress in parallel execution
        results = Parallel(n_jobs=n_jobs, verbose=10)(delayed(process_user)(uid) for uid in users)
        
        # Count valid results
        valid_results = sum(1 for r in results if r is not None)
        print(f"Completed processing {valid_results}/{total_users} valid users")
        
        # Aggregate results
        user_eers = []
        all_genu = []
        all_forg = []
        for result in results:
            if result:
                user_eer, genu, forg = result
                user_eers.append(user_eer)
                all_genu.extend(genu)
                all_forg.extend(forg)
        
        # Calculate macro-average (per-user) EER
        if user_eers:
            macro_eer = np.mean(user_eers)
            per_user_eers.append(macro_eer)
            print(f"Trial {trial+1} Macro-average EER: {macro_eer:.4f}")
            
            # Also report micro-average for reference
            micro_eer, _ = calculate_eer(all_genu, all_forg)
            print(f"Trial {trial+1} Micro-average EER: {micro_eer:.4f}")
        
        all_trials_eers.extend(per_user_eers)
        
        trial_time = time.time() - trial_start
        print(f"Trial {trial+1} completed in {trial_time:.2f}s with Macro-avg EER: {macro_eer:.4f}, Micro-avg EER: {micro_eer:.4f}")
    
    total_time = time.time() - phase_start_time
    final_eer = np.mean(all_trials_eers)
    print(f"{name} evaluation completed in {total_time:.2f}s")
    print(f"Final EER (avg over {num_trials} trials): {final_eer:.4f}")
    
    return final_eer

# Data loading functions (unchanged from original)
def download_svc2004(task):
    url = f"http://www.cse.ust.hk/svc2004/Task{task}.zip"
    zip_filename = f"svc2004_task{task}.zip"
    extract_dir = f"svc2004_task{task}"
    if os.path.exists(extract_dir):
        print(f"Directory {extract_dir} already exists.")
        return
    print(f"Downloading SVC2004 Task{task} data from {url}...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        with open(zip_filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    print(f"\rDownload progress: {downloaded/total_size*100:.1f}%", end="")
        print("\nExtracting zip file...")
        with zipfile.ZipFile(zip_filename, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
        print(f"Data for Task{task} extracted to {extract_dir}")
    else:
        warnings.warn(f"Failed to download SVC2004 Task{task}. Status code: {response.status_code}")

def load_svc2004_task1(directory):
    pat = re.compile(r'U(\d+)S(\d+)\.txt', re.IGNORECASE)
    users_data = {}
    for root, _, files in os.walk(directory):
        for fname in files:
            m = pat.match(fname)
            if not m:
                continue
            user_id = int(m.group(1))
            sig_id = int(m.group(2))
            path = os.path.join(root, fname)
            users_data.setdefault(user_id, {'genuine': [], 'forged': []})
            with open(path, 'r') as f:
                _ = f.readline()  # Skip the first line (point count)
                stroke_id = 0
                sig = []
                for line in f:
                    t = line.split()
                    if len(t) < 4:
                        continue
                    # Correct order: x y timestamp button_status
                    x, y = float(t[0]), float(t[1])
                    timestamp = int(t[2])
                    button_status = int(t[3])
                    pt = SignaturePoint(x=x, y=y, pressure=0, timestamp=timestamp, strokeId=stroke_id)
                    sig.append(pt)
                    if button_status == 0:
                        stroke_id += 1
            bucket = 'genuine' if sig_id <= 20 else 'forged'
            users_data[user_id][bucket].append(sig)
    return users_data

def load_svc2004_task2(directory):
    pat = re.compile(r'U(\d+)S(\d+)\.txt', re.IGNORECASE)
    users_data = {}
    for root, _, files in os.walk(directory):
        for fname in files:
            m = pat.match(fname)
            if not m:
                continue
            user_id = int(m.group(1))
            sig_id = int(m.group(2))
            path = os.path.join(root, fname)
            users_data.setdefault(user_id, {'genuine': [], 'forged': []})
            with open(path, 'r') as f:
                _ = f.readline()  # Skip the first line (point count)
                stroke_id = 0
                sig = []
                for line in f:
                    t = line.split()
                    if len(t) < 7:  # Task2 has 7 columns
                        continue
                    # Correct order: x y timestamp button_status azimuth altitude pressure
                    x = float(t[0])
                    y = float(t[1])
                    timestamp = int(t[2])
                    button_status = int(t[3])
                    # azimuth and altitude are ignored
                    pressure = float(t[6])  # Pressure is the 7th column
                    
                    pt = SignaturePoint(x=x, y=y, pressure=pressure, timestamp=timestamp, strokeId=stroke_id)
                    sig.append(pt)
                    if button_status == 0:
                        stroke_id += 1
            bucket = 'genuine' if sig_id <= 20 else 'forged'
            users_data[user_id][bucket].append(sig)
    return users_data

def load_mcyt100(directory):
    pat = re.compile(r'U(\d+)S(\d+)\.txt', re.IGNORECASE)
    users_data = {}
    for root, _, files in os.walk(directory):
        for fname in files:
            m = pat.match(fname)
            if not m:
                continue
            user_id = int(m.group(1))
            sig_id = int(m.group(2))
            path = os.path.join(root, fname)
            users_data.setdefault(user_id, {'genuine': [], 'forged': []})
            with open(path, 'r') as f:
                _ = f.readline()
                sig = []
                for line in f:
                    t = line.split()
                    if len(t) < 4:
                        continue
                    timestamp = int(t[0])
                    x = float(t[1])
                    y = float(t[2])
                    pressure = float(t[3])
                    pt = SignaturePoint(x=x, y=y, pressure=pressure, timestamp=timestamp, strokeId=0)
                    sig.append(pt)
            bucket = 'genuine' if sig_id <= 25 else 'forged'
            users_data[user_id][bucket].append(sig)
    return users_data

if __name__ == "__main__":
    print("Starting signature verification evaluation")
    
    # Add overall timing
    overall_start_time = time.time()
    
    # Download SVC2004 Task1 and Task2 data
    download_svc2004(1)
    download_svc2004(2)
    
    # Use all available cores by default
    n_jobs = -1  # -1 means use all available CPUs
    
    # CHANGE: Enable grid search to match paper's results
    task1_eer = evaluate_dataset("SVC2004_Task1", "svc2004_task1", n_jobs, grid_search=True)
    task1_time = time.time() - overall_start_time
    
    task2_eer = evaluate_dataset("SVC2004_Task2", "svc2004_task2", n_jobs, grid_search=True)
    task2_time = time.time() - task1_time
    
    # Summary
    print("\n=== Final Results ===")
    total_time = time.time() - overall_start_time
    
    if task1_eer is not None:
        print(f"SVC2004 Task1: EER={task1_eer:.4f}, Time={task1_time:.2f}s")
    if task2_eer is not None:
        print(f"SVC2004 Task2: EER={task2_eer:.4f}, Time={task2_time:.2f}s")
    
    print(f"Total evaluation time: {total_time:.2f}s")
    print("Evaluation complete!")
