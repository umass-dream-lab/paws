import numpy as np
import json
import os
import random
import sys 

# 1. Get the absolute path of the current file (D.py)
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Go up one level to the parent directory (Folder A)
parent_dir = os.path.dirname(current_dir)

# 3. Add Folder A to Python's search path
sys.path.append(parent_dir)

# We no longer need assign_weights_resolution for loading samples!
# from dataset_preparation import assign_weights_resolution

# ==========================================
# 1. CONFIGURATION (Edit these paths)
# ==========================================
DATASET_NAME = None # or "student_performance"
SAMPLE_SIZES = [250, 500, 1500, 3000, 4271]

# Path to your raw data text/csv file
RAW_DATA_FILE = None 

def update_sample_sizes_for_dataset(dataset_name):
	global SAMPLE_SIZES
	if dataset_name == "intro":
		SAMPLE_SIZES =[250, 500, 1000, 1500, 3000]
	else:
		SAMPLE_SIZES = [250, 375, 844, 1898, 4271, 9611]


gamma = None

# ==========================================
# 2. MATHEMATICAL FALLBACK GENERATORS
# ==========================================
def generate_mock_raw_data(total_points=20000):
	print("  -> Generating mathematical mock data for RAW (Three Diagonal Clusters)...")
	raw = []
	
	cov = [[0.002, 0.004], [0.004, 0.015]] 
	
	centers = [
		([0.35, 0.40], 0.35),
		([0.55, 0.60], 0.20),
		([0.70, 0.65], 0.35)
	]
	
	for center, proportion in centers:
		num_points = int(total_points * proportion)
		points = np.random.multivariate_normal(center, cov, num_points)
		for x, y in points:
			dist_to_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
			saliency = 1.0 if (0.05 < dist_to_center < 0.15) else random.uniform(0.1, 0.5)
			density_proxy = 1.0 / (dist_to_center + 0.01)
			
			raw.append({
				"x": float(np.clip(x, 0, 1)), 
				"y": float(np.clip(y, 0, 1)), 
				"saliency": float(saliency),
				"density": float(density_proxy),
				"perception": float(saliency + (gamma * density_proxy))
			})
			
	for _ in range(int(total_points * 0.10)):
		raw.append({
			"x": random.uniform(0, 1), 
			"y": random.uniform(0, 1), 
			"saliency": random.uniform(0.7, 1.0),
			"density": random.uniform(0, 0.1),
			"perception": random.uniform(0.7, 1.0)
		})
		
	return raw

def fallback_sample(raw_data, algo, k):
	print(f"  -> Computing mathematical fallback sample for {algo} (k={k})")
	safe_k = min(k, len(raw_data))
	
	if algo == "random":
		sample = random.sample(raw_data, safe_k)
	elif algo == "density_biased":
		sorted_data = sorted(raw_data, key=lambda pt: pt.get("density", 1.0))
		sparse_points = sorted_data[:int(safe_k * 0.7)]
		random_points = random.sample(raw_data, int(safe_k * 0.3))
		sample = sparse_points + random_points
	elif algo == "blue_noise":
		grid_size = int(np.sqrt(safe_k))
		sorted_data = sorted(raw_data, key=lambda pt: (int(pt["x"] * grid_size), int(pt["y"] * grid_size)))
		indices = np.linspace(0, len(sorted_data) - 1, safe_k, dtype=int)
		sample = [sorted_data[i] for i in indices]
	elif algo in ["perception_aware", "perception_aware_with_density"]:
		sorted_data = sorted(raw_data, key=lambda pt: pt["perception" if algo == "perception_aware_with_density" else "saliency"], reverse=True)
		high_saliency_edges = sorted_data[:int(safe_k * 0.6)]
		core_coverage = random.sample(raw_data, int(safe_k * 0.4))
		sample = high_saliency_edges + core_coverage
	else:
		sample = random.sample(raw_data, safe_k)
		
	return sample[:safe_k]

# ==========================================
# 3. FILE LOADERS
# ==========================================
def load_raw_data(filepath, gamma):
	print(f"\nLoading Raw Data: {filepath}")
	if not os.path.exists(filepath):
		print(f"  -> WARNING: File not found!")
		return generate_mock_raw_data()
		
	raw_points = []
	try:
		data = np.genfromtxt(filepath, delimiter=',', skip_header=1) 
		min_perception, max_perception = float('inf'), float('-inf')
		for row in data:
			raw_points.append({
				"x": float(row[0]), 
				"y": float(row[1]), 
				"saliency": float(row[2]), 
				"density": float(row[3]), 
				"perception": float(row[2] + (gamma * row[3]))
			})
			min_perception = min(min_perception, raw_points[-1]["perception"])
			max_perception = max(max_perception, raw_points[-1]["perception"])
		for i in range(0, len(raw_points)):
			if max_perception > min_perception:
				raw_points[i]["perception"] = (raw_points[i]["perception"] - min_perception) / (max_perception - min_perception)
			else:
				raw_points[i]["perception"] = 0.0
		print(filepath, gamma)
		return raw_points
	except Exception as e:
		print(f"  -> ERROR reading CSV: {e}")
		return generate_mock_raw_data()

# ==========================================
# 4. THE DIRECT MAPPING EXTRACTION
# ==========================================
def load_npz_sample(algo_folder, sample_size, raw_data, gamma=None):
	global DATASET_NAME
	if algo_folder == "perception_aware_with_density":
		SEED = "454545"
	elif algo_folder == "vas":
		SEED = "126"
	else:
		SEED = "123"
		
	filepath = os.path.join('.', DATASET_NAME, "sampling_techniques", algo_folder, 'cumulative_saliency', "data", f"repetition_{SEED}", f"{algo_folder}_{sample_size}.npz")
	print(f"Loading NPZ: {filepath}")
	
	if not os.path.exists(filepath):
		print(f"  -> WARNING: File not found!")
		return fallback_sample(raw_data, algo_folder, sample_size)
		
	sample_points = []
	try:
		data = np.load(filepath)['data']
		print(f"  -> Successfully loaded NPZ sample with shape {data.shape} ")
		
		# Build a NumPy array of raw coordinates for ultra-fast distance matching
		raw_coords = np.array([[pt["x"], pt["y"]] for pt in raw_data])
		
		for row in data:
			sx, sy = float(row[0]), float(row[1])
			
			# Vectorized nearest-neighbor search using squared Euclidean distance
			distances = (raw_coords[:, 0] - sx)**2 + (raw_coords[:, 1] - sy)**2
			nearest_idx = np.argmin(distances)
			
			# Extract the exact Ground Truth values from the matched Raw Data point
			matched_raw_pt = raw_data[nearest_idx]
			
			sample_points.append({
				"x": matched_raw_pt["x"],
				"y": matched_raw_pt["y"],
				"saliency": matched_raw_pt["saliency"],
				"density": matched_raw_pt.get("density", 0.0),
				"perception": matched_raw_pt.get("perception", 0.0)
			})
			
		return sample_points
	except Exception as e:
		print(f"  -> ERROR reading NPZ: {e}")
		return fallback_sample(raw_data, algo_folder, sample_size)

# ==========================================
# 5. NORMALIZATION FUNCTION
# ==========================================
def normalize_data(all_datasets):
	raw_points = all_datasets.get("raw", [])
	
	if not raw_points:
		return all_datasets
		
	min_x = min(pt["x"] for pt in raw_points)
	max_x = max(pt["x"] for pt in raw_points)
	min_y = min(pt["y"] for pt in raw_points)
	max_y = max(pt["y"] for pt in raw_points)
	
	min_sal = min(pt["saliency"] for pt in raw_points)
	max_sal = max(pt["saliency"] for pt in raw_points)
	min_den = min(pt.get("density", 0) for pt in raw_points)
	max_den = max(pt.get("density", 1) for pt in raw_points)
	min_per = min(pt.get("perception", 0) for pt in raw_points)
	max_per = max(pt.get("perception", 1) for pt in raw_points)

	def scale(val, v_min, v_max):
		return (val - v_min) / (v_max - v_min) if v_max > v_min else 0

	print("\nNormalizing all coordinates and scores to [0, 1]...")
	for key, dataset in all_datasets.items():
		if key == "raw":
			for pt in dataset:
				pt["x"] = scale(pt["x"], min_x, max_x)
				pt["y"] = scale(pt["y"], min_y, max_y)
				pt["saliency"] = scale(pt["saliency"], min_sal, max_sal)
				pt["density"] = scale(pt.get("density", 0), min_den, max_den)
				pt["perception"] = scale(pt.get("perception", 0), min_per, max_per)
		else:
			for algo, points in dataset.items():
				min_sal_sample = min(pt["saliency"] for pt in points)
				max_sal_sample = max(pt["saliency"] for pt in points)
				for pt in points:
					pt["x"] = scale(pt["x"], min_x, max_x)
					pt["y"] = scale(pt["y"], min_y, max_y)
					pt["saliency"] = scale(pt["saliency"], min_sal_sample, max_sal_sample)

					if "density" in pt: pt["density"] = scale(pt["density"], min_den, max_den)
					if "perception" in pt: pt["perception"] = scale(pt["perception"], min_per, max_per)
					
	return all_datasets

if __name__ == "__main__":
	dataset_names = ['intro', 'epileptic_corr', 'estate_anomalies', 'estate_corr']
	renames = {
		'intro': 'cs_grades',
		'epileptic_corr': 'epileptic_corr',
		'estate_anomalies': 'estate_anomalies', 
		'estate_corr': 'estate_corr'
	}
	gamma_values = [0.9999998807907104, 0.9879446029663086, 0.9574578404426575, 0.9423098564147949]
	merged_data = {}
	for dataset in range(len(dataset_names)): 
		DATASET_NAME = dataset_names[dataset]
		update_sample_sizes_for_dataset(DATASET_NAME)
		print("Starting compilation...")
		RAW_DATA_FILE = os.path.join('.', DATASET_NAME, 'data', 'saliency_and_density_weights.csv')
		gamma = gamma_values[dataset]

		raw_dataset = load_raw_data(RAW_DATA_FILE, gamma)

		compiled_data = {
			"raw": raw_dataset
		}

		print("\nProcessing Sampling Algorithms...")
		for size in SAMPLE_SIZES:
			size_str = str(size)
			compiled_data[size_str] = {
				"random": load_npz_sample("random", size, raw_dataset, gamma),
				"density": load_npz_sample("density_biased", size, raw_dataset, gamma),
				"blue_noise": load_npz_sample("blue_noise", size, raw_dataset, gamma),
				"vas": load_npz_sample("vas", size, raw_dataset, gamma),
				"max_min": load_npz_sample("max_min", size, raw_dataset, gamma),

				"paws": load_npz_sample("perception_aware_with_density", size, raw_dataset, gamma),
				"pawsD": load_npz_sample("perception_aware_with_tester", size, raw_dataset, gamma),

				"naive_saliency": load_npz_sample("perception_aware", size, raw_dataset, gamma) 
			}
		normalize_data(all_datasets=compiled_data)
		merged_data[renames.get(DATASET_NAME, DATASET_NAME)] = compiled_data

	output_filename = os.path.join('.', 'VLDB_2026', 'demo_data.js')
	print(f"\nExporting to {output_filename}...")
	with open(output_filename, 'w') as f:
		f.write("const precomputedData = ")
		json.dump(merged_data, f, separators=(',', ':'))
		f.write(";\n")
		
	print(f"\nSuccess! Generated {(os.path.getsize(output_filename) / 1024 / 1024):.2f} MB of data for the HTML demo.")