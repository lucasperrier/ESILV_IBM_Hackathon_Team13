# AI Agent Instructions for ESILV IBM Hackathon - Drone Fault Detection Project

## Project Overview
This is a **drone fault detection** and **data analysis** project using the DronePropA Motion Trajectories Dataset. The project combines:
- Machine learning for drone fault classification
- Time-series sensor data analysis from commercial drones
- A web-based AI chatbot interface (Vue.js + Watsonx integration planned)

## Critical Dataset Knowledge

### Data Location & Structure
- **Raw data**: Gitignored `.mat` files in `DronePropA Motion Trajectories Dataset for Commercial Drones with Defective Propellers/`
- **Processed data**: `data_processed/` contains preprocessed NumPy arrays and JSON mappings
  - `X_windows.npy` - windowed time-series features
  - `y_fault.npy`, `y_sev.npy`, `y_type.npy` - labels for fault, severity, trajectory type
  - `fault_type_mapping.json` - label encoding mappings
  - `normalization_stats.npz` - feature scaling parameters

### Dataset Naming Convention (Critical for File Selection)
Files follow pattern: `F{fault}_SV{severity}_SP{speed}_t{trajectory}_D{drone}_R{run}.mat`

**Fault Groups (F)**:
- `F0` = Healthy drone
- `F1`, `F2`, `F3` = Faulty drone groups

**Speed Settings (SP)**:
- `SP1` = Max speed (2 m/s, 3.14 rad/s) - fast maneuvers
- `SP2` = Max speed (0.333 m/s, 0.52 rad/s) - slow maneuvers

**Trajectory Types (t)**:
- `t1` = Diagonal motion (1×1m square diagonals)
- `t2` = Square-shaped motion (1×1m square edges)
- `t3` = Ascending/descending steps (0.2m to 0.8m in 0.2m increments)
- `t4` = Ascending/descending direct (0.2m to 0.8m in one shot)
- `t5` = Yawing maneuver (45°, -45°, 90°, -90° rotations)

**Drone ID (D)**: `{1,2,3}` - Healthy: 3 drones, Faulty: drone 1 only
**Repetition (R)**: `{1,2,3}` - Repeated experiments

### MAT File Internal Structure
Each `.mat` file contains three key arrays (use `scipy.io.loadmat()`):

**Commander_data** - Position & reference trajectories:
- Column 0: Time
- Columns 21-24: Measured Position (X, Y, Z) and Yaw angle
- Columns 25-28: Reference Position (X, Y, Z) and Yaw angle
- Columns 33-36: Reference Thrust, Roll, Pitch, Yaw Rate

**QDrone_data** - Sensor readings (54 columns):
- Columns 1-9: IMU #1 (Roll, Pitch, Yaw + rates + accelerations)
- Columns 10-18: IMU #2 (Roll, Pitch, Yaw + rates + accelerations)
- Column 23: Battery Level
- Columns 26-38: Gyroscope & Accelerometer data (2 units each)
- Columns 45: Height sensor
- Columns 46-53: Motor & ESC commands (Front Left/Right, Back Left/Right)

**Stabilizer_data**:
- Column 0: Time
- Column 6: Flight Mode

## Python Environment & Dependencies

### Key Libraries (infer from code)
```python
import scipy.io          # For .mat file loading
import numpy as np       # Array operations on processed data
import pandas as pd      # DataFrame manipulation
import matplotlib.pyplot as plt  # Visualization
```

### Path Configuration Pattern
Use `parameter.py` for centralized paths:
```python
from pathlib import Path
PATH_TO_DATA = Path("D:\\Cours\\A5\\S9\\BI Pipeline\\ESILV_IBM_Hackathon_Team13")
```
**Important**: Windows paths use raw strings or double backslashes.

## Jupyter Notebook Workflow (`jules_ws/exploration.ipynb`)

### Standard Analysis Pattern
1. **Load data** from `data_processed/` (preprocessed) OR raw `.mat` files
2. **Filter by criteria** using filename parsing (fault type, trajectory, speed)
3. **Extract features** using column index mappings (see dataset structure above)
4. **Visualize comparisons**: Healthy vs Faulty, Different trajectories, Speed settings
5. **Compute metrics**: Position error, motor imbalance, trajectory tracking accuracy

### Common Analysis Tasks
- **Trajectory visualization**: Plot X-Y position, 3D paths, altitude over time
- **Fault detection features**: Motor command differences, IMU anomalies, position tracking errors
- **Comparative analysis**: Overlay healthy (F0) vs faulty (F1/F2) trajectories

## Web Interface (`index.html`)

### Architecture
- **Frontend**: Standalone Vue.js 3 (CDN-based, no build step)
- **Styling**: Inline CSS with gradient design system (#667eea to #764ba2)
- **Backend integration**: Placeholder for Watsonx RAG API (see `callRAGAPI()` method)

### To Run Locally
```powershell
# Simple HTTP server (no dependencies)
python -m http.server 8000
# Then open http://localhost:8000/index.html
```

### Integration Points
The chatbot UI is **not yet connected** to the drone analysis backend. When implementing:
- Modify `callRAGAPI()` to query processed drone data
- Could expose analytics via REST API or load preprocessed results
- Consider integrating Watsonx for natural language queries about drone faults

## Project-Specific Conventions

### Data Paths
- **Never hardcode** absolute paths in notebooks - use `parameter.py` or relative paths from `DATA_DIR`
- Preprocessed data in `data_processed/` is the primary source for ML workflows
- Raw `.mat` files are gitignored but needed for exploration/reprocessing

### Visualization Standards
- Use 3D plots (`projection='3d'`) for spatial trajectories
- Overlay measured vs reference trajectories (solid vs dashed lines)
- Color code by fault type: healthy (blue/green) vs faulty (red/orange)
- Include time-series plots for motor commands and IMU readings

### Feature Engineering Pattern
- Extract windows from time-series sensor data (see `X_windows.npy` shape)
- Normalize using saved `normalization_stats.npz` for consistency
- Multi-target classification: fault type, severity level, trajectory type

## Common Pitfalls

1. **Path issues**: Windows backslashes require raw strings (`r"path"`) or double backslashes
2. **MAT file indexing**: Arrays are 0-indexed in Python but documentation may use 1-indexed columns
3. **Missing data**: Not all files have all drones (faulty data only for D1)
4. **Vue.js**: CDN version is used - no npm/build process, all code is in single HTML file

## Quick Start for AI Agents

```python
# Load preprocessed data for ML
import numpy as np
X = np.load('data_processed/X_windows.npy')
y_fault = np.load('data_processed/y_fault.npy')

# Or load raw .mat for exploration
import scipy.io
data = scipy.io.loadmat('path/to/F0_SV0_SP1_t1_D1_R1.mat')
commander = data['Commander_data']
position_x = commander[:, 21]  # Measured X position
```

```python
# Filter files by criteria (pattern from exploration notebook)
def get_files_by_criteria(fault='F0', speed='SP1', trial='t1'):
    # Parse filenames and filter DataFrame
    # Return list of matching .mat files
```

## Next Steps / TODO
- [ ] Connect Vue.js chatbot to Watsonx API
- [ ] Implement RAG pipeline using drone fault documentation
- [ ] Build classification model using preprocessed data
- [ ] Create API endpoint to serve predictions to web UI
