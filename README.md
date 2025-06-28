# DEPTH-GUIDED GRASP
### Robust Multi-Object Grasping Through Depth-Informed Topological Analysis

![Project Banner](./images/banner.png)

## 🎯 Overview

DEPTH-GUIDED GRASP is a novel robotic grasping system that leverages depth image topology for reliable object manipulation in cluttered environments. Unlike traditional methods that rely on RGB data or simple depth thresholds, our approach performs sophisticated topological analysis to identify optimal three-point grasp configurations that adapt to complex object geometries.

### Key Achievements
- **92% success rate** across various object geometries and arrangements
- **285ms average planning time** for real-time robotic applications
- **95% collision-free rate** in multi-object scenarios
- **Superior performance** compared to traditional grasping methods

![System Architecture](./images/system_architecture.png)

## 🚀 Key Features

### 🔍 Topological Depth Analysis
- Identifies "valleys" in depth images as optimal finger placement locations
- Combines gradient and Laplacian analysis for surface geometry characterization
- Uses persistence-based filtering to distinguish meaningful features from noise

![Depth Analysis Visualization](./images/depth_analysis.png)

### 🎯 Multi-Objective Optimization
- Balances grasp stability, collision avoidance, and kinematic feasibility
- Dynamic weight adjustment based on scene complexity
- Generates multiple viable grasp configurations with diversity assurance

![Multiple Grasp Configurations](./images/multiple_grasps.png)

### 🤖 Dynamic Adaptation
- Automatically adjusts to object proximity without recalibration
- Real-time collision detection along grasp trajectories
- Adapts grasp angles and scales based on environmental constraints

![Collision Avoidance](./images/collision_avoidance.png)

### 📐 Three-Point Grasp Planning
- Triangle-based approach for enhanced stability through force closure
- Optimizes for near-equilateral triangles with even force distribution
- Provides resistance to rotational forces during manipulation

![Three-Point Grasp Planning](./images/grasp_planning.png)

## 🛠️ Technical Stack

- **Programming Language:** Python 3.6
- **Framework:** Robot Operating System (ROS) Melodic
- **Computer Vision:** OpenCV 4.2.0
- **Scientific Computing:** NumPy, SciPy
- **Visualization:** Matplotlib
- **Hardware:** Intel RealSense D455 RGB-D Camera
- **OS:** Ubuntu 18.04

## 📋 Requirements

### Hardware
- Intel RealSense D455 RGB-D camera
- Standard desktop computer (Intel i7 CPU, 16GB RAM recommended)
- Robotic manipulator (ROS-compatible)

### Software Dependencies
```bash
# Core dependencies
sudo apt-get install ros-melodic-desktop-full
pip install opencv-python==4.2.0
pip install numpy scipy matplotlib

# RealSense SDK
sudo apt-get install ros-melodic-realsense2-camera
```

## 🚀 Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/depth-guided-grasp.git
cd depth-guided-grasp
```

2. **Set up ROS workspace**
```bash
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src
ln -s /path/to/depth-guided-grasp .
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

3. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure camera parameters**
```bash
# Edit config/camera_params.yaml with your RealSense settings
```

## 🎮 Usage

### Quick Start
```bash
# Launch the complete system
roslaunch depth_guided_grasp complete_system.launch

# Or run individual components
rosrun depth_guided_grasp camera_node.py
rosrun depth_guided_grasp grasp_planning_node.py
```

### Running Experiments
```bash
# Single object grasping
rosrun depth_guided_grasp single_object_demo.py

# Multi-object cluttered scene
rosrun depth_guided_grasp cluttered_scene_demo.py

# Performance evaluation
rosrun depth_guided_grasp evaluate_performance.py
```

![Experimental Results](./images/experimental_results.png)

## 📊 Performance

| Metric | Result | Baseline Comparison |
|--------|--------|-------------------|
| Overall Success Rate | **92%** | +14% vs traditional methods |
| Planning Time (avg) | **285ms** | Real-time capable |
| Collision-Free Rate | **95%** | +27% vs geometric approaches |
| Grasp Stability | **94%** | Superior force closure |

![Performance Comparison](./images/performance_comparison.png)

## 🏗️ System Architecture

The system follows a modular pipeline architecture:

1. **Sensor Input** - RealSense D455 camera data acquisition
2. **Preprocessing** - Frame alignment and depth filtering
3. **Object Detection** - Hough circle transform for target identification
4. **Depth Analysis** - Topological valley detection
5. **Grasp Planning** - Three-point configuration generation
6. **Collision Detection** - Multi-object interference checking
7. **Pose Optimization** - Robot coordinate transformation
8. **Visualization** - Debug and result presentation

![Pipeline Visualization](./images/pipeline.png)

## 🔬 Algorithm Details

### Core Algorithms

**Valley Detection Algorithm**
- Gaussian smoothing for noise reduction
- Local minima identification in depth neighborhoods
- Gradient magnitude and persistence calculation
- Circumference-based filtering for optimal placement

**Multi-Objective Optimization**
- Depth score evaluation
- Surface gradient analysis
- Laplacian-based concavity detection
- Collision penalty integration
- Dynamic weight adjustment

**Triangle Generation**
- Sector-based valley grouping
- Shape quality assessment
- Persistence-based scoring
- Diversity assurance mechanism

![Algorithm Visualization](./images/algorithm_details.png)

## 🧪 Experimental Validation

### Test Scenarios
- **Isolated Objects**: Single object manipulation
- **Paired Objects**: Varying proximity challenges  
- **Multiple Objects**: 3-5 object cluttered scenes
- **Occlusion Cases**: Partially hidden objects
- **Reflective Surfaces**: Challenging material properties

### Comparative Analysis
Evaluated against three baseline methods:
- Circle-Only approach
- Random Triangle placement
- Force Closure optimization

![Experimental Setup](./images/experimental_setup.png)

## 📈 Results & Findings

### Key Findings
1. **Depth-informed grasp points** consistently outperformed geometric approaches
2. **Valley-based finger placement** provided superior contact stability
3. **Multi-objective optimization** effectively balanced competing constraints
4. **Topological persistence** reliably identified optimal contact points
5. **Triangle-based planning** demonstrated excellent adaptability

### Performance Breakdown
- Image preprocessing: 15.8%
- Circle detection: 23.9%
- Depth analysis: 32.3%
- Grasp evaluation: 19.3%
- Coordinate transformation: 8.8%

![Results Visualization](./images/results_breakdown.png)

## 🔮 Future Work

- **Extension to arbitrary shapes** beyond circular objects
- **Dynamic scene understanding** for moving objects
- **Learning-based refinement** through execution feedback
- **Deformable object handling** with material properties
- **Multi-sensor fusion** for enhanced robustness
- **Task-aware grasping** for manipulation planning

## 📄 Citation

If you use this work in your research, please cite:

```bibtex
@article{akhtar2025depthguided,
  title={DEPTH-GUIDED GRASP: Robust Multi-Object Grasping Through Depth-Informed Topological Analysis},
  author={Akhtar, Ayesha},
  year={2025},
  month={May}
}
```

## 📝 License

This project is licensed under the Creative Commons license.

## 👥 Contact

**Ayesha Akhtar**
- LinkedIn: [Your LinkedIn Profile](https://linkedin.com/in/ayeshakhtar209)
- Project Link: [https://github.com/yourusername/depth-guided-grasp](https://github.com/yourusername/depth-guided-grasp)

## 🙏 Acknowledgments

- Intel RealSense team for excellent depth sensing hardware
- ROS community for the robust robotics framework
- OpenCV contributors for computer vision tools
- Research advisors and lab colleagues for valuable feedback

---

⭐ **Star this repository if you find it helpful!**
