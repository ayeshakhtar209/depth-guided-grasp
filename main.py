#!/usr/bin/env python3

import pyrealsense2 as rs
import numpy as np
import cv2
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
# import tf2_ros
# import tf.transformations as tf_trans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from datetime import datetime
from scipy import ndimage
from sklearn.cluster import DBSCAN
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
import matplotlib.cm as cm
import networkx as nx


class ObjectDetectionNode:
    def __init__(self):
        rospy.init_node('depth_guided_grasp')
        self.color_pub = rospy.Publisher('detected_circles_color', Image, queue_size=10)
        self.depth_pub = rospy.Publisher('detected_circles_depth', Image, queue_size=10)
        self.grasp_pose_pub = rospy.Publisher('grasp_pose', PoseStamped, queue_size=10)
        self.vertices_pub = rospy.Publisher('detected_grasp_vertices', Float32MultiArray, queue_size=9)

        # self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        # Initialize RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Configure streams
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        # Start the pipeline
        self.pipeline_profile = self.pipeline.start(self.config)
        
        # Get depth scale for depth measurements
        self.depth_sensor = self.pipeline_profile.get_device().first_depth_sensor()
        self.depth_scale = self.depth_sensor.get_depth_scale()
        
        # Get camera intrinsics
        self.depth_profile = self.pipeline_profile.get_stream(rs.stream.depth)
        self.color_profile = self.pipeline_profile.get_stream(rs.stream.color)
        
        self.depth_intrinsics = self.pipeline.get_active_profile().get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        self.color_intrinsics = self.pipeline.get_active_profile().get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        
        # Create alignment object
        self.align_to_color = rs.align(rs.stream.color)

        # Define ROI - Region of Interest (adjustable)
        self.roi_x_min, self.roi_x_max = 250, 400
        self.roi_y_min, self.roi_y_max = 150, 300

        # Create directory for results with timestamp
        self.results_dir = "depth_guided_grasp_results"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Create subdirectories for different visualizations
        self.viz_dirs = {
            'raw_data': os.path.join(self.results_dir, 'raw_data'),
            'detection': os.path.join(self.results_dir, 'circle_detection'),
            'gradient': os.path.join(self.results_dir, 'gradient_analysis'),
            'grasp': os.path.join(self.results_dir, '3d_grasp'),
            'comparison': os.path.join(self.results_dir, 'comparison')
        }
        
        for dir_path in self.viz_dirs.values():
            os.makedirs(dir_path, exist_ok=True)

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Flag to trigger processing
        self.process_current_frame = False
        
        # Save the latest frames
        self.latest_color_frame = None
        self.latest_depth_frame = None
        
        # Add debug mode flag
        self.debug_mode = False
        
        # Performance tracking
        self.performance_metrics = {
            'successful_detections': 0,
            'total_attempts': 0,
            'successful_grasps': 0,
            'failed_grasps': 0,
            'grasp_success_rate': 0.0,
            'detection_times': [],
            'planning_times': []
        }
        
        # Methods for performance comparison
        self.comparison_methods = {
            'depth_guided': {'success': 0, 'attempts': 0},
            'circle_only': {'success': 0, 'attempts': 0},
            'random_placement': {'success': 0, 'attempts': 0},
            'force_closure': {'success': 0, 'attempts': 0}
        }
        
        # Initialize performance with simulated data (for visualization)
        self._init_simulated_performance()

    def _init_simulated_performance(self):
        """Initialize with simulated performance data for visualization"""
        # Simulated performance for different methods
        self.comparison_methods = {
            'DEPTH-GUIDED': {'success': 92, 'attempts': 100}, 
            'Circle-Only': {'success': 78, 'attempts': 100},
            'Random Triangle': {'success': 45, 'attempts': 100},
            'Force Closure': {'success': 81, 'attempts': 100}
        }
        
        # Simulated condition performance
        self.condition_performance = {
            'Isolated Objects': {'success': 97, 'attempts': 100},
            'Two Objects': {'success': 92, 'attempts': 100},
            'Multiple Objects': {'success': 85, 'attempts': 100},
            'Occlusion': {'success': 76, 'attempts': 100},
            'Reflective': {'success': 82, 'attempts': 100}
        }

    def numpy_to_imgmsg(self, img, encoding):
        """Convert numpy array to ROS Image message"""
        img_msg = Image()
        img_msg.height = img.shape[0]
        img_msg.width = img.shape[1]
        img_msg.encoding = encoding
        img_msg.is_bigendian = 0

        img_msg.step = img.shape[1] * (3 if encoding == "bgr8" else 2)
        img_msg.data = img.tobytes()
        return img_msg

    def find_valley_points(self, depth_image, circle_center, radius, angle_offset=0, num_angles=20, search_radius_factor=1.2):
        """
        Find optimal valley points around a circle for gripper finger placement.
        Keeps points closer to circle circumference.
        
        Args:
            depth_image: The depth image from the camera (numpy array)
            circle_center: Center coordinates of the detected circle [x, y]
            radius: Radius of the detected circle
            angle_offset: Initial angle offset to start searching from (default: 0)
            num_angles: Number of angles to check (default: 20)
            search_radius_factor: How far to search beyond the circle radius (default: 1.2)
            
        Returns:
            List of 3 points (numpy arrays) representing optimal finger placement locations
        """
        # Create a copy and fill invalid (zero) depth values with maximum depth
        if not isinstance(depth_image, np.ndarray):
            raise TypeError("depth_image must be a numpy array")
        
        depth_copy = np.copy(depth_image).astype(float)
        max_depth = np.max(depth_copy[depth_copy > 0])
        depth_copy[depth_copy == 0] = max_depth
        
        # Apply smoothing to reduce noise
        depth_smooth = cv2.GaussianBlur(depth_copy, (21, 21), 3)
        
        # Calculate image gradients for valley detection
        grad_y = cv2.Sobel(depth_smooth, cv2.CV_64F, 0, 1, ksize=5)
        grad_x = cv2.Sobel(depth_smooth, cv2.CV_64F, 1, 0, ksize=5)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Calculate Laplacian for valley detection (second derivative)
        # Positive Laplacian values in a depth image usually indicate valleys
        laplacian = cv2.Laplacian(depth_smooth, cv2.CV_64F)
        
        # Define search region - MODIFIED: keep points closer to circumference
        min_search_radius = radius * 0.95  # Stay near the circle boundary
        max_search_radius = radius * search_radius_factor  # Limit how far we go
        
        best_valley_points = []
        
        # Look for 3 points spread approximately 120 degrees apart
        for i in range(3):
            # Calculate angle for this finger, spaced evenly at 120° intervals with offset
            angle = angle_offset + i * (2 * np.pi / 3)
            
            best_point = None
            best_valley_score = float('inf')
            
            # Search along ray from center at different distances
            for r in np.linspace(min_search_radius, max_search_radius, 15):
                x = int(circle_center[0] + r * np.cos(angle))
                y = int(circle_center[1] + r * np.sin(angle))
                
                # Skip if outside image boundaries
                if not (0 <= x < depth_image.shape[1] and 0 <= y < depth_image.shape[0]):
                    continue
                    
                # Skip if no valid depth data
                if depth_image[y, x] == 0:
                    continue
                    
                # Calculate valley score
                depth_val = depth_smooth[y, x]
                laplacian_val = laplacian[y, x]
                gradient_val = gradient_magnitude[y, x]
                
                # Combined score (lower is better)
                valley_score = depth_val - (0.5 * laplacian_val) + (0.3 * gradient_val)
                
                # Add a penalty for being far from the ideal radius (exactly on circumference)
                distance_from_ideal = abs(r - radius)
                radius_penalty = distance_from_ideal * 10  # Increase weight to keep close to circumference
                
                valley_score += radius_penalty
                
                if valley_score < best_valley_score:
                    best_valley_score = valley_score
                    best_point = np.array([x, y])
            
            # If we found a valid point, add it; otherwise use fallback
            if best_point is not None:
                best_valley_points.append(best_point)
            else:
                # Fallback to a point exactly on circle if no good valley found
                x = int(circle_center[0] + radius * np.cos(angle))
                y = int(circle_center[1] + radius * np.sin(angle))
                best_valley_points.append(np.array([x, y]))
        
        return best_valley_points

    def find_optimal_grasp_triangles(self, circle_center, radius, depth_image, color_image=None, num_poses=3, angle_threshold=0.5):
        """Find multiple optimal grasp triangles placed in depth valleys"""
        all_triangle_configs = []
        
        # Try multiple angle offsets to find diverse triangle configurations
        for angle_offset in np.linspace(0, 2 * np.pi / 3, 10):  # Try 10 different offsets
            # Find valley points around the circle
            valley_points = self.find_valley_points(
                depth_image, 
                circle_center, 
                radius, 
                angle_offset, 
                search_radius_factor=1.2
            )
            
            if len(valley_points) < 3:
                continue  # Skip if we couldn't find enough points
                
            # Calculate score based on multiple factors
            # 1. Evaluate depth at the points
            depth_score = self.evaluate_grasp_points(depth_image, valley_points, 
                                                   cv2.GaussianBlur(depth_image.astype(float), (15, 15), 0))
            
            # 2. Check for collisions 
            collision_count = self.detect_potential_collisions(valley_points, depth_image, circle_center, radius)
            
            # 3. Check for point clearance
            clearance_count = sum(1 for p in valley_points if self.check_point_clearance(p, depth_image))
            
            # Calculate adjusted score
            collision_penalty = collision_count * 2000
            clearance_bonus = 0.5 ** clearance_count  # Lower is better
            
            adjusted_score = depth_score * clearance_bonus + collision_penalty
            
            # Add a penalty for points too far from circle circumference
            circle_adherence_penalty = 0
            for point in valley_points:
                dist_to_center = np.sqrt(np.sum((point - circle_center)**2))
                diff_from_radius = abs(dist_to_center - radius)
                circle_adherence_penalty += diff_from_radius * 100  # Higher penalty to enforce staying on circle
                
            adjusted_score += circle_adherence_penalty
            
            # Add to configurations
            triangle_shape = self.calculate_triangle_shape_signature(valley_points)
            all_triangle_configs.append((valley_points, angle_offset, adjusted_score, clearance_count, triangle_shape))
        
        # Sort configurations by score (lower is better)
        all_triangle_configs.sort(key=lambda x: x[2])
        
        # Select diverse top configurations
        selected_configs = []
        shape_signatures = set()
        
        for config in all_triangle_configs:
            points, angle, score, quality, shape_sig = config
            
            # Check if this shape is sufficiently different from already selected ones
            is_unique = True
            for existing_sig in shape_signatures:
                similarity = self.calculate_shape_similarity(shape_sig, existing_sig)
                if similarity > 0.7:  # If more than 70% similar, consider it a duplicate
                    is_unique = False
                    break
            
            if is_unique and len(selected_configs) < num_poses:
                selected_configs.append((points, angle, score, quality))
                shape_signatures.add(shape_sig)
                
                if len(selected_configs) >= num_poses:
                    break
        
        # If we couldn't find enough diverse configurations, add more from the sorted list
        while len(selected_configs) < num_poses and len(all_triangle_configs) > len(selected_configs):
            next_idx = len(selected_configs)
            if next_idx < len(all_triangle_configs):
                points, angle, score, quality, _ = all_triangle_configs[next_idx]
                selected_configs.append((points, angle, score, quality))
        
        return selected_configs

    def find_optimal_valley_triangle(self, depth_image, circle_center, radius, other_circles=None, num_configs=3):
        """Find optimal triangle with vertices in depth valleys"""
        # Detect valleys topologically
        valleys = self.detect_valleys_topological(depth_image, circle_center, radius)
        
        # Filter out valleys that collide with other circles
        if other_circles is not None:
            filtered_valleys = []
            for valley in valleys:
                collides = False
                for other_circle, _ in other_circles:
                    cx, cy, r = other_circle
                    # Check if valley center point is inside or too close to other circle
                    dist = np.sqrt((valley['center'][0] - cx)**2 + (valley['center'][1] - cy)**2)
                    if dist < r + 5:  # 5px safety margin
                        collides = True
                        break
                if not collides:
                    filtered_valleys.append(valley)
            valleys = filtered_valleys

        if len(valleys) < 3:
            # Fall back to original method if not enough valleys found
            return self.find_optimal_grasp_triangles(circle_center, radius, depth_image, num_poses=num_configs)
        
        # Group valleys by angle sectors around the circle
        angle_sectors = 8
        sector_valleys = [[] for _ in range(angle_sectors)]
        
        for valley in valleys:
            # Calculate angle relative to circle center
            dx = valley['center'][0] - circle_center[0]
            dy = valley['center'][1] - circle_center[1]
            angle = np.arctan2(dy, dx)
            
            # Calculate distance from circle center
            dist = np.sqrt(dx*dx + dy*dy)
            
            # Skip valleys that are too far from the circle circumference
            if dist < radius * 0.9 or dist > radius * 1.3:
                continue
                
            # Map angle to sector (0 to angle_sectors-1)
            sector = int(((angle + np.pi) / (2 * np.pi)) * angle_sectors) % angle_sectors
            sector_valleys[sector].append(valley)
        
        # Try to find diverse triangle configurations using valleys from different sectors
        triangle_configs = []
        
        # Try various combinations of sectors to get diverse triangles
        sector_combinations = []
        for i in range(angle_sectors):
            # Try to find sectors roughly 120 degrees apart
            s1 = i
            s2 = (i + angle_sectors//3) % angle_sectors
            s3 = (i + 2*angle_sectors//3) % angle_sectors
            sector_combinations.append((s1, s2, s3))
        
        # Add some random combinations for diversity
        for _ in range(5):
            sectors = np.random.choice(angle_sectors, 3, replace=False)
            sector_combinations.append(tuple(sectors))
        
        # Generate triangle configs from sector combinations
        for s1, s2, s3 in sector_combinations:
            if not (sector_valleys[s1] and sector_valleys[s2] and sector_valleys[s3]):
                continue
                
            # Get best valleys from each sector
            v1 = sector_valleys[s1][0]  # Already sorted by persistence
            v2 = sector_valleys[s2][0]
            v3 = sector_valleys[s3][0]
            
            triangle_points = [
                np.array(v1['center']),
                np.array(v2['center']),
                np.array(v3['center'])
            ]
            
            # Calculate score based on valley quality and triangle geometry
            persistence_score = v1['persistence'] + v2['persistence'] + v3['persistence']
            
            # Check triangle shape (prefer equilateral)
            sides = []
            for i in range(3):
                p1 = triangle_points[i]
                p2 = triangle_points[(i+1)%3]
                sides.append(np.sqrt(np.sum((p2-p1)**2)))
            
            sides.sort()
            shape_score = sides[0] / max(sides[2], 1e-6)  # Ratio of shortest to longest side
            
            # Check for collisions
            collision_count = self.detect_potential_collisions(triangle_points, depth_image, circle_center, radius)
            
            # Combined score (higher is better)
            combined_score = (persistence_score * shape_score) / (1 + collision_count * 10)
            
            triangle_configs.append((triangle_points, combined_score))
        
        # Sort by score (highest first)
        triangle_configs.sort(key=lambda x: x[1], reverse=True)
        
        # Return top configs
        result_configs = []
        for i in range(min(num_configs, len(triangle_configs))):
            points, score = triangle_configs[i]
            # Calculate angle for compatibility with existing code
            angle = np.arctan2(points[0][1] - circle_center[1], points[0][0] - circle_center[0])
            result_configs.append((points, angle, -score, 3))  # Negate score as existing code expects lower=better
        
        return result_configs

    def get_valid_depth(self, depth_image, center_x, center_y, radius):
        """Get valid depth value at or near the specified point"""
        center_depth = depth_image[center_y, center_x]
        if center_depth > 0:
            return center_depth * self.depth_scale

        valid_depths = []
        for r in range(1, min(radius, 10)):
            for theta in np.linspace(0, 2 * np.pi, 8 * r):
                x = int(center_x + r * np.cos(theta))
                y = int(center_y + r * np.sin(theta))

                if 0 <= x < depth_image.shape[1] and 0 <= y < depth_image.shape[0]:
                    depth_val = depth_image[y, x]
                    if depth_val > 0:
                        valid_depths.append(depth_val)

        return np.median(valid_depths) * self.depth_scale if valid_depths else 0.5

    def compute_local_minima_map(self, depth_image):
        """Compute a map of local minima in the depth image"""
        depth_copy = depth_image.copy().astype(float)
        max_depth = np.max(depth_copy[depth_copy > 0])
        depth_copy[depth_copy == 0] = max_depth

        depth_smooth = cv2.GaussianBlur(depth_copy, (15, 15), 0)

        if len(depth_smooth.shape) > 2:
            if depth_smooth.shape[2] == 3:
                depth_smooth = cv2.cvtColor(depth_smooth.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(float)
            else:
                depth_smooth = depth_smooth[:, :, 0]

        height, width = depth_smooth.shape[:2]

        grad_y = cv2.Sobel(depth_smooth, cv2.CV_64F, 0, 1, ksize=5)
        grad_x = cv2.Sobel(depth_smooth, cv2.CV_64F, 1, 0, ksize=5)

        gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

        # Normalize gradient for better visualization
        normalized_gradient = gradient_magnitude / (np.max(gradient_magnitude) + 1e-6)

        # Enhanced local minima detection
        minima_indicator = np.zeros_like(depth_smooth)
        kernel_size = 9  # Smaller kernel for finer detail
        half_kernel = kernel_size // 2

        for y in range(half_kernel, height - half_kernel):
            for x in range(half_kernel, width - half_kernel):
                if depth_image[y, x] == 0:
                    continue

                center_depth = depth_smooth[y, x]
                region = depth_smooth[y - half_kernel:y + half_kernel + 1, x - half_kernel:x + half_kernel + 1]

                # Consider a point a local minimum if it's among the lowest 5% in its neighborhood
                is_minimum = center_depth <= np.percentile(region, 5)

                if is_minimum:
                    minima_indicator[y, x] = 1

        return minima_indicator, normalized_gradient, depth_smooth
    
    def evaluate_grasp_points(self, depth_image, points, depth_smooth):
        """Evaluate grasp points based on depth and gradient values"""
        depth_values = []
        grad_values = []

        for point in points:
            x, y = int(point[0]), int(point[1])

            # Ensure coordinates are within image bounds
            if 0 <= y < depth_image.shape[0] and 0 <= x < depth_image.shape[1]:
                if depth_image[y, x] > 0:
                    depth_values.append(depth_smooth[y, x])

                    kernel_size = 5
                    half_kernel = kernel_size // 2

                    if half_kernel <= y < depth_image.shape[0] - half_kernel and half_kernel <= x < depth_image.shape[1] - half_kernel:
                        region = depth_smooth[y - half_kernel:y + half_kernel + 1, x - half_kernel:x + half_kernel + 1]
                        center_val = depth_smooth[y, x]

                        if center_val <= np.min(region):
                            grad_values.append(0)
                        else:
                            grad_y = cv2.Sobel(region, cv2.CV_64F, 0, 1, ksize=3)
                            grad_x = cv2.Sobel(region, cv2.CV_64F, 1, 0, ksize=3)
                            gradient_magnitude = np.sqrt(
                                grad_x[half_kernel, half_kernel] ** 2 + grad_y[half_kernel, half_kernel] ** 2)
                            grad_values.append(gradient_magnitude)
                    else:
                        grad_values.append(1000)  # Use a large number but not infinity
                else:
                    depth_values.append(1000)  # Use a large number but not infinity
                    grad_values.append(1000)
            else:
                depth_values.append(1000)  # Use a large number but not infinity
                grad_values.append(1000)

        # If we didn't get any valid values, return a high but not infinite score
        if not depth_values or not grad_values:
            return 10000

        score = np.sum(depth_values) + np.sum(grad_values)
        return score

    def check_point_clearance(self, point, depth_image, color_image=None, radius=10):
        """Check if a point has enough clearance around it for a gripper finger"""
        x, y = int(point[0]), int(point[1])

        # Make sure point is within image bounds
        if not (0 <= x < depth_image.shape[1] and 0 <= y < depth_image.shape[0]):
            return False

        # Get the depth at this point
        point_depth = depth_image[y, x] * self.depth_scale if depth_image[y, x] > 0 else 0

        # If the point has no valid depth, it's not valid
        if point_depth == 0:
            return False

        # Check surrounding area for depth discontinuities
        clearance_score = 0
        valid_points = 0

        for r in range(1, radius):
            for theta in np.linspace(0, 2 * np.pi, 8 * r):
                check_x = int(x + r * np.cos(theta))
                check_y = int(y + r * np.sin(theta))

                if 0 <= check_x < depth_image.shape[1] and 0 <= check_y < depth_image.shape[0]:
                    check_depth = depth_image[check_y, check_x] * self.depth_scale if depth_image[check_y, check_x] > 0 else 0

                    if check_depth > 0:
                        # If depth difference is significant, we have an edge
                        depth_diff = abs(point_depth - check_depth)
                        if depth_diff > 0.003:  # Even lower threshold - 3mm
                            clearance_score += 1
                        valid_points += 1

        # Calculate percentage of points that indicate clearance
        clearance_percentage = clearance_score / valid_points if valid_points > 0 else 0

        # Reduced threshold to detect more potential grasp points
        return clearance_percentage > 0.1

    def line_circle_intersection(self, p1, p2, circle_center, radius):
        """
        Check if a line segment intersects with a circle
        
        Args:
            p1, p2: Line segment endpoints (numpy arrays)
            circle_center: Circle center coordinates (numpy array)
            radius: Circle radius
            
        Returns:
            True if there is an intersection, False otherwise
        """
        # Vector from p1 to p2
        d = p2 - p1
        # Vector from p1 to circle center
        f = p1 - circle_center

        # Calculate quadratic equation coefficients
        a = np.dot(d, d)
        b = 2 * np.dot(f, d)
        c = np.dot(f, f) - radius * radius

        discriminant = b * b - 4 * a * c
        
        if discriminant < 0:
            # No intersection
            return False
        else:
            # Calculate intersection parameters
            discriminant = np.sqrt(discriminant)
            t1 = (-b - discriminant) / (2 * a)
            t2 = (-b + discriminant) / (2 * a)
            
            # Check if intersection is within line segment
            if (0 <= t1 <= 1) or (0 <= t2 <= 1):
                return True
            
            return False

    def detect_potential_collisions(self, points, depth_image, circle_center, radius):
        """Check if there are potential collisions between grasp points"""
        collisions = 0

        # Convert points to homogeneous coordinates for line calculations
        hom_points = []
        for point in points:
            hom_points.append(np.array([point[0], point[1], 1]))

        # Check for collisions along each side of the triangle
        for i in range(3):
            p1 = hom_points[i]
            p2 = hom_points[(i + 1) % 3]

            # Calculate line equation ax + by + c = 0
            line = np.cross(p1, p2)
            a, b, c = line

            # Sample points along the line segment
            num_samples = 40  # Increased sampling for more thorough collision detection
            for t in np.linspace(0, 1, num_samples):
                # Linear interpolation between points
                x = int(p1[0] * (1 - t) + p2[0] * t)
                y = int(p1[1] * (1 - t) + p2[1] * t)

                if 0 <= x < depth_image.shape[1] and 0 <= y < depth_image.shape[0]:
                    # Check if this point is outside the circle
                    dx = x - circle_center[0]
                    dy = y - circle_center[1]
                    dist_to_center = np.sqrt(dx * dx + dy * dy)

                    if dist_to_center > radius * 0.65:  # Check more of the triangle
                        current_depth = depth_image[y, x] * self.depth_scale if depth_image[y, x] > 0 else 0

                        # Check nearby depths to see if there's an object in the way
                        if current_depth > 0:
                            surrounding_depths = []
                            for dy in [-3, -2, -1, 0, 1, 2, 3]:  # Wider sampling window
                                for dx in [-3, -2, -1, 0, 1, 2, 3]:
                                    nx, ny = x + dx, y + dy
                                    if 0 <= nx < depth_image.shape[1] and 0 <= ny < depth_image.shape[0]:
                                        d = depth_image[ny, nx] * self.depth_scale if depth_image[ny, nx] > 0 else 0
                                        if d > 0:
                                            surrounding_depths.append(d)

                            if surrounding_depths:
                                avg_depth = np.mean(surrounding_depths)

                                # If this point has significantly different depth than grasp points,
                                # it's likely a collision with another object
                                if abs(avg_depth - current_depth) < 0.002:  # More sensitive collision detection
                                    collisions += 1

        return collisions

    def calculate_triangle_shape_signature(self, points):
        """
        Calculate a signature for the triangle shape to help identify unique configurations.
        
        This function creates a rotation-invariant signature based on the side lengths
        and angles of the triangle, which can be used to compare different triangles
        and determine if they represent similar shapes.
        
        Args:
            points: List of 3 points (numpy arrays) representing the triangle vertices
            
        Returns:
            A signature tuple containing sorted side lengths and angles
        """
        # Calculate side lengths
        sides = []
        for i in range(3):
            p1 = points[i]
            p2 = points[(i + 1) % 3]
            dist = np.sqrt(np.sum((p2 - p1) ** 2))
            sides.append(dist)

        # Sort sides to make signature rotation-invariant
        sides.sort()

        # Calculate angles
        angles = []
        for i in range(3):
            p1 = points[i]
            p2 = points[(i - 1) % 3]
            p3 = points[(i + 1) % 3]

            v1 = p2 - p1
            v2 = p3 - p1

            dot = np.dot(v1, v2)
            norm = np.linalg.norm(v1) * np.linalg.norm(v2)

            if norm > 0:
                cos_angle = max(-1, min(1, dot / norm))
                angle = np.arccos(cos_angle)
                angles.append(angle)

        # Sort angles to make signature rotation-invariant
        angles.sort()

        # Combine sides and angles into a shape signature
        signature = tuple(sides + angles)
        return signature

    def calculate_shape_similarity(self, sig1, sig2):
        """
        Calculate similarity between two triangle shape signatures
        
        Args:
            sig1, sig2: Shape signatures as returned by calculate_triangle_shape_signature
            
        Returns:
            Similarity score between 0 and 1, where 1 means identical shapes
        """
        # Ensure both signatures have the same length
        if len(sig1) != len(sig2):
            return 0.0
        
        # Normalize differences by the maximum value in each signature
        max_val1 = max(sig1)
        max_val2 = max(sig2)
        max_val = max(max_val1, max_val2)
        
        if max_val == 0:
            return 1.0  # Both signatures are all zeros
        
        # Calculate normalized Euclidean distance
        squared_diff_sum = 0
        for i in range(len(sig1)):
            # Normalize each value by the maximum
            norm_val1 = sig1[i] / max_val
            norm_val2 = sig2[i] / max_val
            squared_diff_sum += (norm_val1 - norm_val2) ** 2
        
        distance = np.sqrt(squared_diff_sum / len(sig1))
        
        # Convert distance to similarity (1 - normalized distance)
        similarity = 1.0 - min(1.0, distance)
        
        return similarity

    def visualize_grasp(self, color_image, circle_center, circle_radius, grasp_points, depth_value, label_suffix="", color=(0, 255, 0)):
        """Create a visualization of the grasp"""
        vis_image = color_image.copy()

        cv2.circle(vis_image, (int(circle_center[0]), int(circle_center[1])), int(circle_radius), (255, 0, 0), 2)

        for i, point in enumerate(grasp_points):
            point = point.astype(int)
            cv2.circle(vis_image, (point[0], point[1]), 5, color, -1)
            cv2.putText(vis_image, f"F{i + 1}{label_suffix}", (point[0] + 10, point[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        points = [p.astype(int) for p in grasp_points]
        for i in range(3):
            cv2.line(vis_image, (points[i][0], points[i][1]), (points[(i + 1) % 3][0], points[(i + 1) % 3][1]), color, 2)

        grasp_center = circle_center.astype(int)
        for point in grasp_points:
            point = point.astype(int)
            cv2.line(vis_image, (point[0], point[1]), (grasp_center[0], grasp_center[1]), color, 1)

        cv2.circle(vis_image, (grasp_center[0], grasp_center[1]), 8, (0, 0, 255), -1)

        cv2.putText(vis_image,
                    f"Depth: {depth_value:.3f}m",
                    (grasp_center[0] - 60, grasp_center[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        return vis_image

    def detect_valleys_topological(self, depth_image, circle_center, radius, search_radius_factor=3.0):
        """
        Detect valleys in the depth image using topological persistence
        
        Args:
            depth_image: The depth image
            circle_center: Center of the detected circle [x, y]
            radius: Radius of the detected circle
            search_radius_factor: How far to search beyond circle radius
            
        Returns:
            List of valley dictionaries with center coordinates and persistence values
        """
        try:
            # Debug information
            rospy.loginfo(f"detect_valleys_topological: depth_image shape: {depth_image.shape}, dtype: {depth_image.dtype}")
            rospy.loginfo(f"detect_valleys_topological: circle_center: {circle_center}, radius: {radius}")
            
            # Create a copy and fill invalid points
            depth_copy = depth_image.copy().astype(float)
            max_depth = np.max(depth_copy[depth_copy > 0])
            depth_copy[depth_copy == 0] = max_depth
            
            # Apply smoothing
            depth_smooth = cv2.GaussianBlur(depth_copy, (15, 15), 0)
            
            # Ensure depth_smooth is 2D
            if len(depth_smooth.shape) > 2:
                rospy.logwarn(f"depth_smooth has shape {depth_smooth.shape}, converting to 2D")
                if depth_smooth.shape[2] == 3:
                    depth_smooth = cv2.cvtColor(depth_smooth.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(float)
                else:
                    depth_smooth = depth_smooth[:, :, 0]
                    
            rospy.loginfo(f"After conversion, depth_smooth shape: {depth_smooth.shape}")
            
            # Define region of interest around the circle
            # Check if circle_center is valid
            if not isinstance(circle_center, np.ndarray) or len(circle_center) < 2:
                rospy.logerr(f"Invalid circle_center: {circle_center}")
                return []
                
            # Ensure circle center coordinates are valid numbers
            cx, cy = int(circle_center[0]), int(circle_center[1])
            
            # Check if cx and cy are valid coordinates
            if cx <= 0 or cy <= 0 or cx >= depth_smooth.shape[1] or cy >= depth_smooth.shape[0]:
                rospy.logerr(f"Circle center coordinates out of bounds: cx={cx}, cy={cy}, " +
                            f"image dimensions: {depth_smooth.shape}")
                return []
                
            # Check if radius is valid
            if radius <= 0:
                rospy.logerr(f"Invalid radius: {radius}")
                return []
                
            search_radius = int(radius * search_radius_factor)
            
            # Extract ROI with bounds checking
            x_min = max(0, cx - search_radius)
            x_max = min(depth_smooth.shape[1], cx + search_radius)
            y_min = max(0, cy - search_radius)
            y_max = min(depth_smooth.shape[0], cy + search_radius)
            
            # Check ROI dimensions
            if x_min >= x_max or y_min >= y_max:
                rospy.logerr(f"Invalid ROI dimensions: x_min={x_min}, x_max={x_max}, y_min={y_min}, y_max={y_max}")
                return []
                
            roi = depth_smooth[y_min:y_max, x_min:x_max]
            rospy.loginfo(f"ROI shape: {roi.shape}")
            
            # Alternative approach: use a simpler algorithm instead of image processing functions
            # that might be causing dimension errors
            valleys = []
            kernel_size = 11
            half_kernel = kernel_size // 2
            
            # Iterate through ROI manually to find local minima
            for y in range(half_kernel, roi.shape[0] - half_kernel):
                for x in range(half_kernel, roi.shape[1] - half_kernel):
                    # Get the neighborhood
                    neighborhood = roi[y - half_kernel:y + half_kernel + 1, 
                                       x - half_kernel:x + half_kernel + 1]
                    
                    # Check if this point is a local minimum
                    if roi[y, x] <= np.min(neighborhood) + 0.001:  # Small tolerance
                        # Convert back to original image coordinates
                        orig_x = x + x_min
                        orig_y = y + y_min
                        
                        # Calculate distance from circle center
                        distance_from_center = np.sqrt((orig_x - cx)**2 + (orig_y - cy)**2)
                        
                        # Only consider points near the circle circumference
                        # The original search was too wide - now we only look from 0.9*radius to 1.2*radius
                        if distance_from_center < radius * 0.9 or distance_from_center > radius * 1.2:
                            continue
                        
                        # Calculate "persistence" (valley depth)
                        # Use a small window to calculate gradient magnitude around this point
                        window_size = 5
                        window_half = window_size // 2
                        
                        # Ensure window is within ROI bounds
                        if (y - window_half >= 0 and y + window_half + 1 <= roi.shape[0] and
                            x - window_half >= 0 and x + window_half + 1 <= roi.shape[1]):
                            
                            window = roi[y - window_half:y + window_half + 1,
                                        x - window_half:x + window_half + 1]
                            
                            # Calculate gradient magnitude
                            grad_y = np.gradient(window, axis=0)
                            grad_x = np.gradient(window, axis=1)
                            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
                            persistence = np.mean(gradient_magnitude)
                            
                            # Add valley information
                            valleys.append({
                                'center': (orig_x, orig_y),
                                'depth': roi[y, x],
                                'persistence': persistence,
                                'distance': distance_from_center
                            })
            
            # Sort valleys by persistence (higher first)
            valleys.sort(key=lambda v: v['persistence'], reverse=True)
            
            rospy.loginfo(f"Found {len(valleys)} valleys")
            return valleys
        
        except Exception as e:
            rospy.logerr(f"Error in detect_valleys_topological: {e}")
            import traceback
            rospy.logerr(f"Traceback: {traceback.format_exc()}")
            return []  # Return empty list on error

    def plot_gradient_descent_3d(self, depth_image, circle_center, grasp_points, radius, 
                                elev=30, azim=45, reverse_view=True):
        """
        Create a 3D visualization of the grasp points on the depth surface with top view
        
        Args:
            depth_image: Depth image to visualize
            circle_center: Center coordinates of the detected circle [x, y]
            grasp_points: List of 3 points for the grasp triangle
            radius: Radius of the detected circle
            elev: Elevation angle for 3D view (default: 30)
            azim: Azimuth angle for 3D view (default: 45)
            reverse_view: If True, reverse the view direction by adding 180° to azimuth (default: True)
            
        Returns:
            Matplotlib figure with 3D visualization
        """
        # Create a figure with two subplots - 3D view and top-down view
        fig = plt.figure(figsize=(18, 10))
        
        # 3D view subplot
        ax1 = fig.add_subplot(121, projection='3d')
        
        # Top-down view subplot
        ax2 = fig.add_subplot(122)
        
        # Prepare data
        y, x = np.mgrid[0:depth_image.shape[0], 0:depth_image.shape[1]]
        
        depth_copy = depth_image.copy().astype(float)
        max_depth = np.max(depth_copy[depth_copy > 0])
        depth_copy[depth_copy == 0] = max_depth
        
        depth_smooth = cv2.GaussianBlur(depth_copy, (15, 15), 0)
        
        # For 3D surface plot
        stride = 8
        x_sub = x[::stride, ::stride]
        y_sub = y[::stride, ::stride]
        z_sub = depth_smooth[::stride, ::stride]
        
        # Convert depth to meters for better visualization
        z_sub_meters = z_sub * self.depth_scale
        
        # Apply the depth range filter (486mm to 1500mm)
        z_sub_meters = np.clip(z_sub_meters, 0.486, 1.5)
        
        # 3D surface plot
        surf = ax1.plot_surface(x_sub, y_sub, z_sub_meters, cmap='jet', alpha=0.7, 
                               linewidth=0, antialiased=True)
        
        # Plot circle center and grasp points on 3D plot
        center_depth = depth_smooth[int(circle_center[1]), int(circle_center[0])] * self.depth_scale
        ax1.scatter(circle_center[0], circle_center[1], center_depth,
                   color='blue', s=100, marker='o', label='Circle Center')
        
        for i, point in enumerate(grasp_points):
            point_y, point_x = int(point[1]), int(point[0])
            if 0 <= point_y < depth_smooth.shape[0] and 0 <= point_x < depth_smooth.shape[1]:
                point_depth = depth_smooth[point_y, point_x] * self.depth_scale
                ax1.scatter(point[0], point[1], point_depth,
                           color='red', s=100, marker='^', 
                           label=f'Grasp Point {i + 1}' if i == 0 else "")
        
        # Set the view angle
        if reverse_view:
            # Reverse the view by adding 180° to azimuth
            ax1.view_init(elev=elev, azim=(azim + 180) % 360)
        else:
            ax1.view_init(elev=elev, azim=azim)
        
        # Add a wireframe to better visualize the surface structure
        ax1.plot_wireframe(x_sub, y_sub, z_sub_meters, color='gray', alpha=0.2, linewidth=0.5)
        
        # Create a fill_between for the grasp triangle to make it more visible
        if len(grasp_points) >= 3:
            triangle_xs = [grasp_points[0][0], grasp_points[1][0], grasp_points[2][0], grasp_points[0][0]]
            triangle_ys = [grasp_points[0][1], grasp_points[1][1], grasp_points[2][1], grasp_points[0][1]]
            triangle_zs = []
            
            for i in range(3):
                point_y, point_x = int(grasp_points[i][1]), int(grasp_points[i][0])
                if 0 <= point_y < depth_smooth.shape[0] and 0 <= point_x < depth_smooth.shape[1]:
                    point_depth = depth_smooth[point_y, point_x] * self.depth_scale
                    triangle_zs.append(point_depth)
                else:
                    triangle_zs.append(center_depth)  # Fallback to center depth
            
            # Add closing point to complete the polygon
            triangle_zs.append(triangle_zs[0])
            
            # Plot the triangle in 3D space
            ax1.plot(triangle_xs, triangle_ys, triangle_zs, 'r-', linewidth=3)
        
        ax1.set_xlabel('X (px)')
        ax1.set_ylabel('Y (px)')
        ax1.set_zlabel('Depth (m)')
        ax1.set_title('Grasp Points on Depth Surface (3D View)')
        
        # Set better axis ranges based on the circle and some margin
        margin = radius * 3
        ax1.set_xlim([circle_center[0] - margin, circle_center[0] + margin])
        ax1.set_ylim([circle_center[1] - margin, circle_center[1] + margin])
        
        # Top-down view with contour plot
        contour = ax2.contourf(x, y, depth_smooth * self.depth_scale, 30, cmap='jet')
        
        # Draw circle on top-down view
        circle_patch = plt.Circle((circle_center[0], circle_center[1]), radius, 
                                 fill=False, edgecolor='black', linewidth=2)
        ax2.add_patch(circle_patch)
        
        # Plot grasp points on top-down view
        ax2.scatter(circle_center[0], circle_center[1], color='blue', s=100, marker='o', label='Circle Center')
        
        # Draw triangle
        points = np.array([p for p in grasp_points])
        triangle = plt.Polygon(points, fill=False, edgecolor='white', linewidth=2)
        ax2.add_patch(triangle)
        
        for i, point in enumerate(grasp_points):
            ax2.scatter(point[0], point[1], color='red', s=100, marker='^')
            ax2.text(point[0]+5, point[1]+5, f'P{i+1}', color='white', fontsize=12)
        
        ax2.set_xlabel('X (px)')
        ax2.set_ylabel('Y (px)')
        ax2.set_title('Depth Map with Grasp Triangle (Top View)')
        ax2.set_xlim(circle_center[0] - margin, circle_center[0] + margin)
        ax2.set_ylim(circle_center[1] - margin, circle_center[1] + margin)
        
        # Add colorbar
        cbar = fig.colorbar(contour, ax=ax2, shrink=0.7, aspect=10)
        cbar.set_label('Depth (m)')
        fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=10, label='Depth (m)')
        
        # Add a text description of the view angle settings
        view_text = f"View Settings: Elevation = {elev}°, Azimuth = {azim + 180 if reverse_view else azim}°"
        fig.text(0.5, 0.01, view_text, ha='center', fontsize=12)
        
        plt.tight_layout()
        
        return fig

    # def calculate_ros_grasp_pose(self, circle_center, depth_value, grasp_points):
    #     """Calculate a ROS PoseStamped message for the grasp"""
    #     z = depth_value
    #     x = (circle_center[0] - self.color_intrinsics.ppx) * z / self.color_intrinsics.fx
    #     y = (circle_center[1] - self.color_intrinsics.ppy) * z / self.color_intrinsics.fy

    #     pose_msg = PoseStamped()
    #     pose_msg.header.stamp = rospy.Time.now()
    #     pose_msg.header.frame_id = "camera_color_optical_frame"

    #     pose_msg.pose.position.x = x
    #     pose_msg.pose.position.y = y
    #     pose_msg.pose.position.z = z

    #     # Calculate orientation based on the grasp triangle
    #     # Get normal vector to grasp triangle plane
    #     v1 = grasp_points[1] - grasp_points[0]
    #     v2 = grasp_points[2] - grasp_points[0]
        
    #     # Cross product gives normal vector
    #     normal = np.cross(np.append(v1, 0), np.append(v2, 0))
    #     normal = normal / np.linalg.norm(normal)
        
    #     # Create a rotation matrix that aligns z-axis with normal
    #     # We also need to ensure the gripper orientation makes sense
    #     # First, define desired z-axis (towards object)
    #     z_axis = normal
        
    #     # Choose y-axis directed toward one of the grasp points
    #     # This is arbitrary, but helps with consistent orientation
    #     y_temp = grasp_points[0] - circle_center
    #     y_temp = np.append(y_temp, 0)
    #     y_temp = y_temp / np.linalg.norm(y_temp)
        
    #     # x-axis is perpendicular to both
    #     x_axis = np.cross(y_temp, z_axis)
    #     x_axis = x_axis / np.linalg.norm(x_axis)
        
    #     # Recalculate y-axis to be orthogonal
    #     y_axis = np.cross(z_axis, x_axis)
        
    #     # Create rotation matrix
    #     rot_matrix = np.eye(4)
    #     rot_matrix[:3, 0] = x_axis[:3]
    #     rot_matrix[:3, 1] = y_axis[:3]
    #     rot_matrix[:3, 2] = z_axis[:3]
        
    #     # Convert to quaternion
    #     q = tf_trans.quaternion_from_matrix(rot_matrix)
        
    #     pose_msg.pose.orientation.x = q[0]
    #     pose_msg.pose.orientation.y = q[1]
    #     pose_msg.pose.orientation.z = q[2]
    #     pose_msg.pose.orientation.w = q[3]

    #     return pose_msg

    def publish_grasp_vertices(self, grasp_points, depth_image):
        """Publish grasp vertices to the ROS topic"""
        if grasp_points is None:
            rospy.logwarn("Cannot publish grasp vertices: grasp_points is None")
            return
            
        # Create a Float32MultiArray message
        vertices_msg = Float32MultiArray()
        
        # Setup dimension information (optional but good practice)
        dim1 = MultiArrayDimension()
        dim1.label = "vertices"
        dim1.size = len(grasp_points)
        dim1.stride = len(grasp_points) * 3
        
        dim2 = MultiArrayDimension()
        dim2.label = "coordinates"
        dim2.size = 3
        dim2.stride = 3
        
        vertices_msg.layout.dim = [dim1, dim2]
        vertices_msg.layout.data_offset = 0
        
        # Flatten the grasp_points array [x1,y1,z1,x2,y2,z2,x3,y3,z3]
        flat_vertices = []
        for point in grasp_points:
            # Get x,y coordinates from the point
            x, y = float(point[0]), float(point[1])
            
            # Get z (depth) from the depth image
            z = 0.0
            if 0 <= int(y) < depth_image.shape[0] and 0 <= int(x) < depth_image.shape[1]:
                depth_val = depth_image[int(y), int(x)]
                if depth_val > 0:
                    z = float(depth_val) * self.depth_scale
            
            # Add the x,y,z coordinates to the flattened array
            flat_vertices.extend([x, y, z])
        
        # Set the message data
        vertices_msg.data = flat_vertices
        
        # Publish the message
        self.vertices_pub.publish(vertices_msg)
        rospy.loginfo(f"Published grasp vertices: {flat_vertices}")
        
        return vertices_msg

    def detect_circles(self, color_image):
        """Detect circles in the color image"""
        # Convert to grayscale with enhanced contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        gray = clahe.apply(gray)
        
        # Stronger blur to reduce noise but keep edges
        blur = cv2.GaussianBlur(gray, (5, 5), 2)

        # Crop to ROI
        roi_blur = blur[self.roi_y_min:self.roi_y_max, self.roi_x_min:self.roi_x_max]    

        # Canny edge map (optional but can help filter spurious Hough votes)
        edges = cv2.Canny(roi_blur, 100, 150)
        
        # Circle detection parameters
        dp = 1.0         # Perfect precision for small objects
        minDist = 1      # Minimum distance between detected circles in pixels
        param1 = 40      # Lower Canny edge detector threshold for better sensitivity
        param2 = 25      # Lower accumulator threshold to detect more subtle circles
        minR = 3         # Minimum radius in pixels
        maxR = 15        # Maximum radius in pixels

        circles = cv2.HoughCircles(
            roi_blur,                       
            cv2.HOUGH_GRADIENT,
            dp=dp,
            minDist=minDist,
            param1=param1,
            param2=param2,
            minRadius=minR,
            maxRadius=maxR
        )
        
        # Debug print for circle detection
        if circles is not None:
            rospy.loginfo(f"Found {len(circles[0])} circles")
            # Adjust circle coordinates back to full image space
            circles[0, :, 0] += self.roi_x_min  # Add x offset
            circles[0, :, 1] += self.roi_y_min  # Add y offset
        else:
            rospy.loginfo("No circles detected")
        
        return circles

    def get_valid_circles(self, circles, depth_image):
        """Get valid circles with depth information"""
        valid_circles = []
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            
            for circle in circles[0, :]:
                center_x, center_y, radius = circle
                depth_value = self.get_valid_depth(depth_image, center_x, center_y, radius)
                if depth_value > 0:
                    valid_circles.append((circle, depth_value))
        
        return valid_circles

    def select_best_circle(self, valid_circles, depth_image, color_image=None):
        """Select the best circle for grasping"""
        if not valid_circles:
            return None, None, -1
            
        best_circle_idx = 0
        best_overall_score = float('inf')

        # Debug print
        rospy.loginfo(f"Processing {len(valid_circles)} valid circles for grasp generation")

        for i, (circle, depth_value) in enumerate(valid_circles):
            center_x, center_y, radius = circle
            rospy.loginfo(f"Circle {i+1}: center=({center_x}, {center_y}), radius={radius}, depth={depth_value:.3f}m")

            # Generate diverse grasp configs for this circle
            grasp_configs = self.find_optimal_grasp_triangles(
                np.array([center_x, center_y]), radius, depth_image, color_image,
                num_poses=3
            )
            
            # Check if grasp configs were generated
            if not grasp_configs:
                rospy.logwarn(f"No grasp configurations found for circle {i+1}")
                continue
            else:
                rospy.loginfo(f"Generated {len(grasp_configs)} grasp configs for circle {i+1}")

            # Find config with best score (and preferably no collisions)
            best_config_score = float('inf')

            for grasp_points, angle, score, quality in grasp_configs:
                # Check collisions with other circles
                collision_found = False

                for k, (other_circle, _) in enumerate(valid_circles):
                    if i != k:  # Don't check against itself
                        ox, oy, or_radius = other_circle

                        # Check grasp points distance to other circles
                        for point in grasp_points:
                            px, py = point
                            dist = np.sqrt((px - ox) ** 2 + (py - oy) ** 2)
                            if dist < or_radius + 10:  # Safety margin
                                collision_found = True
                                break

                        # Check triangle edges
                        if not collision_found:
                            for p1_idx in range(3):
                                p2_idx = (p1_idx + 1) % 3
                                p1 = grasp_points[p1_idx]
                                p2 = grasp_points[p2_idx]

                                if self.line_circle_intersection(p1, p2, np.array([ox, oy]), or_radius):
                                    collision_found = True
                                    break

                        if collision_found:
                            break

                # Prioritize collision-free configurations
                if not collision_found and score < best_config_score:
                    best_config_score = score

            # If this circle has a better overall grasp configuration, select it
            if best_config_score < best_overall_score:
                best_overall_score = best_config_score
                best_circle_idx = i

        best_circle, best_depth = valid_circles[best_circle_idx]
        rospy.loginfo(f"Selected circle {best_circle_idx+1} as best for grasping")
        
        return best_circle, best_depth, best_circle_idx

    def generate_grasp_configurations(self, best_circle, radius, depth_image, other_circles=None):
        """Generate grasp configurations for the selected circle"""
        center_x, center_y, radius = best_circle
        
        # Get distinctly different grasp configurations for the best circle
        # Each will have a different triangle shape
        top_grasp_configs = self.find_optimal_valley_triangle(
            depth_image,
            np.array([center_x, center_y]), 
            radius,
            other_circles=other_circles,
            num_configs=3
        )
        
        # If no grasp configurations were found, force generate some basic ones
        if not top_grasp_configs:
            rospy.logwarn("No grasp configurations found by valley triangle method - using fallback")
            # Fallback: Generate basic equilateral triangle configurations
            top_grasp_configs = []
            for angle_offset in [0, np.pi/6, np.pi/3]:
                points = []
                for i in range(3):
                    theta = angle_offset + i * 2 * np.pi / 3
                    x = int(center_x + radius * np.cos(theta))
                    y = int(center_y + radius * np.sin(theta))
                    points.append(np.array([x, y]))
                
                top_grasp_configs.append((points, angle_offset, 1000.0, 3))
        
        rospy.loginfo(f"Generated {len(top_grasp_configs)} final grasp configurations")
        return top_grasp_configs

    def ensure_grasp_diversity(self, top_grasp_configs, center_x, center_y, radius, depth_image, depth_smooth):
        """Ensure diversity among grasp configurations"""
        # First pass - compute shape differences
        shape_signatures = []
        for curr_grasp_points, angle, _, _ in top_grasp_configs:
            sig = self.calculate_triangle_shape_signature(curr_grasp_points)
            shape_signatures.append(sig)

        # Check if shapes are too similar
        too_similar = False
        for i in range(len(shape_signatures)):
            for j in range(i + 1, len(shape_signatures)):
                if self.calculate_shape_similarity(shape_signatures[i], shape_signatures[j]) > 0.9:
                    too_similar = True
                    break

        # If shapes are too similar, create more diverse configurations
        if too_similar and len(top_grasp_configs) > 1:
            rospy.loginfo("Detected similar grasp configurations - forcing diversity")
            verified_configs = []

            # Keep the best config
            verified_configs.append(top_grasp_configs[0])
            grasp_points = top_grasp_configs[0][0]  # Get grasp points from the first config

            # Add configurations with deliberately different orientations
            base_angle = float(np.arctan2(grasp_points[0][1] - center_y, grasp_points[0][0] - center_x))

            # Add configurations with deliberately different orientations
            for offset_angle in [np.pi / 2, np.pi]:  # 90° and 180° offsets
                new_points = []
                for i in range(3):
                    theta = base_angle + offset_angle + i * 2 * np.pi / 3
                    r = radius * (0.9 + 0.2 * (i % 2))  # Alternate between 0.9r and 1.1r
                    x = int(center_x + r * np.cos(theta))
                    y = int(center_y + r * np.sin(theta))

                    # Ensure coordinates are within image bounds
                    x = max(0, min(x, depth_image.shape[1] - 1))
                    y = max(0, min(y, depth_image.shape[0] - 1))

                    new_points.append(np.array([x, y]))

                # Refine points to nearest local minima
                refined_points = []
                for point in new_points:
                    x, y = int(point[0]), int(point[1])
                    search_radius = 12

                    best_local_point = point
                    best_local_depth = float('inf')

                    for dx in range(-search_radius, search_radius + 1):
                        for dy in range(-search_radius, search_radius + 1):
                            nx, ny = x + dx, y + dy
                            if (0 <= nx < depth_image.shape[1] and
                                    0 <= ny < depth_image.shape[0] and
                                    depth_image[ny, nx] > 0):

                                current_depth = depth_smooth[ny, nx]
                                if current_depth < best_local_depth:
                                    best_local_depth = current_depth
                                    best_local_point = np.array([nx, ny])

                    refined_points.append(best_local_point)

                score = self.evaluate_grasp_points(depth_image, refined_points, depth_smooth)
                verified_configs.append((refined_points, base_angle + offset_angle, score, 0))

                if len(verified_configs) >= 3:
                    break

            return verified_configs
        return top_grasp_configs

    def visualize_gradient_analysis(self, depth_image, circle_center, radius):
        """Create visualizations of gradient analysis for depth-guided grasping"""
        # Get smooth depth image for analysis
        depth_copy = depth_image.copy().astype(float)
        max_depth = np.max(depth_copy[depth_copy > 0])
        depth_copy[depth_copy == 0] = max_depth
        depth_smooth = cv2.GaussianBlur(depth_copy, (15, 15), 0)
        
        # Calculate gradients
        grad_y = cv2.Sobel(depth_smooth, cv2.CV_64F, 0, 1, ksize=5)
        grad_x = cv2.Sobel(depth_smooth, cv2.CV_64F, 1, 0, ksize=5)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Calculate Laplacian
        laplacian = cv2.Laplacian(depth_smooth, cv2.CV_64F)
        
        # Calculate local minima map
        minima_indicator, _, _ = self.compute_local_minima_map(depth_image)
        
        # Create visualization figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot depth map
        depth_plot = axes[0, 0].imshow(depth_smooth, cmap='viridis')
        axes[0, 0].set_title('Smoothed Depth Map')
        fig.colorbar(depth_plot, ax=axes[0, 0])
        
        # Circle around the object for reference
        if circle_center is not None and radius is not None:
            circle = plt.Circle((circle_center[0], circle_center[1]), radius, 
                               fill=False, color='red', linewidth=2)
            axes[0, 0].add_patch(circle)
        
        # Plot gradient magnitude
        grad_plot = axes[0, 1].imshow(gradient_magnitude, cmap='hot')
        axes[0, 1].set_title('Gradient Magnitude')
        fig.colorbar(grad_plot, ax=axes[0, 1])
        
        # Plot Laplacian (valleys and ridges)
        lap_plot = axes[1, 0].imshow(laplacian, cmap='coolwarm')
        axes[1, 0].set_title('Laplacian (Valleys and Ridges)')
        fig.colorbar(lap_plot, ax=axes[1, 0])
        
        # Plot local minima
        min_plot = axes[1, 1].imshow(minima_indicator, cmap='gray')
        axes[1, 1].set_title('Local Depth Minima (Valleys)')
        fig.colorbar(min_plot, ax=axes[1, 1])
        
        # Annotate all plots with the circle
        for ax in axes.flatten():
            if circle_center is not None and radius is not None:
                circle = plt.Circle((circle_center[0], circle_center[1]), radius, 
                                   fill=False, color='red', linewidth=2)
                ax.add_patch(circle)
        
        plt.tight_layout()
        return fig
    
    def visualize_collision_avoidance(self, color_image, depth_image, circles, best_circle_idx, grasp_points):
        """Create a visualization showing collision avoidance between multiple objects"""
        vis_image = color_image.copy()
        
        # Draw all circles
        for i, (circle, _) in enumerate(circles):
            cx, cy, r = circle
            # Draw the best circle in green, others in red
            color = (0, 255, 0) if i == best_circle_idx else (0, 0, 255)
            thickness = 3 if i == best_circle_idx else 2
            cv2.circle(vis_image, (cx, cy), r, color, thickness)
            cv2.putText(vis_image, f"Obj {i+1}", (cx-10, cy-r-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Calculate and draw potential collision areas
        if len(circles) > 1:
            for i, (circle, _) in enumerate(circles):
                if i == best_circle_idx:
                    continue
                    
                cx, cy, r = circle
                best_cx, best_cy, best_r = circles[best_circle_idx][0]
                
                # Calculate danger zone between circles
                mid_x = (cx + best_cx) // 2
                mid_y = (cy + best_cy) // 2
                
                # Draw a danger zone with low opacity
                overlay = vis_image.copy()
                cv2.line(overlay, (cx, cy), (best_cx, best_cy), (0, 0, 255), 2)
                cv2.circle(overlay, (mid_x, mid_y), 30, (0, 0, 255), -1)
                alpha = 0.3 # Transparency factor
                cv2.addWeighted(overlay, alpha, vis_image, 1 - alpha, 0, vis_image)
                
                cv2.putText(vis_image, "Collision Zone", (mid_x-50, mid_y+5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Draw the grasp triangle
        if grasp_points is not None:
            points = [p.astype(int) for p in grasp_points]
            # Draw filled triangle with transparency
            overlay = vis_image.copy()
            triangle_pts = np.array([points[0], points[1], points[2]])
            cv2.fillPoly(overlay, [triangle_pts], (0, 255, 0))
            alpha = 0.4
            cv2.addWeighted(overlay, alpha, vis_image, 1 - alpha, 0, vis_image)
            
            # Draw triangle edges
            for i in range(3):
                cv2.line(vis_image, tuple(points[i]), tuple(points[(i + 1) % 3]), (0, 255, 0), 2)
            
            # Mark grasp points
            for i, point in enumerate(points):
                cv2.circle(vis_image, tuple(point), 5, (0, 255, 0), -1)
                cv2.putText(vis_image, f"Grasp {i+1}", (point[0]+10, point[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add title and legend
        cv2.putText(vis_image, "Collision-Free Grasp Planning", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # Add descriptive text
        description = [
            "Green Circle: Target Object",
            "Red Circles: Obstacles",
            "Green Triangle: Collision-Free Grasp",
            "Red Zones: Potential Collision Areas"
        ]
        
        for i, text in enumerate(description):
            cv2.putText(vis_image, text, (20, 70 + 30*i), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return vis_image
    
    def create_coordinate_transformation_diagram(self, color_image, circle_center, grasp_points, depth_value):
        """Create a diagram showing coordinate transformation from image to robot coordinates"""
        # Create a figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))
        
        # First, show the image coordinates
        ax1.imshow(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
        
        # Plot the circle center and grasp points
        ax1.scatter(circle_center[0], circle_center[1], color='blue', s=100, marker='o', label='Object Center')
        
        # Draw grasp triangle
        if grasp_points is not None:
            for i, point in enumerate(grasp_points):
                ax1.scatter(point[0], point[1], color='red', s=100, marker='^', 
                           label=f'Grasp Point {i+1}' if i == 0 else "")
            
            # Connect the points to form the triangle
            points = np.array([grasp_points[0], grasp_points[1], grasp_points[2], grasp_points[0]])
            ax1.plot(points[:, 0], points[:, 1], 'g-', linewidth=2)
        
        # Add image coordinate system arrows
        ax1.arrow(50, 50, 50, 0, head_width=10, head_length=10, fc='r', ec='r')
        ax1.arrow(50, 50, 0, 50, head_width=10, head_length=10, fc='g', ec='g')
        ax1.text(105, 50, 'X (u)', color='r', fontsize=12)
        ax1.text(50, 105, 'Y (v)', color='g', fontsize=12)
        
        ax1.set_title('2D Image Coordinates (pixels)', fontsize=14)
        ax1.legend()
        
        # Now create a 3D plot for robot/camera coordinates
        ax2 = fig.add_subplot(1, 2, 1, projection='3d')
        
        # Transform image coordinates to 3D (we'll use placeholder values)
        # Get camera parameters
        fx, fy = self.color_intrinsics.fx, self.color_intrinsics.fy
        ppx, ppy = self.color_intrinsics.ppx, self.color_intrinsics.ppy
        
        # Transform 2D image coordinates to 3D camera coordinates
        # Z is the depth, X and Y are computed using the camera model
        z = depth_value  # Depth in meters
        x = (circle_center[0] - ppx) * z / fx
        y = (circle_center[1] - ppy) * z / fy
        
        # Plot the object center
        ax2.scatter([x], [y], [z], color='blue', s=100, marker='o', label='Object Center')
        
        # Transform and plot the grasp points
        if grasp_points is not None:
            grasp_3d_points = []
            for i, point in enumerate(grasp_points):
                # Estimate depth at grasp point similar to object center
                point_z = z  # Assuming same depth, in real usage you'd get the actual depth
                point_x = (point[0] - ppx) * point_z / fx
                point_y = (point[1] - ppy) * point_z / fy
                
                grasp_3d_points.append((point_x, point_y, point_z))
                ax2.scatter([point_x], [point_y], [point_z], color='red', s=100, marker='^',
                           label=f'Grasp Point {i+1}' if i == 0 else "")
            
            # Connect the 3D points to form a triangle
            if len(grasp_3d_points) >= 3:
                points = np.array([grasp_3d_points[0], grasp_3d_points[1], grasp_3d_points[2], grasp_3d_points[0]])
                ax2.plot(points[:, 0], points[:, 1], points[:, 2], 'g-', linewidth=2)
        
        # Add coordinate system arrows
        arrow_len = 0.1
        ax2.quiver(0, 0, 0, arrow_len, 0, 0, color='r')
        ax2.quiver(0, 0, 0, 0, arrow_len, 0, color='g')
        ax2.quiver(0, 0, 0, 0, 0, arrow_len, color='b')
        ax2.text(arrow_len, 0, 0, 'X', color='r', fontsize=12)
        ax2.text(0, arrow_len, 0, 'Y', color='g', fontsize=12)
        ax2.text(0, 0, arrow_len, 'Z', color='b', fontsize=12)
        
        # Set axis labels
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_zlabel('Z (m)')
        
        # Set axis limits for better visualization
        limit = 0.2
        ax2.set_xlim([-limit, limit])
        ax2.set_ylim([-limit, limit])
        ax2.set_zlim([0, 2*limit])
        
        ax2.set_title('3D Camera/Robot Coordinates (meters)', fontsize=14)
        ax2.legend()
        
        # Add text describing the transformation
        transform_text = (
            "2D to 3D Transformation:\n"
            f"X = (u - ppx) * Z / fx\n"
            f"Y = (v - ppy) * Z / fy\n"
            f"Z = depth\n\n"
            f"Camera Parameters:\n"
            f"fx = {fx:.1f}, fy = {fy:.1f}\n"
            f"ppx = {ppx:.1f}, ppy = {ppy:.1f}"
        )
        
        fig.text(0.5, 0.01, transform_text, ha='center', fontsize=12, 
                bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def create_system_architecture_diagram(self):
        """Create a system architecture diagram for depth-guided grasp planning"""
        # Create a directed graph using NetworkX
        G = nx.DiGraph()
        
        # Add nodes (components of the system)
        components = [
            "RealSense Camera", 
            "RGB Image", 
            "Depth Image",
            "Circle Detection",
            "Depth Analysis",
            "Grasp Point Selection",
            "Triangle Generation",
            "Collision Detection",
            "Grasp Pose Optimization",
            "Robot Control"
        ]
        
        # Define node positions for clear layout
        pos = {
            "RealSense Camera": (0, 0),
            "RGB Image": (-1, -1),
            "Depth Image": (1, -1),
            "Circle Detection": (-1, -2),
            "Depth Analysis": (1, -2),
            "Grasp Point Selection": (0, -3),
            "Triangle Generation": (0, -4),
            "Collision Detection": (-1, -5),
            "Grasp Pose Optimization": (1, -5),
            "Robot Control": (0, -6)
        }
        
        # Add nodes with positions
        for component in components:
            G.add_node(component, pos=pos[component])
        
        # Add edges (connections between components)
        edges = [
            ("RealSense Camera", "RGB Image"),
            ("RealSense Camera", "Depth Image"),
            ("RGB Image", "Circle Detection"),
            ("Depth Image", "Depth Analysis"),
            ("Circle Detection", "Grasp Point Selection"),
            ("Depth Analysis", "Grasp Point Selection"),
            ("Grasp Point Selection", "Triangle Generation"),
            ("Triangle Generation", "Collision Detection"),
            ("Triangle Generation", "Grasp Pose Optimization"),
            ("Collision Detection", "Grasp Pose Optimization"),
            ("Grasp Pose Optimization", "Robot Control")
        ]
        
        G.add_edges_from(edges)
        
        # Create a figure
        plt.figure(figsize=(12, 10))
        
        # Draw the graph
        node_pos = nx.get_node_attributes(G, 'pos')
        
        # Draw nodes with custom colors and sizes
        node_colors = {
            "RealSense Camera": "lightblue",
            "RGB Image": "lightgreen",
            "Depth Image": "lightgreen",
            "Circle Detection": "lightsalmon",
            "Depth Analysis": "lightsalmon",
            "Grasp Point Selection": "gold",
            "Triangle Generation": "gold",
            "Collision Detection": "pink",
            "Grasp Pose Optimization": "pink",
            "Robot Control": "lightgrey"
        }
        
        # Draw nodes
        for node, position in node_pos.items():
            nx.draw_networkx_nodes(G, {node: position}, nodelist=[node], 
                                  node_color=node_colors[node], node_size=3000, alpha=0.8)
        
        # Draw edges
        nx.draw_networkx_edges(G, node_pos, width=2, arrowsize=20)
        
        # Add labels
        nx.draw_networkx_labels(G, node_pos, font_size=10, font_weight="bold")
        
        # Add informative labels to edges
        edge_labels = {
            ("RealSense Camera", "RGB Image"): "Color Stream",
            ("RealSense Camera", "Depth Image"): "Depth Stream",
            ("RGB Image", "Circle Detection"): "Hough Transforms",
            ("Depth Image", "Depth Analysis"): "Valley Detection",
            ("Circle Detection", "Grasp Point Selection"): "Object Detection",
            ("Depth Analysis", "Grasp Point Selection"): "Valley Mapping",
            ("Grasp Point Selection", "Triangle Generation"): "Optimal Points",
            ("Triangle Generation", "Collision Detection"): "Grasp Candidates",
            ("Triangle Generation", "Grasp Pose Optimization"): "Coordinate Transform",
            ("Collision Detection", "Grasp Pose Optimization"): "Validation",
            ("Grasp Pose Optimization", "Robot Control"): "Pose Commands"
        }
        
        nx.draw_networkx_edge_labels(G, node_pos, edge_labels=edge_labels, font_size=8)
        
        # Add a title
        plt.title("DEPTH-GUIDED GRASP: System Architecture", fontsize=16)
        
        # Remove the axes
        plt.axis('off')
        
        # Add legend for node colors
        legend_patches = [plt.Rectangle((0,0),1,1, color=color, alpha=0.8) for color in 
                         ["lightblue", "lightgreen", "lightsalmon", "gold", "pink", "lightgrey"]]
        legend_labels = ["Sensor", "Data", "Processing", "Planning", "Validation", "Execution"]
        plt.legend(legend_patches, legend_labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), 
                  ncol=3, frameon=True)
        
        plt.tight_layout()
        return plt.gcf()
    
    def create_performance_comparison_chart(self):
        """Create a performance comparison chart showing grasp success rates"""
        # Create a figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Prepare data for method comparison (from simulated performance)
        methods = list(self.comparison_methods.keys())
        success_rates = [100.0 * self.comparison_methods[m]['success'] / 
                         max(1, self.comparison_methods[m]['attempts']) for m in methods]
        
        # Create bar chart for method comparison
        bars = ax1.bar(methods, success_rates, color=['#2C7BB6', '#D7191C', '#FDAE61', '#ABD9E9'])
        
        # Add values on top of bars
        for bar, value in zip(bars, success_rates):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f"{value:.1f}%", ha='center', va='bottom', fontsize=12)
        
        ax1.set_ylim(0, 105)  # Make room for percentage labels
        ax1.set_ylabel('Success Rate (%)', fontsize=12)
        ax1.set_title('Grasp Success Rate by Method', fontsize=14)
        ax1.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Add horizontal line for baseline
        ax1.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='Baseline (50%)')
        ax1.legend()
        
        # Prepare data for condition comparison (from simulated performance)
        conditions = list(self.condition_performance.keys())
        condition_rates = [100.0 * self.condition_performance[c]['success'] / 
                          max(1, self.condition_performance[c]['attempts']) for c in conditions]
        
        # Create bar chart for condition comparison
        condition_bars = ax2.bar(conditions, condition_rates, color='#2C7BB6')
        
        # Add values on top of bars
        for bar, value in zip(condition_bars, condition_rates):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f"{value:.1f}%", ha='center', va='bottom', fontsize=12)
        
        ax2.set_ylim(0, 105)
        ax2.set_ylabel('Success Rate (%)', fontsize=12)
        ax2.set_title('Performance Under Different Conditions', fontsize=14)
        ax2.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Label rotation for better readability
        plt.setp(ax2.get_xticklabels(), rotation=30, ha='right')
        
        plt.tight_layout()
        return fig
    
    def visualize_diverse_grasps(self, color_image, circle_center, radius, grasp_configs):
        """Visualize multiple diverse grasp configurations on the same object"""
        if not grasp_configs:
            return color_image.copy()
            
        # Create a visualization image
        vis_image = color_image.copy()
        
        # Draw the object circle
        cv2.circle(vis_image, (int(circle_center[0]), int(circle_center[1])), 
                  int(radius), (255, 0, 0), 2)
        
        # Colors for different grasp configurations
        colors = [(0, 255, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]
        
        # Draw each grasp configuration with a different color
        for i, (grasp_points, angle, score, quality) in enumerate(grasp_configs):
            if i >= len(colors):
                break
                
            color = colors[i]
            
            # Draw triangle vertices (grasp points)
            for j, point in enumerate(grasp_points):
                point = point.astype(int)
                cv2.circle(vis_image, (point[0], point[1]), 5, color, -1)
                cv2.putText(vis_image, f"{j+1}_{i+1}", (point[0] + 5, point[1] + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw triangle edges
            points = [p.astype(int) for p in grasp_points]
            for j in range(3):
                cv2.line(vis_image, (points[j][0], points[j][1]), 
                        (points[(j+1)%3][0], points[(j+1)%3][1]), color, 2)
            
            # Add angle and score information
            angle_deg = angle * 180 / np.pi
            cv2.putText(vis_image, f"Config {i+1}: {angle_deg:.1f}°, Score: {score:.0f}",
                       (20, 30 + 30*i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Add a title
        cv2.putText(vis_image, "Diverse Grasp Configurations", 
                   (vis_image.shape[1]//2 - 150, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        return vis_image
    
    def annotate_rgb_view(self, color_image, circles, best_circle_idx, best_grasp_points):
        """Create an annotated RGB view of the scene with detection and grasp results"""
        # Create a copy of the input image
        annotated_image = color_image.copy()
        
        # Draw ROI
        cv2.rectangle(annotated_image, (self.roi_x_min, self.roi_y_min), 
                     (self.roi_x_max, self.roi_y_max), (0, 255, 255), 2)
        cv2.putText(annotated_image, "ROI", (self.roi_x_min + 5, self.roi_y_min - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Draw all detected circles
        if circles:
            for i, (circle, depth) in enumerate(circles):
                cx, cy, r = circle
                # Use different colors: green for the selected circle, red for others
                color = (0, 255, 0) if i == best_circle_idx else (0, 0, 255)
                thickness = 3 if i == best_circle_idx else 2
                
                # Draw the circle
                cv2.circle(annotated_image, (cx, cy), r, color, thickness)
                
                # Add depth information
                cv2.putText(annotated_image, f"d={depth:.3f}m", (cx - 30, cy + r + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Add object ID
                cv2.putText(annotated_image, f"Obj {i+1}", (cx - 15, cy - r - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Draw the grasp triangle if available
        if best_grasp_points is not None:
            # Draw the triangle vertices
            for i, point in enumerate(best_grasp_points):
                point = point.astype(int)
                cv2.circle(annotated_image, (point[0], point[1]), 5, (255, 0, 255), -1)
                cv2.putText(annotated_image, f"F{i+1}", (point[0] + 10, point[1]),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            
            # Draw the triangle
            points = [p.astype(int) for p in best_grasp_points]
            for i in range(3):
                cv2.line(annotated_image, (points[i][0], points[i][1]),
                        (points[(i+1)%3][0], points[(i+1)%3][1]), (255, 0, 255), 2)
        
        # Add title and description
        cv2.putText(annotated_image, "DEPTH-GUIDED GRASP: Detection Results", 
                   (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        # Add legend
        legends = [
            "Yellow: Region of Interest",
            "Green: Selected Target Object",
            "Red: Other Detected Objects",
            "Purple: Optimal Grasp Configuration"
        ]
        
        for i, text in enumerate(legends):
            cv2.putText(annotated_image, text, (20, annotated_image.shape[0] - 20 - 30*i),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return annotated_image
    
    def process_frames(self, color_image, depth_image):
        """Process camera frames to detect circles and plan grasps"""
        try:
            if color_image is None or depth_image is None:
                rospy.logerr("Received empty frames")
                return False

            rospy.loginfo(f"Processing frames - Color: {color_image.shape}, Depth: {depth_image.shape}")
            
            # Save raw data
            cv2.imwrite(f"{self.viz_dirs['raw_data']}/color_{self.timestamp}.png", color_image)
            cv2.imwrite(f"{self.viz_dirs['raw_data']}/depth_{self.timestamp}.png", depth_image)
            
            # Save the ROI visualization
            roi_vis = color_image.copy()
            cv2.rectangle(roi_vis, (self.roi_x_min, self.roi_y_min), (self.roi_x_max, self.roi_y_max), (0, 0, 255), 2)  # Red rectangle
            cv2.putText(roi_vis, "ROI", (self.roi_x_min + 10, self.roi_y_min + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imwrite(f"{self.viz_dirs['raw_data']}/roi_{self.timestamp}.png", roi_vis)

            # Step 1: Detect circles in the color image
            circles = self.detect_circles(color_image)
            
            # Step 2: Get valid circles with depth information
            valid_circles = self.get_valid_circles(circles, depth_image)
            
            # Initialize variables that might be used later
            depth_smooth = None
            best_circle = None
            best_depth = 0.0
            best_circle_idx = -1
            grasp_idx = -1  # Default value if no grasp is found
            
            # Get depth image smoothed copy (for various processing steps)
            depth_copy = depth_image.copy().astype(float)
            max_depth = np.max(depth_copy[depth_copy > 0])
            depth_copy[depth_copy == 0] = max_depth
            depth_smooth = cv2.GaussianBlur(depth_copy, (15, 15), 0)
            
            if valid_circles:
                # Step 3: Select the best circle for grasping
                best_circle, best_depth, best_circle_idx = self.select_best_circle(
                    valid_circles, depth_image, color_image
                )
                
                if best_circle is not None:
                    center_x, center_y, radius = best_circle
                    
                    # Step 4: Get other circles (for collision avoidance)
                    other_circles = [c for i, c in enumerate(valid_circles) if i != best_circle_idx]

                    # Step 5: Generate grasp configurations for the selected circle
                    top_grasp_configs = self.generate_grasp_configurations(
                        best_circle, radius, depth_image, other_circles
                    )
                    
                    # Step 6: Ensure grasp diversity
                    top_grasp_configs = self.ensure_grasp_diversity(
                        top_grasp_configs, center_x, center_y, radius, depth_image, depth_smooth
                    )

                    # Step 7: Visualize and process all grasp configurations
                    # Colors for different grasp configurations
                    grasp_colors = [(0, 255, 0), (0, 255, 255), (255, 0, 255)]
                    
                    # Create combined visualization with all circles
                    result_image = color_image.copy()
                    
                    # Draw all detected circles
                    for i, (circle, _) in enumerate(valid_circles):
                        cx, cy, r = circle
                        if i == best_circle_idx:
                            cv2.circle(result_image, (cx, cy), r, (0, 255, 0), 3)
                        else:
                            cv2.circle(result_image, (cx, cy), r, (255, 0, 0), 2)
                    
                    # Process each grasp configuration
                    for grasp_idx, (grasp_points, angle, score, _) in enumerate(top_grasp_configs):
                        if grasp_points is None:
                            continue

                        # Use different colors for each grasp configuration
                        color = grasp_colors[grasp_idx % len(grasp_colors)]

                        # Add debug info to visualization
                        angle_deg = angle * 180 / np.pi

                        # Plot 3D visualization for each configuration
                        fig = self.plot_gradient_descent_3d(
                            depth_image, 
                            np.array([center_x, center_y]), 
                            grasp_points, 
                            radius,
                            elev=30,    # Adjust elevation angle
                            azim=45,    # Adjust azimuth angle
                            reverse_view=True  # Set to False to use the original view
                        )
                        fig.savefig(
                            f"{self.viz_dirs['grasp']}/grasp_3d_{self.timestamp}_option{grasp_idx + 1}.png", 
                            dpi=100
                        )
                        plt.close(fig)

                        # Create individual grasp visualization
                        grasp_vis = self.visualize_grasp(
                            color_image.copy(),
                            np.array([center_x, center_y]),
                            radius,
                            grasp_points,
                            best_depth,
                            label_suffix=f"_Option{grasp_idx + 1}",
                            color=color
                        )

                        # Add angle to visualization
                        cv2.putText(
                            grasp_vis,
                            f"Angle: {angle_deg:.1f}°",
                            (center_x - 60, center_y + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
                        )

                        # Save grasp visualization
                        cv2.imwrite(
                            f"{self.viz_dirs['grasp']}/grasp_vis_{self.timestamp}_option{grasp_idx + 1}.png",
                            grasp_vis
                        )

                        # Add this grasp to the result image
                        for i, point in enumerate(grasp_points):
                            point = point.astype(int)
                            cv2.circle(result_image, tuple(point), 5, color, -1)
                            cv2.putText(
                                result_image, 
                                f"F{i + 1}_{grasp_idx + 1}", 
                                (point[0] + 10, point[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
                            )

                        points = [p.astype(int) for p in grasp_points]
                        for i in range(3):
                            cv2.line(
                                result_image, 
                                tuple(points[i]), 
                                tuple(points[(i + 1) % 3]), 
                                color, 
                                2
                            )

                        # Add angle info to result image
                        cv2.putText(
                            result_image,
                            f"A{grasp_idx + 1}: {angle_deg:.1f}°",
                            (center_x - 30, center_y + 20 + grasp_idx * 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                        )

                        # Publish ROS grasp pose for the first (best) configuration
                        if grasp_idx == 0:
                            # Calculate and publish ROS grasp pose
                            # ros_grasp_pose = self.calculate_ros_grasp_pose(
                            #     np.array([center_x, center_y]),
                            #     best_depth,
                            #     grasp_points
                            # )
                            # self.grasp_pose_pub.publish(ros_grasp_pose)
                            
                            # Publish grasp vertices
                            self.publish_grasp_vertices(grasp_points, depth_image)
                            
                            # Generate and save additional visualizations for the best grasp
                            
                            # 1. Gradient analysis visualization
                            gradient_fig = self.visualize_gradient_analysis(
                                depth_image, np.array([center_x, center_y]), radius
                            )
                            gradient_fig.savefig(
                                f"{self.viz_dirs['gradient']}/gradient_analysis_{self.timestamp}.png",
                                dpi=100
                            )
                            plt.close(gradient_fig)
                            
                            # 2. Collision avoidance visualization (if multiple objects)
                            if len(valid_circles) > 1:
                                collision_vis = self.visualize_collision_avoidance(
                                    color_image, depth_image, valid_circles, best_circle_idx, grasp_points
                                )
                                cv2.imwrite(
                                    f"{self.viz_dirs['detection']}/collision_avoidance_{self.timestamp}.png",
                                    collision_vis
                                )
                            
                            # 3. Coordinate transformation diagram
                            coord_fig = self.create_coordinate_transformation_diagram(
                                color_image, np.array([center_x, center_y]), grasp_points, best_depth
                            )
                            coord_fig.savefig(
                                f"{self.viz_dirs['grasp']}/coordinate_transform_{self.timestamp}.png",
                                dpi=100
                            )
                            plt.close(coord_fig)

                    # Create and save diverse grasp visualization
                    diverse_grasps = self.visualize_diverse_grasps(
                        color_image, np.array([center_x, center_y]), radius, top_grasp_configs
                    )
                    cv2.imwrite(
                        f"{self.viz_dirs['grasp']}/diverse_grasps_{self.timestamp}.png",
                        diverse_grasps
                    )
                    
                    # Create annotated RGB view
                    annotated_rgb = self.annotate_rgb_view(
                        color_image, valid_circles, best_circle_idx, 
                        top_grasp_configs[0][0] if top_grasp_configs else None
                    )
                    cv2.imwrite(
                        f"{self.viz_dirs['detection']}/annotated_rgb_{self.timestamp}.png",
                        annotated_rgb
                    )
                    
                    # Save final result image with all grasps
                    cv2.imwrite(f"{self.viz_dirs['grasp']}/all_grasps_{self.timestamp}.png", result_image)
                
                # Create detection results visualization (side-by-side RGB and depth)
                if best_circle is not None:
                    # Create side-by-side visualization
                    detection_vis = np.zeros((color_image.shape[0], color_image.shape[1]*2, 3), dtype=np.uint8)
                    
                    # Convert depth image to color visualization
                    depth_colormap = cv2.applyColorMap(
                        cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
                    )
                    
                    # Mark circles on both images
                    color_with_circles = color_image.copy()
                    depth_with_circles = depth_colormap.copy()
                    
                    for i, (circle, depth_val) in enumerate(valid_circles):
                        cx, cy, r = circle
                        color = (0, 255, 0) if i == best_circle_idx else (0, 0, 255)
                        thickness = 3 if i == best_circle_idx else 2
                        
                        # Draw on RGB image
                        cv2.circle(color_with_circles, (cx, cy), r, color, thickness)
                        cv2.putText(color_with_circles, f"{i+1}", (cx-5, cy+5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                        # Draw on depth image
                        cv2.circle(depth_with_circles, (cx, cy), r, color, thickness)
                        cv2.putText(depth_with_circles, f"{i+1}", (cx-5, cy+5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Combine images side by side
                    detection_vis[:, :color_image.shape[1]] = color_with_circles
                    detection_vis[:, color_image.shape[1]:] = depth_with_circles
                    
                    # Add titles
                    cv2.putText(detection_vis, "RGB Image", (50, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                    cv2.putText(detection_vis, "Depth Image", (color_image.shape[1] + 50, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                    
                    # Save the detection visualization
                    cv2.imwrite(f"{self.viz_dirs['detection']}/circle_detection_{self.timestamp}.png", detection_vis)
                
                # Publish processed images to ROS topics
                color_message = self.numpy_to_imgmsg(result_image if 'result_image' in locals() else color_image, "bgr8")
                depth_message = self.numpy_to_imgmsg(depth_image, "16UC1")
                
                self.color_pub.publish(color_message)
                self.depth_pub.publish(depth_message)
            
            # Create and save system architecture diagram
            system_fig = self.create_system_architecture_diagram()
            system_fig.savefig(f"{self.results_dir}/system_architecture.png", dpi=100)
            plt.close(system_fig)
            
            # Create and save performance comparison charts
            perf_fig = self.create_performance_comparison_chart()
            perf_fig.savefig(f"{self.viz_dirs['comparison']}/performance_comparison.png", dpi=100)
            plt.close(perf_fig)

            # Handle keyboard events
            key = cv2.waitKey(1)
            if key == ord('s'):
                # If 's' is pressed, save the current frame
                self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"{self.viz_dirs['raw_data']}/color_{self.timestamp}.png", color_image)
                cv2.imwrite(f"{self.viz_dirs['raw_data']}/depth_{self.timestamp}.png", depth_image)
                rospy.loginfo(f"Saved frames with timestamp {self.timestamp}")
            elif key == ord('p'):
                # If 'p' is pressed, toggle processing pause
                self.process_current_frame = not self.process_current_frame
                rospy.loginfo(f"Processing {'paused' if not self.process_current_frame else 'resumed'}")
            elif key == 27 or key == ord('q'):  # ESC or 'q'
                return False  # Return False to indicate we should exit

            return True

        except Exception as e:
            rospy.logerr(f"Error in processing frames: {e}")
            import traceback
            rospy.logerr(f"Traceback: {traceback.format_exc()}")
            return False

    def filter_depth_by_range(self, depth_image, min_depth_mm=486, max_depth_mm=1500):
        """Filter depth image to keep only values within a specific range"""
        # Create a copy of the depth image
        filtered_depth = depth_image.copy()
        
        # Convert from millimeters to raw depth units
        min_depth = int(min_depth_mm / self.depth_scale / 1000)
        max_depth = int(max_depth_mm / self.depth_scale / 1000)
        
        # Filter out values outside the range
        filtered_depth[filtered_depth < min_depth] = 0
        filtered_depth[filtered_depth > max_depth] = 0
        
        return filtered_depth

    def run(self):
        """Main processing loop"""
        try:
            # MODIFIED: Take a single snapshot at beginning
            self.process_current_frame = True  # Always process
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Warmup - wait for stable frames
            rospy.loginfo("Warming up camera...")
            for _ in range(30):  # Wait for 30 frames
                self.pipeline.wait_for_frames()
                
            # Take a single snapshot after warmup
            frames = self.pipeline.wait_for_frames()
            
            # Align depth frame to color frame
            aligned_frames = self.align_to_color.process(frames)
            
            # Get aligned frames
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if depth_frame and color_frame:
                # Convert images to numpy arrays
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
                
                # Filter depth image by range (as per requirements)
                depth_image = self.filter_depth_by_range(depth_image, 486, 1500)
                
                # Save the snapshot
                self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"{self.viz_dirs['raw_data']}/color_{self.timestamp}.png", color_image)
                cv2.imwrite(f"{self.viz_dirs['raw_data']}/depth_{self.timestamp}.png", depth_image)
                rospy.loginfo(f"Saved initial snapshot with timestamp {self.timestamp}")
                
                # Print extra debug info
                rospy.loginfo("Beginning processing of snapshot...")
                rospy.loginfo(f"Image dimensions: {color_image.shape}")
                rospy.loginfo(f"Depth range: min={np.min(depth_image[depth_image > 0])}, " +
                     f"max={np.max(depth_image)}, " +
                     f"mean={np.mean(depth_image[depth_image > 0])}")
                
                # Process the captured snapshot
                success = self.process_frames(color_image.copy(), depth_image.copy())
                if not success:
                    rospy.logwarn("Processing failed - will retry with debug mode")
                    # Force grasp generation with debug mode
                    self.debug_mode = True
                    success = self.process_frames(color_image.copy(), depth_image.copy())
                
                # Wait for key press to exit
                rospy.loginfo("Processing complete. Press 'q' or ESC to exit.")
                while not rospy.is_shutdown():
                    key = cv2.waitKey(100)
                    if key == 27 or key == ord('q'):  # ESC or 'q'
                        break
            
        except Exception as e:
            rospy.logerr(f"Error in RealSense processing: {e}")
            import traceback
            rospy.logerr(f"Traceback: {traceback.format_exc()}")
        finally:
            # Stop streaming
            self.pipeline.stop()
            cv2.destroyAllWindows()
            rospy.loginfo("RealSense pipeline stopped")


def main():
    """Main entry point for the application"""
    try:
        print("Starting DEPTH-GUIDED GRASP detection node...")
        node = ObjectDetectionNode()
        node.run()
    except rospy.ROSInterruptException:
        print("Program interrupted")
        pass
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        print(traceback.format_exc())
    finally:
        print("Shutting down")
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()