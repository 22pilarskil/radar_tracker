import cv2
import numpy as np
import math
import time
import os

from scipy.optimize import linear_sum_assignment
from scipy.interpolate import splprep, splev
from scipy.spatial.distance import cdist
from skimage.color import rgb2lab, deltaE_ciede94
from sklearn.cluster import KMeans



# Tracker values
THRESHOLD1_CANNY = 50
THRESHOLD2_CANNY = 100
RHO = 8
THETA = 2 * np.pi/180
THRESHOLD_HOUGH = 200
N_CLUSTERS = 4


orb = cv2.ORB_create()


def draw_lines(img, lines):
    """
    Overlay lines onto provided image
    :param img: image to overlay lines on
    :param lines: array of lines to overlay onto image
    :return:
        img: image with overlaid lines
    """
    img = img.copy()
    if lines is not None:
        for rho_, theta_ in lines[:, 0]:

            a = np.cos(theta_)
            b = np.sin(theta_)
            x0 = a * rho_
            y0 = b * rho_
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    return img

def normalize_points(points):
    """
    Normalize provided set of points around the origin
    :param points: set of points to normalize
    :return:
        scaled_points: normalized set of points
    """
    centroid = np.mean(points, axis=0)
    normalized_points = points - centroid
    max_distance = np.max(np.linalg.norm(normalized_points, axis=1))
    scaled_points = normalized_points / max_distance
    return scaled_points


def calculate_intersection_points(lines):
    """
    Calculate all intersection points between passed in set of lines
    :param lines: array of lines to calculate intersection points of
    :return:
        intersection_points: set of intersection points for given set of lines
    """
    intersection_points = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            rho1, theta1 = lines[i][0]
            rho2, theta2 = lines[j][0]

            A = np.array([[np.cos(theta1), np.sin(theta1)],
                          [np.cos(theta2), np.sin(theta2)]])
            b = np.array([rho1, rho2])
            intersection = np.linalg.solve(A, b)
            intersection_points.append(intersection)

    return np.array(intersection_points)


def reduce_lines(lines, img, n_clusters=N_CLUSTERS):
    """
    Generate a specified number of lines that best represents a given set of lines
    :param lines: array of cv2.HoughLines generated lines to reduce
    :param img: image to overlay reduced lines upon (visualization purposes)
    :param n_clusters: number of lines to reduce to
    :return:
        img: image overlaid with lines from reduced_lines array
        points: normalized intersection points of lines from reduced_lines array
    """

    # Transform rho, theta to Cartesian coordinates
    rho = lines[:, 0]
    theta = lines[:, 1]
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)

    # Reshape to 2D array for KMeans
    points = np.column_stack((x, y))

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=N_CLUSTERS, n_init=1)
    kmeans.fit(points)

    # Get cluster centers and convert back to polar coordinates
    x_centers, y_centers = kmeans.cluster_centers_.T
    rho_centers = np.hypot(x_centers, y_centers)
    theta_centers = np.arctan2(y_centers, x_centers)

    reduced_lines = np.column_stack((rho_centers, theta_centers)).reshape(-1, 1, 2)
    img = draw_lines(img, reduced_lines)
    points = normalize_points(calculate_intersection_points(reduced_lines))

    return img, points


def rotate_points(points, angle):
    """
    Rotate points through a specified angle
    :param points: array of points to be rotated
    :param angle: angle to rotate by
    :return:
        rotated_points: array of rotated points
    """
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rotated_points = np.dot(points, rotation_matrix)
    return rotated_points


def minimize_point_distance(points1, points2):
    """
    Find ordering of points2 array that most closely matches points1 (smallest elementwise difference)
    :param points1: array to match
    :param points2: array to reorder
    :return:
        matched_points2: reordered points2 array
    """
    distances = np.linalg.norm(points1[:, None] - points2, axis=-1)
    row_ind, col_ind = linear_sum_assignment(distances)
    matched_points2 = points2[col_ind]
    return matched_points2


def find_closest_match(points_tracker, points_canny):
    """
    Find which rotation of points_canny most closely matches points_tracker (smallest elementwise difference)
    :param points_tracker: point profile of reference image
    :param points_canny: point profile of canny image
    :return:
        closest_index: angle of closest match (indexes represent angle rotations from 1-360 degrees)
        closest_points: set of rotated points which minimize elementwise difference between tracker and canny points
        closest_distance: elementwise difference between tracker and canny points
    """
    min_distance = np.inf
    closest_points = None
    closest_index = None

    for i in range(360):
        rotated_points = rotate_points(points_canny, (i * np.pi) / 180.)
        minimized_points = minimize_point_distance(points_tracker, rotated_points)
        total_distance = np.sum(np.linalg.norm(minimized_points - points_tracker, axis=1))

        if total_distance < min_distance:
            min_distance = total_distance
            closest_points = rotated_points
            closest_index = i

    return closest_index, closest_points, min_distance


def determine_angle(bbox, points_tracker):
    """
    Determine orientation of tracker within bounding box
    :param bbox: bounding box of a given frame containing a YOLOv5 detected tracker object
    :param points_tracker: point profile of tracker reference image
    :return:
        find_closest_match(...): output from find_closest_match
        img: image overlaid with lines from reduced_lines array
    """
    edges = cv2.Canny(bbox, THRESHOLD1_CANNY, THRESHOLD2_CANNY)
    lines = cv2.HoughLines(edges, RHO, THETA, THRESHOLD_HOUGH)


    if lines is None or len(lines) < N_CLUSTERS:
        edges = cv2.Canny(bbox, THRESHOLD1_CANNY // 2, THRESHOLD2_CANNY // 2)
        lines = cv2.HoughLines(edges, RHO, THETA, THRESHOLD_HOUGH)

    if lines is None or len(lines) < N_CLUSTERS:
        raise ValueError("Not enough HoughLines generated: {}".format(len(lines) if lines is not None else None))

    img, points = reduce_lines(np.squeeze(lines), edges)

    return *find_closest_match(points_tracker, points), img


def equalize_points(set1, set2):
    """
    Compress longer of two sets to length of shorter set
    :param set1:
    :param set2:
    :return:
        set1, set2: Compressed forms of set1 and set2 with same length
    """
    if len(set1) < len(set2):
        kmeans = KMeans(n_clusters=len(set1), n_init=1)
        kmeans.fit(set2)
        set2 = kmeans.cluster_centers_
    elif len(set1) > len(set2):
        kmeans = KMeans(n_clusters=len(set2), n_init=1)
        kmeans.fit(set1)
        set1 = kmeans.cluster_centers_
    return set1, set2


def normalize_keypoints(points, rotation_angle):

    mean = np.mean(points, axis=0)
    points -= mean
    theta = np.radians(rotation_angle)
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    points = np.dot(points, rotation_matrix.T)
    return points


def assign_points(keypoints_1, keypoints_2):
    # Calculate pairwise Euclidean distances between keypoints
    keypoint_cap = int(min(len(keypoints_1), len(keypoints_2)) * 0.8)
    distance_matrix = cdist(keypoints_1, keypoints_2)

    # Find the optimal assignment using the Hungarian algorithm
    row_indices, col_indices = linear_sum_assignment(distance_matrix)

    # Select the keypoint pairs with the lowest distances
    selected_keypoints_1 = keypoints_1[row_indices[:keypoint_cap]]
    selected_keypoints_2 = keypoints_2[col_indices[:keypoint_cap]]

    # Calculate the total cost of the selected pairs
    total_cost = distance_matrix[row_indices[:keypoint_cap], col_indices[:keypoint_cap]].sum()

    return selected_keypoints_1, selected_keypoints_2, total_cost / len(selected_keypoints_2)


def determine_angle_kps(reference_img, kps_tracker):
    kps = generate_keypoint_profile(reference_img)
    distance = np.inf
    angle = 0
    for i in range(360):
        kp1_normalized = normalize_keypoints(kps, 0)
        kp2_normalized = normalize_keypoints(kps_tracker, i)

        # Assume kp1_normalized and kp2_normalized are the normalized keypoints from previous steps
        set1 = np.array(kp1_normalized)
        set2 = np.array(kp2_normalized)

        kp1, kp2, total_cost = assign_points(set1, set2)
        if total_cost < distance:
            distance = total_cost
            angle = i

    return distance, angle


def generate_point_profile(reference_img):
    """
    Generate a point profile of a reference image from intersection points
    :param reference_img: file path of reference image
    :return:
        points: array containing intersection points from reduced cv2.HoughLines output on reference_img's Canny edges
    """
    img = cv2.Canny(cv2.resize(cv2.imread(reference_img), None, fx=16, fy=16), THRESHOLD1_CANNY, THRESHOLD2_CANNY)
    lines = np.squeeze(cv2.HoughLines(img, RHO, THETA, THRESHOLD_HOUGH))
    img, points = reduce_lines(lines, img)
    return points


def generate_keypoint_profile(reference_img):
    """
    Generate a point profile of a reference image from ORB generated keypoints
    :param reference_img: file path of reference image
    :return:
        kps: array containing intersection points from cv2.ORB output on reference_img
    """
    kp, des = orb.detectAndCompute(reference_img, None)
    kps = np.array([kp_.pt for kp_ in kp])
    return kps


def plot_points(points1, points2):
    """
    Visualization helper method that plots two sets of point profiles against each other
    :param points1: first point profile
    :param points2: second point profile
    :return:
        img: Image with point profiles overlaid on top of each other
    """
    # Create a black background image
    img = np.zeros((500, 500, 3), dtype=np.uint8)

    # Scale the normalized points to image dimensions
    height, width = img.shape[:2]
    scaled_points1 = ((points1 + 1) * 0.5 * np.array([width, height])).astype(np.int32) if len(points1) else points1
    scaled_points2 = ((points2 + 1) * 0.5 * np.array([width, height])).astype(np.int32) if len(points2) else points2

    # Plot the first set of points as blue circles
    for point in scaled_points1:
        x, y = point
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(img, tuple(point), 5, (255, 0, 0), -1)

    # Plot the second set of points as green circles
    for point in scaled_points2:
        x, y = point
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(img, tuple(point), 5, (0, 255, 0), -1)

    # Display the image
    return img


def find_intersection(image, center, angle, length, color, circle_center, circle_radius, draw, thickness=2):
    """
    Generates a line from the center of specified bounding box in the direction of specified angle. Returns intersection point with tracker circle if one exists
    :param image: image to overlay direction line onto
    :param center: center of bounding box (point from which direction line originates from)
    :param angle: angle at which direction line is directed
    :param length: length of direction line (if no intersection point exists)
    :param color: color to draw direction line in
    :param circle_center: center of tracker circle
    :param circle_radius: radius of tracker circle
    :param draw: boolean determining whether generated line should be displayed
    :param thickness: thickness of direction line
    :return:
        intersect_point: intersection point against specified circle if it exists, else None
    """
    if angle is None:
        return None

    angle = angle - 90

    end_point = (
        int(center[0] + length * np.cos(np.deg2rad(angle))),
        int(center[1] + length * np.sin(np.deg2rad(angle)))
    )

    center = np.array(center)
    circle_center = np.array(circle_center)

    angle_rad = np.deg2rad(angle)

    # Calculate direction vector of the line
    line_direction = np.array([np.cos(angle_rad), np.sin(angle_rad)])

    # Calculate vector from circle center to line origin
    circle_to_origin = center - circle_center

    # Calculate the components of the quadratic equation
    a = np.dot(line_direction, line_direction)
    b = 2 * np.dot(circle_to_origin, line_direction)
    c = np.dot(circle_to_origin, circle_to_origin) - circle_radius ** 2

    # Calculate the discriminant
    discriminant = b ** 2 - 4 * a * c

    # Check if there are intersections
    if discriminant < 0:
        # No intersections
        intersect_points = []
    else:
        # Calculate the intersection points
        sqrt_discriminant = np.sqrt(discriminant)
        t1 = (-b + sqrt_discriminant) / (2 * a)
        t2 = (-b - sqrt_discriminant) / (2 * a)
        intersection1 = center + t1 * line_direction
        intersection2 = center + t2 * line_direction
        intersect_points = [intersection1, intersection2]

    intersect_point = find_closest_point(end_point, intersect_points)
    if draw:
        if intersect_point is not None:
            # If intersection point exists, extend the line to it
            x, y = intersect_point
            x, y = int(x), int(y)
            cv2.line(image, tuple(center), (x, y), color, thickness)
            cv2.circle(image, (x, y), radius=5, color=color, thickness=-1)
            cv2.putText(image, f"({x}, {y})", (x, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
                        cv2.LINE_AA)
            return (int(x), int(y))
        else:
            # Otherwise, draw the line as before
            cv2.line(image, center, end_point, color, thickness)
            return None


def find_closest_point(reference, candidates):
    """
    Determine which point from candidates array is closest to reference point. Used to choose between multiple line-circle intersections
    :param reference: point from which candidate point distance is measured from
    :param candidates: points to test against reference
    :return:
        candidate point with smallest Euclidean distance from reference point
    """
    if not candidates:
        return None

    reference = np.array(reference)
    candidates = np.array(candidates)

    distances = np.linalg.norm(candidates - reference, axis=1)
    closest_index = np.argmin(distances)

    return tuple(candidates[closest_index])


def circular_distance(angle1, angle2):
    """
    Calculate the absolute difference between the angles
    :param angle1:
    :param angle2:
    :return: absolute difference between angle1 and angle2
    """
    diff = abs(angle1 - angle2)

    diff = diff % 360
    return 360 - diff if diff >= 180 else diff


def rgb_to_lab(input_color):
    """
    Converts input RGB color (r, g, b) to lab format for use in skimage.color analysis
    :param input_color: RGB input color code
    :return:
        lab: converted lab color code
    """
    num = 0
    rgb = [0, 0, 0]

    for value in input_color:
        value = float(value) / 255

        if value > 0.04045:
            value = ((value + 0.055) / 1.055) ** 2.4
        else:
            value = value / 12.92

        rgb[num] = value * 100
        num = num + 1

    xyz = [0, 0, 0, ]

    x = rgb[0] * 0.4124 + rgb[1] * 0.3576 + rgb[2] * 0.1805
    y = rgb[0] * 0.2126 + rgb[1] * 0.7152 + rgb[2] * 0.0722
    z = rgb[0] * 0.0193 + rgb[1] * 0.1192 + rgb[2] * 0.9505
    xyz[0] = round(x, 4)
    xyz[1] = round(y, 4)
    xyz[2] = round(z, 4)

    xyz[0] = float(xyz[0]) / 95.047         # ref_x =  95.047
    xyz[1] = float(xyz[1]) / 100.0          # ref_y = 100.000
    xyz[2] = float(xyz[2]) / 108.883        # ref_z = 108.883

    num = 0
    for value in xyz:
        if value > 0.008856:
            value = value ** 0.3333333333333333
        else:
            value = (7.787 * value) + (16 / 116)

        xyz[num] = value
        num = num + 1

    lab = [0, 0, 0]

    l = (116 * xyz[1]) - 16
    a = 500 * (xyz[0] - xyz[1])
    b = 200 * (xyz[1] - xyz[2])

    lab[0] = round(l, 4)
    lab[1] = round(a, 4)
    lab[2] = round(b, 4)

    return lab


def count_pixels(image, color1, color2, threshold=25.0, dist_func=deltaE_ciede94):
    """
    Return number of pixels in image within threshold distance of color1 and color2's lab representations
    :param image: cv2 BGR image to analyze
    :param color1: first color to get counts for
    :param color2: second color to get counts for
    :param threshold: maximum allowed distance of a pixel's lab representation from either color to be included in count
    :param dist_func: distance calculation to be used for comparing pixel coloration to
    :return:
        count1, count2: integer tallies of how many pixels corresponding to each color exist in the image
    """

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    lab_image = rgb2lab(image)

    color1_lab = np.array(rgb_to_lab(color1)).reshape(1, 1, 3)
    color2_lab = np.array(rgb_to_lab(color2)).reshape(1, 1, 3)

    dist1 = dist_func(lab_image, color1_lab)
    dist2 = dist_func(lab_image, color2_lab)

    count1 = np.sum(dist1 < threshold)
    count2 = np.sum(dist2 < threshold)

    return color1 if count1 > count2 else color2

def plot_path(image, points, color):
    try:
        if len(points) > 1:
            indices = np.round(np.linspace(0, len(points) - 1, int(len(points) / np.log(len(points))))).astype(int)
            points = [points[i] for i in indices]

            pts = np.array(points, dtype=np.float32)

            tck, _ = splprep(pts.T, u=None, s=50.0)

            u_new = np.linspace(0, 1, 1000)
            x_new, y_new = splev(u_new, tck, der=0)

            pts_new = np.column_stack((x_new, y_new)).astype(np.int32)
            cv2.polylines(image, [pts_new], False, color, 2)
    except TypeError as e:
        print(e)

    return image


class RunningAverage:
    def __init__(self, threshold):
        self.threshold = threshold
        self.angles = []

    def update(self, angle, distance):
        if distance <= self.threshold:
            self.angles.append(angle)
            if len(self.angles) > 5:
                self.angles.pop(0)

    def get_angle(self):
        if self.angles:
            # Convert angles to radians
            radians = [math.radians(a) for a in self.angles]

            # Calculate the average angle in radians
            sum_sin = sum(math.sin(a) for a in radians)
            sum_cos = sum(math.cos(a) for a in radians)
            average_angle_rad = math.atan2(sum_sin, sum_cos)

            # Convert the average angle back to degrees
            average_angle_deg = math.degrees(average_angle_rad)

            # Ensure the angle is within the range of 0 to 360
            if average_angle_deg < 0:
                average_angle_deg += 360

            return average_angle_deg
        return None


def remove_files(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
