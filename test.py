import cv2
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
import argparse
from tools import determine_angle_kps, generate_keypoint_profile

parser = argparse.ArgumentParser(description='Script to print file path.')
parser.add_argument('--file_path', type=str, help='Path to the file')
args = parser.parse_args()
file_path = args.file_path


def assign_points(keypoints_1, keypoints_2):
    # Calculate pairwise Euclidean distances between keypoints
    keypoint_cap = max(len(keypoints_1), len(keypoints_2))#int(min(len(keypoints_1), len(keypoints_2)) * 1)
    distance_matrix = cdist(keypoints_1, keypoints_2)

    # Find the optimal assignment using the Hungarian algorithm
    row_indices, col_indices = linear_sum_assignment(distance_matrix)

    # Select the keypoint pairs with the lowest distances
    selected_keypoints_1 = keypoints_1[row_indices[:keypoint_cap]]
    selected_keypoints_2 = keypoints_2[col_indices[:keypoint_cap]]

    # Calculate the total cost of the selected pairs
    total_cost = distance_matrix[row_indices[:keypoint_cap], col_indices[:keypoint_cap]].sum()

    return selected_keypoints_1, selected_keypoints_2, total_cost / len(selected_keypoints_1)


def equalize_points(set1, set2):
    if len(set1) < len(set2):
        # We have more points in set2, so we want to cluster them into len(set1) clusters
        kmeans = KMeans(n_clusters=len(set1))
        kmeans.fit(set2)
        # The new points are the centroids of the clusters
        set2 = kmeans.cluster_centers_
    elif len(set1) > len(set2):
        # We have more points in set1, so we want to cluster them into len(set2) clusters
        kmeans = KMeans(n_clusters=len(set2))
        kmeans.fit(set1)
        # The new points are the centroids of the clusters
        set1 = kmeans.cluster_centers_
    return set1, set2

scale = 200
coefficient = 2



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

def expand_keypoints(points, scale):
    # Normalize keypoints
    normalized = scale * points / np.ptp(points, axis=0)

    # Shift to center
    normalized += [scale * coefficient / 2, scale * coefficient / 2]

    # Round and convert to int for plotting
    return np.round(normalized).astype(int)


def condense_keypoints(keypoints, threshold):
    condensed_keypoints = []
    num_keypoints = len(keypoints)

    # Calculate the areas of keypoints
    areas = [kp.size * kp.size for kp in keypoints]

    for i in range(num_keypoints):
        overlap = False

        # Check for overlap with other keypoints
        for j in range(i + 1, num_keypoints):
            dx = keypoints[i].pt[0] - keypoints[j].pt[0]
            dy = keypoints[i].pt[1] - keypoints[j].pt[1]
            dist = dx * dx + dy * dy

            # Calculate the overlapping area
            overlapping_area = min(areas[i], areas[j])

            # Check if the keypoints overlap
            if dist < overlapping_area * threshold:
                overlap = True
                break

        # Add the keypoint if no overlap is detected
        if not overlap:
            condensed_keypoints.append(keypoints[i])

    return condensed_keypoints


# Load the images
img1 = cv2.imread(f'dataset_test/precanny{file_path}.png', 0)  # queryImage
img2 = cv2.imread('plane_vert.png', 0)

# Normalize keypoints
best_cost = np.inf
angle = 0

kps = generate_keypoint_profile(img1)
kps_tracker = generate_keypoint_profile(img2)
for i in range(360):
    kp1_normalized = normalize_keypoints(kps, 0)
    kp2_normalized = normalize_keypoints(kps_tracker, i)

    set1 = np.array(kp1_normalized)
    set2 = np.array(kp2_normalized)

    kp1_, kp2_, total_cost = assign_points(set1, set2)
    if total_cost < best_cost:
        best_cost = total_cost
        kp1 = kp1_
        kp2 = kp2_
        angle = i
    print(i, total_cost)

print(f'Total cost: {best_cost}, Angle: {angle}')
distance, angle = determine_angle_kps(img1, kps_tracker)
print(f"{distance}, {angle}")

# Create black canvas
canvas = np.zeros((coefficient * scale, coefficient * scale, 3), dtype="uint8")
print(len(kp1))
print(len(kp2))

# Draw normalized keypoints of image 1 as blue circles
kp1_normalized = expand_keypoints(normalize_keypoints(kp1, 0), scale)
kp2_normalized = expand_keypoints(normalize_keypoints(kp2, 0), scale)

for x, y in kp1_normalized:
    cv2.circle(canvas, (x, y), 3, (255, 0, 0), -1)

# Draw normalized keypoints of image 2 as green circles
for x, y in kp2_normalized:
    cv2.circle(canvas, (x, y), 3, (0, 255, 0), -1)



# Display the canvas
cv2.imshow('Keypoints', canvas)
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)
# kp1 = condense_keypoints(kp1, 0.8)
# kp2 = condense_keypoints(kp2, 0.8)
img1_keypoints = cv2.drawKeypoints(img1, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img2_keypoints = cv2.drawKeypoints(img2, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('Reduced Keypoints 1', img1_keypoints)
cv2.imshow('Reduced Keypoints 2', img2_keypoints)

cv2.waitKey(0)
cv2.destroyAllWindows()
