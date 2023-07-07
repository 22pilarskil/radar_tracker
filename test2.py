from tools import determine_angle, generate_point_profile, plot_points, draw_lines, reduce_lines, RunningAverage
import cv2
import argparse
import tools
import time
import numpy as np


parser = argparse.ArgumentParser(description='Script to print file path.')
parser.add_argument('--file_path', type=str, help='Path to the file')
args = parser.parse_args()
file_path = args.file_path

times_frame = []
times_pre = []
times_angle = []
image = cv2.imread("readme_imgs/precanny{}.png".format(file_path))
points_tracker = generate_point_profile("tracker.png")
for i in [file_path] * 20:
    start_frame = time.time()
    angle, points, distance, img_ = determine_angle(image, points_tracker)
    times_frame.append(time.time() - start_frame)
    print(f"ANGLE: {angle}\n")

print(sum(times_frame) / 20)

    # cv2.imshow(str(angle) + ":" + str(distance), img_)
    # cv2.imshow("all_lines", img)
    #
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
# nums = [45]
# point_profile = generate_point_profile("tracker.png")
# for num in nums:
#     edges = cv2.cvtColor(cv2.imread(f"dataset_test/Canny{num}.png"), cv2.COLOR_BGR2GRAY)
#     lines = cv2.HoughLines(edges, tools.rho, tools.theta, tools.threshold)
#     img = draw_lines(edges, lines)
#     cv2.imwrite(f"readme_imgs/houghlines{num}.png", img)
#     _, img, points = reduce_lines(lines, edges)
#     cv2.imwrite(f"readme_imgs/reducedlines{num}.png", img)
#     img = plot_points([], points)
#     cv2.imshow("img", img)
#     cv2.imwrite(f"readme_imgs/points{num}.png", img)
#     img = plot_points(point_profile, points)
#     cv2.imwrite(f"readme_imgs/pointsoverlayed{num}.png", img)
#     _, closest_points, _, _ = tools.determine_angle(edges, point_profile)
#     print(closest_points)
#     img = plot_points(point_profile, closest_points)
#     cv2.imwrite(f"readme_imgs/points_matched{num}.png", img)
#     cv2.waitKey(0)