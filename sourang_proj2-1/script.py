import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from numpy import linalg as LA
from itertools import groupby, product
from scipy.spatial.transform import Rotation as Rotats
import matplotlib as mpl


def hough_transform(canny):
    size_x, size_y = canny.shape
    angle = np.arange(0, 181)

    length = int(np.sqrt(size_x**2 + size_y**2))
    Cart_values = np.zeros((length*2, len(angle)))
    height, breadth = np.where(canny != 0)

    for element in range(len(height)):
        y = height[element]
        x = breadth[element]

        for theta in angle:
            d = int((x*np.cos(theta*(math.pi/180)) +
                    y*np.sin(theta*(math.pi/180))))
            Cart_values[d, theta] += 1

    flat_indices = np.argpartition(-Cart_values.ravel(), 30)[:30]
    row_indices, col_indices = np.unravel_index(
        flat_indices, Cart_values.shape)
    position = np.column_stack((row_indices, col_indices))

    return Cart_values, position


# Function to detect corners using hough transform
def corner_detection_function(index):

    empty_list_1 = []
    empty_list_2 = []
    empty_list_3 = []
    length = 2202

    for i in range(len(index)):

        if index[i][0] > length:
            index[i][0] = index[i][0]-(2*length)

        x = -np.cos(index[i][1]*math.pi/180)/(np.sin(index[i][1]*math.pi/180))
        y = (index[i][0])/(np.sin(index[i][1]*math.pi/180))
        empty_list_1.append(x)
        empty_list_2.append(y)
    z = 0
    for one in zip(empty_list_1, empty_list_2):
        y_inter_1, b1 = one
        for two in zip(empty_list_1[z:], empty_list_2[z:]):
            y_inter_2, b2 = two
            if y_inter_2 != y_inter_2 and b1 != b2:
                if -1.15 < (y_inter_2*y_inter_2) < -0.85:
                    mat_1 = np.matrix([[-y_inter_1, 1], [-y_inter_2, 1]])
                    mat_2 = np.matrix([[b1], [b2]])
                    multi = np.linalg.inv(mat_1)@(mat_2)
                    x1 = int(multi[0][0])
                    y1 = int(multi[1][0])
                    coords = (int(x1), int(y1))
                    coord_matrix.append(coords)
    z += 1

    for j in range(len(empty_list_1)-1):
        y_inter_1 = empty_list_1[j]
        y_inter_2 = empty_list_1[j+1]
        b_inter_1 = empty_list_2[j]
        b_inter_2 = empty_list_2[j+1]
        yyy = y_inter_1*y_inter_2
        if -1.10 < (y_inter_1*y_inter_2) < -0.80:
            m_matrix = np.matrix([[-y_inter_1, 1], [-y_inter_2, 1]])
            b_matrix = np.matrix([[b_inter_1], [b_inter_2]])
            coord_matrix = np.linalg.inv(m_matrix) @ b_matrix
            print(coord_matrix)
            x_a = int(coord_matrix[0][0])
            y_a = int(coord_matrix[1][0])
            cords = (x_a, y_a)
            #  print(cords)
            empty_list_3.append(cords)

    return empty_list_3


def obtaining_homography(input):

    length_of_paper = 21.6
    breadth_of_paper = 27.9

    input_list = [(0, 0), (length_of_paper, 0),
                  (0, breadth_of_paper), (length_of_paper, breadth_of_paper)]
    A_matrix = np.array([[input_list[0][0], input_list[0][1], 1, 0, 0, 0, -input[0][0]*input_list[0][0], -input[0][0]*input_list[0][1], -input[0][0]],
                         [0, 0, 0, input_list[0][0], input_list[0][1], 1, -input[0][1] *
                             input_list[0][0], -input[0][1]*input_list[0][1], -input[0][1]],
                         [input_list[1][0], input_list[1][1], 1, 0, 0, 0, -input[1][0] *
                             input_list[1][0], -input[0][0]*input_list[1][1], -input[1][0]],
                         [0, 0, 0, input_list[1][0], input_list[1][1], 1, -input[1][1] *
                             input_list[1][0], -input[0][1]*input_list[1][1], -input[1][1]],
                         [input_list[2][0], input_list[2][1], 1, 0, 0, 0, -input[2][0] *
                             input_list[2][0], -input[2][0]*input_list[2][1], -input[2][0]],
                         [0, 0, 0, input_list[2][0], input_list[2][1], 1, -input[2][1] *
                             input_list[2][0], -input[2][1]*input_list[2][1], -input[2][1]],
                         [input_list[3][0], input_list[3][1], 1, 0, 0, 0, -input[3][0] *
                             input_list[3][0], -input[3][0]*input_list[3][1], -input[3][0]],
                         [0, 0, 0, input_list[3][0], input_list[3][1], 1, -input[3][1]*input_list[3][0], -input[3][1]*input_list[3][1], -input[3][1]]])

    Mid_matrix = np.transpose(A_matrix)@A_matrix
    eigen_value, eigen_vector = LA.eigh(Mid_matrix)
    final_matrix = eigen_vector[:, 0]
    ultimate_matrix = np.array([[final_matrix[0], final_matrix[1], final_matrix[2]],
                                [final_matrix[3], final_matrix[4], final_matrix[5]],
                                [final_matrix[6], final_matrix[7], final_matrix[8]]])
    ultimate_matrix = (1/ultimate_matrix[2, 2])*ultimate_matrix
    return ultimate_matrix


def finding_translation_rotation(homography):
    mat_ein = np.matrix([[1382.58398,	0,	945.743164],
                         [0,	1383.57251,	527.04834],
                         [0, 0, 1]])

    mat_ein_inv = np.linalg.inv(mat_ein)
    rubric = mat_ein_inv@homography

    rubric_1 = rubric[:, 0]
    rubric_2 = rubric[:, 1]

    solu_1 = np.linalg.norm(rubric_1)
    solu_2 = np.linalg.norm(rubric_2)

    mean = (solu_1+solu_2)/2

    new_mat = rubric/mean

    rotation_1 = new_mat[:, 0]
    rotation_2 = new_mat[:, 1]
    translation_matrix = new_mat[:, 2]
    rotation_3 = np.cross(rotation_1, rotation_2, axis=0)

    rotation_matrix = np.hstack((rotation_1, rotation_2, rotation_3))
    return rotation_matrix, translation_matrix


def sequence(seq1, seq2):
    return abs(seq1[0] - seq2[0]) + abs(seq1[1] - seq2[1])


def grouping(input):
    man_tups = [sorted(sub) for sub in product(input, repeat=2)
                if sequence(*sub) <= 20]

    tion = {ele: {ele} for ele in input}
    for tup1, tup2 in man_tups:
        tion[tup1] |= tion[tup2]
        tion[tup2] = tion[tup1]
    res = [[*next(val)] for key, val in groupby(
        sorted(tion.values(), key=id), id)]
    mer = []
    for i in res:
        mx = np.mean(i, axis=0)
        mer.append(mx)
    fm = sorted(mer, key=lambda x: x[0])
    return fm


capture = cv2.VideoCapture('project2.avi')

roll = []
pitch = []
yaw = []

x_ = []
y_ = []
z_ = []

while True:
    isTrue, frame = capture.read()
    if not isTrue:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurring_image = cv2.GaussianBlur(frame, (15, 15), cv2.BORDER_DEFAULT)
    canny_edge_detection = cv2.Canny(blurring_image, 70, 120)
    points, position = hough_transform(canny_edge_detection)
    intercept = corner_detection_function(position)
    mean_value = grouping(intercept)

    if len(mean_value) == 4:
        H = obtaining_homography(mean_value)
        Ro, Tr = finding_translation_rotation(H)
        r = Rotats.from_matrix(Ro)

        roll.append(r.as_euler('zyx', degrees=True)[0])
        pitch.append(r.as_euler('zyx', degrees=True)[1])
        yaw.append(r.as_euler('zyx', degrees=True)[2])

        x_.append(float(Tr[0]))
        y_.append(float(Tr[1]))
        z_.append(float(Tr[2]))
    # axis,sedqq=hough_transform(canny_edge_detection
    for i in mean_value:
        cv2.circle(frame, (int(i[0]), int(i[1])), 5,  (0, 0, 255), -1)

    # finale=obtaining_homography(corners)

    # finding_translation_rotation_solution = finding_translation_rotation(finale)

    cv2.imshow('canny edge detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('d'):
        break

limit = np.arange(1, 127, 1)
fig, ax = plt.subplots(1, 2, figsize=(10, 20))
ax[0].plot(limit, roll, label='roll')
ax[0].plot(limit, pitch, label='pitch')
ax[0].plot(limit, yaw, label='yaw')
ax[0].legend()
ax[1].plot(limit, x_, label='x')
ax[1].plot(limit, y_, label='y')
ax[1].plot(limit, z_, label='z')
ax[1].legend()
plt.show()


cv2.destroyAllWindows()
capture.release()