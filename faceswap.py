import cv2
import dlib
import numpy as np

# Load images
img1 = cv2.imread('image1.jpeg')
img2 = cv2.imread('image2.jpeg')

# Check if images are loaded
if img1 is None:
    raise Exception("Image 1 not loaded. Check the file path.")
if img2 is None:
    raise Exception("Image 2 not loaded. Check the file path.")

# Load the detector and predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Function to extract face landmarks
def get_landmarks(image):
    faces = detector(image)
    if len(faces) == 0:
        return None
    landmarks = predictor(image, faces[0])
    return np.array([(p.x, p.y) for p in landmarks.parts()])

# Get landmarks
landmarks1 = get_landmarks(img1)
landmarks2 = get_landmarks(img2)

if landmarks1 is None:
    raise Exception("Face not detected in image 1")
if landmarks2 is None:
    raise Exception("Face not detected in image 2")

# Function to align faces using eyes
def align_face(image, landmarks):
    # Get the coordinates of the left and right eye
    left_eye = np.mean(landmarks[36:42], axis=0).astype(np.int32)
    right_eye = np.mean(landmarks[42:48], axis=0).astype(np.int32)

    # Compute the angle between the eyes
    dY = right_eye[1] - left_eye[1]
    dX = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dY, dX))

    # Compute the center of the eyes
    eyes_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)

    # Ensure eyes_center is a tuple of integers (x, y)
    eyes_center = tuple(map(int, eyes_center))

    # Get the rotation matrix
    M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)

    # Rotate the image
    (h, w) = image.shape[:2]
    aligned_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)

    return aligned_image

# Align both images
img1_aligned = align_face(img1, landmarks1)
img2_aligned = align_face(img2, landmarks2)

# Get landmarks for aligned images
landmarks1_aligned = get_landmarks(img1_aligned)
landmarks2_aligned = get_landmarks(img2_aligned)

# Convex hull
hull1 = cv2.convexHull(landmarks1_aligned).reshape(-1, 2)
hull2 = cv2.convexHull(landmarks2_aligned).reshape(-1, 2)

# Function to apply Delaunay triangulation
# Function to apply Delaunay triangulation
def delaunay_triangulation(img, landmarks):
    rect = (0, 0, img.shape[1], img.shape[0])
    subdiv = cv2.Subdiv2D(rect)
    for point in landmarks:
        pt = (int(point[0]), int(point[1]))  # Ensure point is a tuple of integers (x, y)
        subdiv.insert(pt)
    triangles = subdiv.getTriangleList()
    triangles = np.array(triangles, dtype=np.int32)
    return triangles

# Get Delaunay triangles
triangles1 = delaunay_triangulation(img1_aligned, landmarks1_aligned)

# Function to warp triangles
def warp_triangle(img1, img2, t1, t2):
    rect1 = cv2.boundingRect(np.float32([t1]))
    rect2 = cv2.boundingRect(np.float32([t2]))

    t1_rect = t1 - rect1[:2]
    t2_rect = t2 - rect2[:2]

    img1_rect = img1[rect1[1]:rect1[1] + rect1[3], rect1[0]:rect1[0] + rect1[2]]
    size = (rect2[2], rect2[3])
    warp_mat = cv2.getAffineTransform(np.float32(t1_rect), np.float32(t2_rect))
    img2_rect = cv2.warpAffine(img1_rect, warp_mat, size, None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    mask = np.zeros((rect2[3], rect2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2_rect), (1.0, 1.0, 1.0), 16, 0)
    img2[rect2[1]:rect2[1] + rect2[3], rect2[0]:rect2[0] + rect2[2]] = img2[rect2[1]:rect2[1] + rect2[3], rect2[0]:rect2[0] + rect2[2]] * (1 - mask) + img2_rect * mask

# Map landmarks to triangles
triangle_indices = []
for t in triangles1:
    indices = []
    for i in range(3):
        pt = (t[2*i], t[2*i+1])
        index = np.where((landmarks1_aligned == pt).all(axis=1))
        if index[0].size == 0:
            continue
        indices.append(index[0][0])
    if len(indices) == 3:
        triangle_indices.append(indices)

# Warp each triangle from img1 to img2
img1_warped = np.copy(img2_aligned)
for indices in triangle_indices:
    t1 = np.float32([landmarks1_aligned[indices[0]], landmarks1_aligned[indices[1]], landmarks1_aligned[indices[2]]])
    t2 = np.float32([landmarks2_aligned[indices[0]], landmarks2_aligned[indices[1]], landmarks2_aligned[indices[2]]])
    warp_triangle(img1_aligned, img1_warped, t1, t2)

# Seamless clone
mask = np.zeros(img2_aligned.shape, dtype=img2_aligned.dtype)
cv2.fillConvexPoly(mask, np.int32(hull2), (255, 255, 255))

r = cv2.boundingRect(np.float32(hull2))
center = (r[0] + r[2] // 2, r[1] + r[3] // 2)

output = cv2.seamlessClone(np.uint8(img1_warped), img2_aligned, mask, center, cv2.NORMAL_CLONE)

# Save the resulting image
cv2.imwrite('downloadswap.jpg', output)

print('Face swapped image saved as downloadswap.jpg')