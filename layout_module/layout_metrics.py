import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from skimage.feature import hog
from skimage import color, io

def perturb_layout(layout, target_bbox_num=5, position_variation=50, size_variation=50):
    """
    Apply slight variations to a layout while maintaining its similarity.

    Args:
    - layout (list): List of tuples, each containing a description and bounding box coordinates.
    - position_variation (int): Maximum number of pixels the position can vary by.
    - size_variation (int): Maximum number of pixels the size can vary by.

    Returns:
    - New layout with perturbations.
    """
    new_layout = []
    if len(layout) > target_bbox_num:
        rand_idx = random.sample(range(1, len(layout)), target_bbox_num)
        layout = [layout[i] for i in rand_idx]
    for bbox in layout:
        x, y, w, h = bbox
        # Apply random variations
        x += random.randint(-position_variation, position_variation) if x-position_variation >= 0 else random.randint(0, position_variation)
        y += random.randint(-position_variation, position_variation) if y-position_variation >= 0 else random.randint(0, position_variation)
        w += random.randint(-size_variation, size_variation) if w-size_variation >= 50 else random.randint(50, size_variation)
        h += random.randint(-size_variation, size_variation) if h-size_variation >= 50 else random.randint(50, size_variation)
        if x >=0 and y>=0 and w>=50 and h>=50:
            if x > 512: x = 512
            if y > 512: y = 512
            if x+w > 512: w = 512-x
            if y+h > 512: h = 512-y
            new_layout.append((x, y, w, h))
            
    return new_layout

# IoU + Distance
def compute_center(box):
    """박스의 중심점 계산"""
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2

def compute_distance(box1, box2):
    """두 박스의 중심점 간의 유클리디안 거리 계산"""
    x1, y1 = compute_center(box1)
    x2, y2 = compute_center(box2)
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def match_boxes(gt_boxes, pred_boxes):
    """각 GT box에 대해 가장 가까운 predicted box와 매칭"""
    matches = []
    used_preds = set()

    for gt_box in gt_boxes:
        min_distance = float('inf')
        best_match = None

        for pred_box in pred_boxes:
            if tuple(pred_box) in used_preds:
                continue

            distance = compute_distance(gt_box, pred_box)
            if distance < min_distance:
                min_distance = distance
                best_match = pred_box

        if best_match:
            matches.append((gt_box, best_match))
            used_preds.add(tuple(best_match))

    return matches

def compute_iou(box1, box2):
    """
    두 bounding box의 IoU를 계산합니다.
    
    각 박스는 (x1, y1, x2, y2) 형태의 좌표로 주어져야 합니다.
    (x1, y1)은 박스의 왼쪽 상단 좌표이고, (x2, y2)는 오른쪽 하단 좌표입니다.
    """
    
    # 교차 영역의 좌표를 계산
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    # 교차 영역의 넓이와 높이를 계산
    width_inter = max(0, x2_inter - x1_inter)
    height_inter = max(0, y2_inter - y1_inter)
    
    # 교차 영역 (intersection)의 면적 계산
    area_inter = width_inter * height_inter
    
    # 각 박스의 면적 계산
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # 합집합 (union)의 면적 계산
    area_union = area_box1 + area_box2 - area_inter
    
    # IoU 계산
    iou = area_inter / area_union if area_union != 0 else 0
    
    return iou

# HOG similarity
def get_layout_image(boxes):
    fig = plt.figure(figsize=(8, 8))
    bbox_coord = show_boxes(boxes)
    # https://stackoverflow.com/questions/7821518/save-plot-to-numpy-array
    # If we haven't already shown or saved the plot, then we need to
    # draw the figure first...
    fig.canvas.draw()

    # Now we can save it to a numpy array.
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.clf()
    return data, bbox_coord

def show_boxes(gen_boxes):
    anns = [{'bbox': gen_box} for gen_box in gen_boxes]

    # White background (to allow line to show on the edge)
    I = np.ones((width+4, height+4, 3), dtype=np.uint8) * 255

    plt.imshow(I)
    plt.axis('off')

    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    bbox_coord = []
    for ann in anns:
        c = (np.random.random((1, 3))*0.6+0.4)
        [bbox_x, bbox_y, bbox_w, bbox_h] = ann['bbox']
        bbox_coord.append([bbox_x, bbox_y, bbox_x+bbox_w, bbox_y+bbox_h])
        poly = [[bbox_x, bbox_y], [bbox_x, bbox_y+bbox_h],
                [bbox_x+bbox_w, bbox_y+bbox_h], [bbox_x+bbox_w, bbox_y]]
        np_poly = np.array(poly).reshape((4, 2))
        polygons.append(Polygon(np_poly))
        color.append(c)

    p = PatchCollection(polygons, facecolor='none',
                        edgecolors=color, linewidths=2)
    ax.add_collection(p)

    return bbox_coord

def cosine_similarity(vecA, vecB):
    """
    Compute cosine similarity between two vectors.
    """
    dot_product = np.dot(vecA, vecB)
    normA = np.linalg.norm(vecA)
    normB = np.linalg.norm(vecB)
    return dot_product / (normA * normB)

def compute_hog_features(image_path):
    """
    Compute HOG features for an image.
    """
    image = io.imread(image_path)
    image_gray = color.rgb2gray(image)
    
    # Extract HOG features
    features, _ = hog(image_gray, visualize=True, block_norm='L2-Hys', pixels_per_cell=(8, 8))
    
    return features

def hog_similarity(image_path1, image_path2):
    """
    Compute HOG similarity between two images using cosine similarity.
    """
    
    hog_features1 = compute_hog_features(image_path1)
    hog_features2 = compute_hog_features(image_path2)
    print(f'HOG1:{hog_features1} HOG2:{hog_features2}')
    
    return cosine_similarity(hog_features1, hog_features2)


def bounding_box_to_vector(bbox):
    """
    Convert a bounding box [x, y, width, height] to a vector.
    """
    return np.array([bbox[0] + bbox[2] / 2,  # Center x
                     bbox[1] + bbox[3] / 2,  # Center y
                     bbox[2],               # Width
                     bbox[3]])              # Height

def bbox_similarity(bbox1, bbox2):
    """
    Compute cosine similarity between two bounding boxes.
    """
    bbox1 = [bbox1[0], bbox1[1], bbox1[2]-bbox1[0], bbox1[3]-bbox1[1]]
    bbox2 = [bbox2[0], bbox2[1], bbox2[2]-bbox2[0], bbox2[3]-bbox2[1]]
    vec1 = bounding_box_to_vector(bbox1)
    vec2 = bounding_box_to_vector(bbox2)
    
    return cosine_similarity(vec1, vec2)

def xywh_to_xyxy(layout):
    xyxy_layout = []
    for bbox in layout:
        c = (np.random.random((1, 3))*0.6+0.4)
        [bbox_x, bbox_y, bbox_w, bbox_h] = bbox
        xyxy_layout.append([bbox_x, bbox_y, bbox_x+bbox_w, bbox_y+bbox_h])
    return xyxy_layout

# Hyperparameters
height, width = 512, 512
diagnoal = np.sqrt(height**2 + width**2) # for distance normalization

alpha, beta = 0.5, 0.5

def cal_layout_sim(old_layout, new_layout):

    bboxes_xywh = [old_layout, new_layout]
    bboxes_xyxy = [xywh_to_xyxy(old_layout), xywh_to_xyxy(new_layout)]

    # IoU + Distance
    sim = 0
    matched_pairs = match_boxes(bboxes_xyxy[0], bboxes_xyxy[1])
    for box1, box2 in matched_pairs:
        iou = compute_iou(box1, box2)
        dis = compute_distance(box1, box2) / diagnoal
        sim += alpha * iou + beta * (1-dis)
    sim /= len(matched_pairs)

    return sim

def generate_layouts(old_layout, recommends_num=10, target_bbox_num=5, position_variation=50, size_variation=50):
    '''
        recommend_layouts: Recommend most similar layout list to given

        old_layout = [(60, 143, 100, 126),(265, 193, 190, 210)] each tuple is single xywh formatted bounding box
        return => List contained 10 xywh formatted layout
    '''
    if target_bbox_num > len(old_layout):
        raise ValueError('Target bbox number cannot be bigger than number of bboxes of the old layout')
    recomms = []
    sample_layouts = [perturb_layout(old_layout, target_bbox_num=target_bbox_num, position_variation=position_variation, size_variation=size_variation) for _ in range(recommends_num)]
    # all sample layouts are xywh format.
    for sample_layout in sample_layouts:
        sim = cal_layout_sim(old_layout, sample_layout)
        recomms.append([sim, sample_layout])
    
    highrecomms = sorted(recomms, key=lambda x:x[0], reverse=True)

    return list(map(lambda x: x[1], highrecomms))