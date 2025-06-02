import itertools

def iou(box1, box2):
    # box format: [x1, y1, x2, y2]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union != 0 else 0

def remove_duplicate_boxes(boxes_with_cls, iou_threshold=0.5):
    unique = []
    used = [False] * len(boxes_with_cls)

    for i, (box1, cls1) in enumerate(boxes_with_cls):
        if used[i]:
            continue
        for j, (box2, cls2) in enumerate(boxes_with_cls):
            if i == j or used[j]:
                continue
            if iou(box1, box2) > iou_threshold:
                used[j] = True
        unique.append((box1, cls1))
    return unique