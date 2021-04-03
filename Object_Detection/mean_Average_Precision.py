import torch
from collections import Counter
from IoU_in_pytorch import intersection_over_union

def mean_average_precision(pred_boxes, true_boxes, iou_threshold=0.5, box_format='corners', num_classes=20):
    # pred_boxes (list) : [[train_idx, class_pred, prob_score, x1, y1, x2, y2], ... ]
    average_precisions = []
    epsilon = 1e-6

    # 각각의 클래스에 대한 AP를 구합니다.
    for c in range(num_classes):
        detections = []
        ground_truths = []

        # 모델이 c를 검출한 bounding box를 detections에 추가합니다.
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        # 실제 c 인 bounding box를 ground_truths에 추가합니다.
        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # amount_bboxes에 class에 대한 bounding box 개수를 저장합니다.
        # 예를 들어, img 0은 3개의 bboxes를 갖고 있고 img 1은 5개의 bboxes를 갖고 있으면
        # amount_bboexs = {0:3, 1:5} 가 됩니다.
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        # class에 대한 bounding box 개수 만큼 0을 추가합니다.
        # amount_boxes = {0:torch.tensor([0,0,0]), 1:torch.tensor([0,0,0,0,0])}
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # detections를 정확도 높은 순으로 정렬합니다.
        detections.sort(key=lambda x: x[2], reverse=True)

        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        # TP와 FP를 구합니다.
        for detection_idx, detection in enumerate(detections):
            ground_truth_img = [bbox for bbox in ground_truths if bbox[0] == detection[0]]
            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(torch.tensor(detection[3:]),
                                              torch.tensor(gt[3:]),
                                              box_format=box_format)

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            else:
                FP[detection_idx] = 1

        # cumsum은 누적합을 의미합니다.
        # [1, 1, 0, 1, 0] -> [1, 2, 2, 3, 3]
        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]),recalls))

        # torch.trapz(y,x) : x-y 그래프를 적분합니다.
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)
