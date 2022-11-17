# https://github.com/Cartucho/mAP
# https://pypi.org/project/mapcalc/
import mapcalc

def map_score(ground_truth,pred,iou_threshold=0.5,format="default"):
    assert len(ground_truth) == len(pred)
    if format == "coco":
        raise NotImplementedError
    elif format != "default":
        raise ValueError("format must be either 'default' or 'coco'")
    # default
    gt = [{
        "boxes" : ground_truth[i]["bbox"],
        "labels" : ground_truth[i]["category"],
    } for i in range(len(ground_truth))]

    # count map
    total_map = 0
    for i in range(len(pred)):
        total_map += mapcalc.calculate_map(gt[i],pred[i],iou_threshold)
    return total_map / len(pred)