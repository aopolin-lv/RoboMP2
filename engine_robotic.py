import cv2
import os
from PIL import Image, ImageDraw
# import clip
import torch
import copy
import numpy as np

# used for storing the click location
from matplotlib.backend_bases import MouseButton
import matplotlib.pyplot as plt
click_point_x, click_point_y = 0, 0
click_points = []
cid, fig = None, None

import open_clip

from scipy.spatial.transform import Rotation as R
from scipy.optimize import linear_sum_assignment
from model.custom_model import MLLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Loading MLLM...")
MMPROBO = MLLM()

def RecAdj(prompt_images, is_reverse):
    img1_rgb = prompt_images[0]["rgb"]["top"]
    img2_rgb = prompt_images[1]["rgb"]["top"]

    # Determine if img1 and img2 are equal, and if they are equal, then adj is the size
    if (img1_rgb == img2_rgb).all():
        img1_rgb = prompt_images[0]["rgb"]["front"]
        img2_rgb = prompt_images[1]["rgb"]["front"]
        # Cut off obj
        img1_segm = prompt_images[0]["segm"]["front"]
        img2_segm = prompt_images[1]["segm"]["front"]

        cropped_img1 = cropped_prompt_img(img1_rgb, img1_segm)
        cropped_img2 = cropped_prompt_img(img2_rgb, img2_segm)

        h1, w1 = cropped_img1.shape[1:]
        h2, w2 = cropped_img2.shape[1:]

        adj = "smaller" if h1 * w1 < h2 * w2 else "larger"

    else:
        # Cut off obj
        img1_segm = prompt_images[0]["segm"]["top"]
        img2_segm = prompt_images[1]["segm"]["top"]
        cropped_img1 = cropped_prompt_img(img1_rgb, img1_segm)
        cropped_img2 = cropped_prompt_img(img2_rgb, img2_segm)

        adj = "lighter" if np.mean(cropped_img1) > np.mean(cropped_img2) else "darker"

    if is_reverse:
        if adj == "smaller": 
            adj = "larger"
        elif adj == "larger": 
            adj = "smaller"
        elif adj == "lighter": 
            adj = "darker"
        elif adj == "darker": 
            adj = "lighter"
    return adj


def RearrangeActions(source, target, bounds):
    target_obj_full_name = list(target.keys())
    
    actions = []
    for obj_name in target_obj_full_name:
        if obj_name not in list(source.keys()):
            continue
        actions.append(
            PickPlace(pick=source[obj_name][0], place=target[obj_name][0], bounds=bounds)
        )
    return actions


def DistractorActions(source, target, bounds):
    obs_obj_full_name_list = list(source.keys())
    scene_obj_full_name = list(target.keys())
    
    actions = []
    # Move away interfering objects
    distractor_obj = [item for item in obs_obj_full_name_list if item not in scene_obj_full_name]
    for obj_name in distractor_obj:
        actions.append(PickPlace(pick=source[obj_name][0], place=[0, 0, 0, 0], bounds=bounds))

    return actions


def SelectFromScene(obs_obj_dict, scene_obj_dict, texture):
    scene_obj_name_list = list(scene_obj_dict.keys())
    drag_obj_name = [obj_name for obj_name in scene_obj_name_list if texture in obj_name][0]
    return obs_obj_dict[drag_obj_name][0]


def MultiPPWithSame(picks, place, bounds):
    threshold = 10
    actions = []
    for pick in picks:
        if CalculateDistance(pick[:2], place[:2]) < threshold:
            continue
        actions.append(PickPlace(pick=pick, place=place, bounds=bounds))
    return actions


def MultiPPWithConstrain(picks, target_table, bounds, times):
    actions = []
    for idx, pick in enumerate(picks):
        actions.append(PickPlace(pick=pick, place=target_table[idx], bounds=bounds))
        if idx == times:
            break
    return actions


def CalculateValidArea(boundary, constraint):
    assert boundary[0] == constraint[0]
    bound_xmin, bound_xmax, bound_ymin, bound_ymax = calcuate_bbox(boundary)
    constraint_xmin, constraint_xmax, constraint_ymin, constraint_ymax = calcuate_bbox(constraint)

    # Subtracting boundaries and constraints to obtain a valid region
    interval = 10
    valid_xmin = max(constraint_xmin, bound_xmin) + interval
    valid_xmax = min(constraint_xmax, bound_xmax) - interval
    valid_ymin = constraint_ymax + 2
    valid_ymax = bound_ymax

    valid_area = calcuate_bbox_inverse([valid_xmin, valid_xmax, valid_ymin, valid_ymax])

    table_map = {
        0: [valid_xmin + valid_area[3] / 5, valid_area[1], 8, 8],
        1: [valid_area[0], valid_area[1], 8, 8],
        2: [valid_xmax - valid_area[3] / 5, valid_area[1], 8, 8],
    }
    return valid_area, table_map


def CalculateDistance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return (x2 - x1) ** 2 + (y2 - y1) ** 2


def calcuate_bbox(bbox):
    x_center, y_center, h, w = bbox
    xmin, xmax = x_center - int(w/2), x_center + int(w/2)
    ymin, ymax = y_center - int(h/2), y_center + int(h/2)
    return xmin, xmax, ymin, ymax


def calcuate_bbox_inverse(bbox):
    xmin, xmax, ymin, ymax = bbox
    x_center, y_center = (xmin + xmax) / 2, (ymin + ymax) / 2
    h, w = ymax - ymin, xmax - xmin
    return [int(x_center), int(y_center), int(h), int(w)]


def PlanOrder(step0_objs_loc, step1_objs_loc, step2_objs_loc, bounds):

    scene_obj_bbox = {k: v for k, v in step0_objs_loc.items()}
    padding_bbox = [0, 0, 0, 0]

    for k, v in scene_obj_bbox.items():
        if k in step1_objs_loc:
            scene_obj_bbox[k].append(step1_objs_loc[k][0])
        else:
            scene_obj_bbox[k].append(padding_bbox)

    for k, v in scene_obj_bbox.items():
        if k in step2_objs_loc:
            scene_obj_bbox[k].append(step2_objs_loc[k][0])
        else:
            scene_obj_bbox[k].append(padding_bbox)

    step_bbox_list = []
    steps = len(list(scene_obj_bbox.values())[0])
    scene_obj_num = len(scene_obj_bbox.keys())
    for step in range(steps):
        step_bbox_list.append([list(scene_obj_bbox.values())[x][step] for x in range(scene_obj_num)])

    pick_poses, place_poses = [], []
    for step, step_bbox in enumerate(step_bbox_list):
        if step == 0: continue
        for obj_id, obj_bbox in enumerate(step_bbox):
            if obj_bbox != [0, 0, 0, 0] and step_bbox_list[step - 1][obj_id] != [0, 0, 0, 0] \
                and obj_bbox != step_bbox_list[step - 1][obj_id]:
                pick_poses.append(step_bbox_list[step - 1][obj_id])
                place_poses.append(obj_bbox)

    actions = []
    for loc1, loc2 in zip(pick_poses, place_poses):
        actions.append(PickPlace(pick=loc1, place=loc2, bounds=bounds))
    return actions


def RotateAll(obj_list, degree, bounds):
    if len(obj_list) == 4 and type(obj_list[0]) == float: obj_list = [obj_list]
    ACTION = []
    for i in range(len(obj_list)):
        ACTION.append(PickPlace(obj_list[i], obj_list[i], bounds=bounds, yaw_angle=degree, degrees=True))
    return ACTION


def RecDegree(image_prompt):
    images = [(image_prompt[i], image_prompt[i + 1]) for i in range(0, len(image_prompt), 2)]
    degrees = [calculate_rotation_angle(x[0], x[1]) for x in images]
    degrees = [x for x in degrees if x!= 0]

    if degrees:
        # Calculate the average and standard deviation of the list
        mean_value = np.mean(degrees)
        std_deviation = np.std(degrees)

        # Set standard deviation threshold
        threshold = 0.8  # The threshold can be adjusted as needed

        # Remove elements with deviations exceeding the threshold
        filtered_list = [x for x in degrees if abs(x - mean_value) < threshold * std_deviation]

        result = np.mean(filtered_list) if len(filtered_list) != 0 else np.mean(degrees)
        # result = max(degrees)

        print(f"|| 计算的旋转角为: {result}")
        return result
    else:
        return 0

def calculate_rotation_angle(image1, image2):
    img1 = image1.transpose(1,2,0)
    img2 = image2.transpose(1,2,0)

    try:
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)

        # Using BFMatcher for feature matching
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)

        # Sort matching points based on distance
        matches = sorted(matches, key=lambda x: x.distance)

        # Take the first N matching points
        N = 10
        good_matches = matches[:N]

        # Obtain the coordinates of the matching point
        points1 = np.float32([kp1[match.queryIdx].pt for match in good_matches]).reshape(-1, 1, 2)
        points2 = np.float32([kp2[match.trainIdx].pt for match in good_matches]).reshape(-1, 1, 2)
        # Calculate rotation angle
        M, _ = cv2.findHomography(points1, points2, cv2.RANSAC)
        rotation_matrix = M[:2, :2]
        angle_rad = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        angle_deg = np.degrees(angle_rad)
        if angle_deg < 0:
            angle_deg += 180
    except:
        angle_deg = 0

    print("旋转角度为:", angle_deg)

    return angle_deg

def SelectObj(front_image, drag_objs_top_loc, drag_objs_front_loc, novel_adj):
    if len(drag_objs_top_loc) == 4 and (type(drag_objs_top_loc[0]) == float or type(drag_objs_top_loc[0]) == int):
        return drag_objs_top_loc
    if len(drag_objs_front_loc) == 0 or len(drag_objs_top_loc) == 0 :
        return []
    # front_bbox_list = list(front_bbox_list.values())
    if len(drag_objs_front_loc) == 4 and (type(drag_objs_front_loc[0]) == float or type(drag_objs_front_loc[0]) == int):
        drag_objs_front_loc = [drag_objs_front_loc]
    # top_bbox_list = front_bbox_list      # The length of the list is different

    new_top_bbox_list = []
    new_front_bbox_list = []
    # Compare front and top
    for top_bbox in drag_objs_top_loc:
        candidate_front_bbox_list = []
        count = 0
        for front_bbox in drag_objs_front_loc:
            if abs(top_bbox[0]-front_bbox[0]) < 15:
                candidate_front_bbox_list.append(front_bbox)
                count = count + 1
        y_list = [abs(top_bbox[1] - candidate_bbox[1]) for candidate_bbox in candidate_front_bbox_list]
        index = y_list.index(min(y_list))
        new_front_bbox_list.append(candidate_front_bbox_list[index])
        if count != 0:
            new_top_bbox_list.append(top_bbox)
        if count != 1:
            print(count)
    if len(new_top_bbox_list) == 0:
        return []
    
    drag_objs_front_loc = new_front_bbox_list
    drag_objs_top_loc = new_top_bbox_list

    if novel_adj in ["smaller", "larger"]:
        candidate_bbox = [x[2:] for x in drag_objs_front_loc]
        candidate_area = [np.prod(x) for x in candidate_bbox]

        value = min(candidate_area) if novel_adj == "smaller" else max(candidate_area)
        index = candidate_area.index(value)

    elif novel_adj in ["darker", "lighter"]:
        candidate_rgb = []
        for bbox in drag_objs_front_loc:
            x_cetner, y_center, h, w = bbox
            xmin, xmax = int(x_cetner - w/2), int(x_cetner + w/2)
            ymin, ymax = int(y_center - h/2), int(y_center + h/2)
            candidate_rgb.append(front_image.transpose(2,0,1)[:, ymin: ymax+1, xmin: xmax+1])
        candidate_bright = [np.mean(x) for x in candidate_rgb]

        value = min(candidate_bright) if novel_adj == "darker" else max(candidate_bright)
        index = candidate_bright.index(value)

    x_center, y_center, h, w = drag_objs_front_loc[index]
    # min_gap = 10000
    # min_index = 10000
    # for idx, bbox in enumerate(top_bbox_list):
    #     if abs(bbox[0] - x_center) < min_gap:
    #         min_gap = abs(bbox[0] - x_center)
    #         min_index = idx
    # return top_bbox_list[min_index]
    return drag_objs_top_loc[index]


def convery_yaw_to_quaternion(yaw, degrees=True):
    r = R.from_euler("z", -yaw, degrees=degrees)
    return r.as_quat()


def get_indices_of_values_above_threshold(values, threshold):
    filter_values = {i: v for i, v in enumerate(values) if v > threshold}
    sorted_ids = sorted(filter_values, key=filter_values.get, reverse=True)
    return sorted_ids


@torch.no_grad()
def retriev_openclip(elements: list, search_text: str) -> int:
    preprocessed_images = [preprocess(image).to(device) for image in elements]
    txt = tokenizer(search_text).to(device)
    stacked_images = torch.stack(preprocessed_images)
    img_features = model.encode_image(stacked_images)
    text_features = model.encode_text(txt)
    img_features /= img_features.norm(dim=-1, keepdim=True)  # imgs * 1024
    text_features /= text_features.norm(dim=-1, keepdim=True)  # 1 * 1024
    probs = 100.0 * img_features @ text_features.T
    return probs[:, 0].softmax(dim=0)

    # batch_preprocessed_images_list = []
    # for batch_elements in elements:
    #     preprocessed_images = [preprocess(batch_element).to(device) for batch_element in batch_elements]
    #     batch_preprocessed_images_list.append(preprocessed_images)
    # stacked_images = torch.stack(batch_preprocessed_images_list)
    # txt = tokenizer(search_text).to(device)
    # img_features = model.encode_image(stacked_images)
    # text_features = model.encode_text(txt)
    # img_features /= img_features.norm(dim=-1, keepdim=True)  # imgs * 1024
    # text_features /= text_features.norm(dim=-1, keepdim=True)  # 1 * 1024
    # probs = 100.0 * img_features @ text_features.T
    # return probs[:, 0].softmax(dim=0)



@torch.no_grad()
def retriev_clip(elements: list, search_text: str) -> int:
    preprocessed_images = [preprocess(image).to(device) for image in elements]
    tokenized_text = clip.tokenize([search_text]).to(device)
    stacked_images = torch.stack(preprocessed_images)
    image_features = model.encode_image(stacked_images)
    text_features = model.encode_text(tokenized_text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    probs = 100.0 * image_features @ text_features.T
    return probs[:, 0].softmax(dim=0)


def retriev_with_text(elements: list, search_text: str) -> int:
    if engine == "openai":
        return retriev_clip(elements, search_text)
    elif engine == "openclip":
        return retriev_openclip(elements, search_text)
    else:
        raise Exception("Engine not supported")


@torch.no_grad()
def retriev_with_template_image(
    elements: list, search_template_image
) -> int:
    preprocessed_search_template_image = (
        preprocess(search_template_image).unsqueeze(0).to(device)
    )
    search_template_image_features = model.encode_image(
        preprocessed_search_template_image
    )
    search_template_image_features /= search_template_image_features.norm(
        dim=-1, keepdim=True
    )  # 1 * 1024

    preprocessed_images = [preprocess(image).to(device) for image in elements]
    stacked_images = torch.stack(preprocessed_images)
    img_features = model.encode_image(stacked_images)
    img_features /= img_features.norm(dim=-1, keepdim=True)  # imgs * 1024

    probs = 100.0 * img_features @ search_template_image_features.T
    return probs[:, 0].softmax(dim=0)


@torch.no_grad()
def get_img_features(imgs: list):
    preprocessed_imgs = [preprocess(img).to(device) for img in imgs]
    stacked_imgs = torch.stack(preprocessed_imgs)
    img_features = model.encode_image(stacked_imgs)
    img_features /= img_features.norm(dim=-1, keepdim=True)
    return img_features


@torch.no_grad()
def img_sets_similarity(targets: list, templates: list):
    targets_features = get_img_features(targets)
    templates_features = get_img_features(templates)
    similarity = targets_features @ templates_features.T

    rotated_targets = [target.rotate(90) for target in targets]
    rotated_targets_features = get_img_features(rotated_targets)
    rotated_similarity = rotated_targets_features @ templates_features.T

    return (similarity + rotated_similarity) / 2


def get_objs_match(target_objs: list, obs_objs: list):
    sim = img_sets_similarity(target_objs, obs_objs)
    cost = 1 - sim.cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost)
    return row_ind, col_ind


def convert_box_xywh_to_xyxy(box):
    x1 = box[0]
    y1 = box[1]
    x2 = box[0] + box[2]
    y2 = box[1] + box[3]
    return [x1, y1, x2, y2]


def bbox_to_center(bbox):
    return (bbox[0] + bbox[2] / 2), (bbox[1] + bbox[3] / 2)


def pixel_coords_to_action_coords(pixel_coords, pxiel_size=0.0078125):
    x = (pixel_coords[1] - 0.6) / 128 * 0.5 + 0.25
    y = (pixel_coords[0] - 128) / 128 * 0.5
    return x, y


def pixel2action_dict(
    pixel_coords_src,
    pixel_coords_target,
    yaw_angle=None,
    degrees=True,
    pxiel_size=0.0075,
):
    x0, y0 = pixel_coords_to_action_coords(pixel_coords_src, pxiel_size)
    x1, y1 = pixel_coords_to_action_coords(pixel_coords_target, pxiel_size)

    if yaw_angle is not None:
        qua = torch.from_numpy(
            convery_yaw_to_quaternion(yaw_angle, degrees=degrees)
        ).to(torch.float32)
    else:
        qua = torch.tensor([0.0, 0.0, 0.0, 0.9999])

    action = {
        "pose0_position": torch.tensor([x0, y0], dtype=torch.float32),
        "pose1_position": torch.tensor([x1, y1], dtype=torch.float32),
        "pose0_rotation": torch.tensor([0.0, 0.0, 0.0, 0.9999]),
        "pose1_rotation": qua,
    }
    return action


def segment_image(image, segmentation_mask):
    image_array = np.array(image)
    segmented_image_array = np.zeros_like(image_array)
    segmented_image_array[segmentation_mask] = image_array[segmentation_mask]
    segmented_image = Image.fromarray(segmented_image_array)
    black_image = Image.new("RGB", image.size, (0, 0, 0))
    transparency_mask = np.zeros_like(segmentation_mask, dtype=np.uint8)
    transparency_mask[segmentation_mask] = 255
    transparency_mask_image = Image.fromarray(transparency_mask, mode="L")
    black_image.paste(segmented_image, mask=transparency_mask_image)
    return black_image


def clamp_action(actions, action_bound):
    x_min = action_bound["low"][0]
    x_max = action_bound["high"][0]
    y_min = action_bound["low"][1]
    y_max = action_bound["high"][1]
    action = copy.deepcopy(actions)
    action["pose0_position"][0] = np.clip(actions["pose0_position"][0], x_min, x_max)
    action["pose0_position"][1] = np.clip(actions["pose0_position"][1], y_min, y_max)
    action["pose1_position"][0] = np.clip(actions["pose1_position"][0], x_min, x_max)
    action["pose1_position"][1] = np.clip(actions["pose1_position"][1], y_min, y_max)

    for ele in range(4):
        action["pose1_rotation"][ele] = np.clip(actions["pose1_rotation"][ele], -1, 1)
        action["pose0_rotation"][ele] = np.clip(actions["pose0_rotation"][ele], -1, 1)

    return action


def list_remove_element(list_, **kwargs):
    for key in kwargs:
        if "pre_obj" in key:
            try:
                list_.remove(kwargs[key])
            except:
                pass
    return list_


def remove_boundary(image, boundary_length=4):
    image[0 : int(boundary_length / 2), :, :] = 47
    image[-boundary_length:, :, :] = 47
    image[:, 0:boundary_length, :] = 47
    image[:, -boundary_length:, :] = 47
    return image


def nms(bboxes, scores, iou_thresh):
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = x1 + bboxes[:, 2]
    y2 = y1 + bboxes[:, 3]
    areas = (y2 - y1) * (x2 - x1)

    result = []
    index = scores.argsort()[::-1]
    while index.size > 0:
        i = index[0]
        result.append(i)

        x11 = np.maximum(x1[i], x1[index[1:]])
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])
        w = np.maximum(0, x22 - x11 + 1)
        h = np.maximum(0, y22 - y11 + 1)
        overlaps = w * h
        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)
        idx = np.where(ious <= iou_thresh)[0]
        index = index[idx + 1]
    return result


def mask_preprocess(MASKS):
    MASKS_filtered = []
    for MASK in MASKS:
        if MASK["bbox"][2] < 4 or MASK["bbox"][3] < 4:
            continue
        if MASK["bbox"][2] > 100 or MASK["bbox"][3] > 100:
            continue
        if MASK["area"] < 100:
            continue

        mask = MASK["segmentation"]
        mask = mask.astype("uint8") * 255
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)
        if np.count_nonzero(mask) < 50:
            continue  # too small, ignore to avoid empty operation
        MASK["area"] = np.count_nonzero(mask)
        ys, xs = np.nonzero(mask)
        xmin, xmax = np.min(xs), np.max(xs)
        ymin, ymax = np.min(ys), np.max(ys)
        MASK["bbox"] = [xmin, ymin, xmax - xmin, ymax - ymin]
        MASK["segmentation"] = mask.astype("bool")
        MASKS_filtered.append(MASK)

    bboxs = np.asarray([MASK["bbox"] for MASK in MASKS_filtered])
    areas = np.asarray([MASK["area"] for MASK in MASKS_filtered])

    # Determine if bbox is empty
    if bboxs.shape == (0,):
        return False


    result = nms(bboxs, areas, 0.3)
    MASKS_filtered = [MASKS_filtered[i] for i in result]
    return MASKS_filtered


def image_preprocess(image):
    # shadow remove to avoid ghost mask
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.GaussianBlur(gray_img, (3, 3), 0)
    gray = cv2.inRange(gray_img, 47, 150)

    kernel = np.ones((3, 3), np.uint8)
    gray = cv2.dilate(gray, kernel, iterations=1)
    kernel = np.ones((5, 5), np.uint8)
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    image = cv2.bitwise_and(image, image, mask=gray)

    empty = np.ones(image.shape, dtype=np.uint8) * 47  # 47 is the background color
    background = cv2.bitwise_and(empty, empty, mask=cv2.bitwise_not(gray))
    image = cv2.add(image, background)
    return image

def unified_mask_representation(masks):
    """
        input: masks: [N, H, W], numpy.ndarray
        output: masks: list(dict(segmentation, bbox, area)) -> bbox XYWH
    """
    MASKS = [] 
    for mask in masks:
        MASK = {}
        MASK["segmentation"] = mask.astype("bool")
        MASK["area"] = np.count_nonzero(mask)
        ys, xs = np.nonzero(mask)
        xmin, xmax = np.min(xs), np.max(xs)
        ymin, ymax = np.min(ys), np.max(ys)
        MASK["bbox"] = [xmin, ymin, xmax - xmin, ymax - ymin]
        MASKS.append(MASK)
    return MASKS

def get_click_point(event):
    global click_points
    global cid, fig
    if event.button is MouseButton.LEFT:
        point_x = int(event.xdata)
        point_y = int(event.ydata)
        click_points.append([point_x, point_y])
    elif event.button is MouseButton.RIGHT:
        # print('disconnecting callback')
        fig.canvas.mpl_disconnect(cid)


def GetObsImage(obs):
    """Get the current image to start the system.
    Examples:
        top_image, front_image = GetObsImage()
    """
    return np.transpose(obs["rgb"]["top"], (1, 2, 0)), np.transpose(obs["rgb"]["front"], (1, 2, 0))


def GetFrontObsImage(obs, view="front"):
    """Get the current image to start the system.
    Examples:
        image = GetObsImage()
    """
    return np.transpose(obs["rgb"][view], (1, 2, 0))

def GetPromptImages(all_images):
    return all_images.get('prompt_images')
    # return [x["top"] for x in prompt_assets.values()]

def GetSceneImage(all_images):
    return all_images.get('scene')


# 获取prompt中的图片
def get_templates(obs,prompt_assets,task_name):
    templates = {}
    IMAGE_INIT_top = np.transpose(obs["rgb"]["top"], (1, 2, 0))
    IMAGE_INIT_front = np.transpose(obs["rgb"]["front"], (1, 2, 0))
    if task_name == "scene_understanding" or task_name == "rearrange" or task_name == "rearrange_then_restore":
        IMAGE_SCENE = prompt_assets["scene"]["rgb"]["top"]
        templates["scene"] = IMAGE_SCENE
    elif task_name == "novel_adj" or task_name == "novel_adj_and_noun":
        # IMAGE_FRONT = IMAGE_INIT_front
        IMAGE_PROMPT = list(prompt_assets.values())
        templates['prompt_images'] = IMAGE_PROMPT
    elif task_name == "twist" or task_name == "follow_order":
        IMAGE_SCENE_LIST = [x["rgb"]["top"] for x in prompt_assets.values()]
        templates["prompt_images"] = IMAGE_SCENE_LIST
    elif task_name == "follow_motion":
        IMAGE_SCENE_LIST = [x["rgb"]["top"] for x in prompt_assets.values()][1:]
        templates['prompt_images'] = IMAGE_SCENE_LIST
    return templates


def ImageCrop(image, masks):
    # batch_cropped_boxes_list = []
    # batch_used_masked_list = []
    # for one_image in image:
    image = Image.fromarray(image)
    cropped_boxes = []
    used_masks = []
    for mask in masks:
        cropped_boxes.append(
            segment_image(image, mask["segmentation"]).crop(
                convert_box_xywh_to_xyxy(mask["bbox"])
            )
        )
        used_masks.append(mask)
    # batch_cropped_boxes_list.append(cropped_boxes)
    # batch_used_masked_list.append(used_masks)
    return cropped_boxes, used_masks
    # batch_cropped_boxes = np.stack(batch_cropped_boxes_list, axis=0)
    # batch_used_masked = np.stack(batch_used_masked_list, axis=0)
    # return batch_cropped_boxes,batch_used_masked


def CLIPRetrieval(objs, query, **kwargs):
    # batch_obj_idx_list = []
    # for obj in objs:

    if isinstance(query, str):
        scores = retriev_with_text(objs, query)
    else:
        scores_1 = retriev_with_template_image(objs, query)
        scores_2 = retriev_with_template_image(objs, query.rotate(90))
        scores = (scores_1 + scores_2) / 2

    obj_idx = get_indices_of_values_above_threshold(scores, 0.5)
        # batch_obj_idx_list.append(obj_idx)
    # batch_obj_idx = np.stack(batch_obj_idx_list, axis=0)
    return obj_idx
    if len(obj_idx) > 1:
        list_remove_element(obj_idx, **kwargs)
    return obj_idx[0]


def Pixel2Loc(obj, masks):
    return bbox_to_center(masks[obj]["bbox"])


def PickPlace(pick, place, bounds, yaw_angle=None, degrees=True):
    action = pixel2action_dict(pick, place, yaw_angle, degrees)
    action = clamp_action(action, bounds)
    action = {k: np.asarray(v) for k, v in action.items()}
    if isinstance(action, tuple):
        return action[0]
    return action


def cropped_prompt_img(rgb, segm, idx=0):
    ys, xs = np.nonzero(segm == idx)
    xmin, xmax = np.min(xs), np.max(xs)
    ymin, ymax = np.min(ys), np.max(ys)
    x_center, y_center = (xmin + xmax) / 2, (ymin + ymax) / 2
    h, w = ymax - ymin, xmax - xmin
    cropped_img = rgb[:, ymin: ymax+1:, xmin: xmax+1]
    return cropped_img

def GetAllObjectsFromImage(image):
    obs_obj_dict = MMPROBO.generate(image=image, query='all objects in the scene')
    return obs_obj_dict

def GetAllObjectsFromPromptImage(image):
    scene_obj_dict = MMPROBO.generate(image=image, query='all objects in the scene')
    return scene_obj_dict

def GetAllSameTextureObjects(image, query):
    same_texture_objs_loc = MMPROBO.generate(image=image, query=f'all objects with the same texture as {query}')
    return same_texture_objs_loc

def GetAllSameProfileObjects(image, query):
    same_shape_objs_loc = MMPROBO.generate(image=image, query=f'all objects with the same shape as {query}')
    return same_shape_objs_loc

def GetSameShapeObjectsAsObject(image, query):
    query = 'all objects with the same shape as ' + query
    drag_obj_loc = MMPROBO.generate(image=image, query=query)
    return drag_obj_loc

def GetObjectsWtihGivenTexture(image, query):
    query = "all objects with the same texture as " + query + " object"
    drag_obj_loc = MMPROBO.generate(image=image, query=query)
    return drag_obj_loc

def GetDragObjName(prompt_assets):
    drag_key_name = [x for x in list(prompt_assets.keys()) if x.startswith("drag")][0]
    prompt_drag_obj_info = prompt_assets[drag_key_name]["segm"]["obj_info"]
    prompt_drag_obj_name = prompt_drag_obj_info['obj_name']
    return prompt_drag_obj_name

def GetBaseObjName(prompt_assets):
    prompt_base_obj_info = prompt_assets["base_obj"]["segm"]["obj_info"]
    prompt_base_obj_name = prompt_base_obj_info['obj_name']
    return prompt_base_obj_name

def GetSweptObjName(prompt_assets):
    swept_obj_info = prompt_assets['swept_obj']['segm']['obj_info']
    swept_obj_name = swept_obj_info['obj_color'] + ' ' + swept_obj_info['obj_name']
    return swept_obj_name

def GetBoundsObjName(prompt_assets):
    bounds_info = prompt_assets['bounds']['segm']['obj_info']
    bounds_name = bounds_info['obj_color'] + ' ' + bounds_info['obj_name']
    return bounds_name

def GetConstraintObjName(prompt_assets):
    constraint_info = prompt_assets['constraint']['segm']['obj_info']
    constraint_name = constraint_info['obj_color'] + ' ' + constraint_info['obj_name']
    return constraint_name

def GetPromptAssets(prompt_assets):
    return prompt_assets

def GetWholeTask(whole_task):
    return whole_task

def GetReverse(whole_task):
    if "less" in whole_task:
        return True
    else:
        return False
    
def GetStepsLocForObject(image, query):
    step0_loc = MMPROBO.generate(image=image[0], query=query)
    step1_loc = MMPROBO.generate(image=image[1], query=query)
    step2_loc = MMPROBO.generate(image=image[2], query=query)
    return step0_loc,step1_loc,step2_loc

def GetAllObjextsStepsLoc(image):
    step0_loc = MMPROBO.generate(image=image[0], query='all objects in the scene')
    step1_loc = MMPROBO.generate(image=image[1], query='all objects in the scene')
    step2_loc = MMPROBO.generate(image=image[2], query='all objects in the scene')
    return step0_loc,step1_loc,step2_loc

def Sorted(sweep_objs, valid_area):
    if len(sweep_objs) == 4 and (type(sweep_objs[0]) == float or type(sweep_objs[0]) == int):
        return [sweep_objs]
    return sorted(sweep_objs, key=lambda x: CalculateDistance(x[:2], valid_area[:2]))

def GetSweptAndBoundaryName(prompt_assets):
    swept_obj_info = prompt_assets['swept_obj']['segm']['obj_info']
    swept_obj_name = swept_obj_info['obj_color'] + ' ' + swept_obj_info['obj_name']    
    bounds_info = prompt_assets['bounds']['segm']['obj_info']
    bounds_name = bounds_info['obj_color'] + ' ' + bounds_info['obj_name']
    constraint_info = prompt_assets['constraint']['segm']['obj_info']
    constraint_name = constraint_info['obj_color'] + ' ' + constraint_info['obj_name']
    return swept_obj_name,bounds_name,constraint_name

def GetTimes(whole_task):
    quantifier = whole_task.strip().split()[1]
    dic = {
    'one': 1,
    'two': 2,
    'three': 3,
    'four': 4,
    'five': 5,
    'any': 1,
    'all': 100
    }
    times = dic[quantifier]
    return times