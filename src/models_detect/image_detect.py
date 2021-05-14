

from tensorflow.keras.models import load_model
import numpy as np
import cv2
import torch
import base64

from src.models_detect.utils_model import attempt_load, non_max_suppression, save_one_box, letterbox, scale_coords
MODEL_PATH="src/dn_models/s_model"
model_compare = load_model(MODEL_PATH)
#model_compare=""



# Initialize
device = torch.device('cpu')
half = device.type != 'cpu'  # half precision only supported on CUDA

# Load model
weights = ['src/dn_models/best.pt']

model = attempt_load(weights, map_location=device)  # load FP32 model
stride = int(model.stride.max())  # model stride

names = model.module.names if hasattr(model, 'module') else model.names  # get class names
img_size = 416

if half:
    model.half()  # to FP16


def detect_products(img_galpao, imgs_products, marcas):
    with torch.no_grad():
        images_list = galpao_products_compare(img_galpao)
        n_elements = len(images_list)

        to_compare = []
        for imgs in imgs_products:
            for img in imgs:
                to_compare.extend(galpao_products_compare(img, True))

    images_list.extend(to_compare)

    groups_thr = []
    groups_compare = []

    n_elements_total = len(images_list)

    for img in imgs_products:
        c_group = [item for item in range(n_elements, n_elements+len(img))]
        groups_compare.append(set(c_group))
    
    for u, item1 in enumerate(images_list):
        if u == []:
            continue

        for v, item2 in enumerate(images_list):

            if u == v or v < u or v == []:
                continue
            else:
                result_compare = model_compare.predict([item1[0], item2[0]])
                if result_compare  >= 0.80:
                    groups_thr.append([u, v])

    groups_thr_np = np.array(groups_thr)
    total_groups = []

    for n in range(n_elements_total):
        select = np.array([groups_thr[idx] for idx in np.argwhere(groups_thr_np==n)[:,0]])
        all_groups = set(select.flatten())
        if len(all_groups):
            total_groups.append(all_groups)

    filter_groups = []
    idx_include = []

    for idx1, gr1 in enumerate(total_groups):
        if idx1 in idx_include:
            continue
        for idx2, gr2 in enumerate(total_groups[1:]):
            result = gr1.intersection(gr2)
            if len(result) > 0:
                gr1 = gr1.union(gr2)
                idx_include.append(idx2+1)
        filter_groups.append(gr1)

    marca_len = 0
    count_groups = []
    results = dict()
    total_count = 0

    for m_compare, items_compare in enumerate(groups_compare):
        count = 0
        for items in filter_groups:
            result = items_compare.intersection(items)
            if len(result) > 0:
                count += len(items) - len(result)
        total_count += count
        results[marcas[m_compare]] = count
    results["concorrentes"] = n_elements - total_count

    #results["test"] = str(filter_groups)
    return results


def galpao_products_compare(img_galpao, select_img=False):
    im_bytes = base64.b64decode(img_galpao)
    im_arr = np.frombuffer(im_bytes, dtype=np.uint8)  # im_arr is one-dim Numpy array
    img0s = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)
    # Padded resize
    img = letterbox(img0s, img_size, stride=stride)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)


    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    img = img.unsqueeze(0)

    pred = model(img, augment=False)[0]
    pred = non_max_suppression(pred, 0.45, 0.3, classes=None, agnostic=True)

    images_detected = []
    images_size = []

    for i, det in enumerate(pred):  # detections per image
    
        im0 = img0s.copy()

        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Write results
            n_= 0
            for *xyxy, conf, cls in reversed(det):

                c = int(cls)  # integer class
                save_dir = "s"
                crp_image = save_one_box(xyxy, img0s, file='', BGR=True, return_image=True)
                crp_image = cv2.resize(crp_image, (160,160))/255.
                

                if select_img:
                    w, h, _ = crp_image.shape
                    images_size.append(w*h)
                
                crp_image = np.expand_dims(crp_image, axis=0)

                images_detected.append([crp_image])
                n_+= 1

        if select_img:
            area_max = max(images_size)
            return [images_detected[images_size.index(area_max)]]

    return images_detected

