import math
import random
import win32gui
import win32ui
from ctypes import windll
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

import nms


def check_img_size(img_size, s=32):
    """
         Verify img_size is a multiple of stride s  输入图片长宽必须是s 32的倍数
    """
    new_size = math.ceil(img_size / int(s)) * int(s)  # ceil gs-multiple
    if new_size != img_size:
        print('WARNING: --img-size %g must be multiple of max stride %g, updating to %g' % (img_size, s, new_size))
    return new_size


def screen_shot_by_hwnd_name(hwnd_name):
    """
        按句柄名称获取窗口区域图片
    """
    hwnd = win32gui.FindWindow(hwnd_name, None)
    left, top, right, bot = win32gui.GetWindowRect(hwnd)
    width = right - left
    height = bot - top
    hwndDC = win32gui.GetWindowDC(hwnd)
    mfcDC = win32ui.CreateDCFromHandle(hwndDC)
    saveDC = mfcDC.CreateCompatibleDC()
    saveBitMap = win32ui.CreateBitmap()
    saveBitMap.CreateCompatibleBitmap(mfcDC, width, height)
    saveDC.SelectObject(saveBitMap)
    result = windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 0)
    bmpinfo = saveBitMap.GetInfo()
    bmpstr = saveBitMap.GetBitmapBits(True)

    im = Image.frombuffer(
        "RGB",
        (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
        bmpstr, 'raw', 'BGRX', 0, 1)

    win32gui.DeleteObject(saveBitMap.GetHandle())
    saveDC.DeleteDC()
    mfcDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwndDC)

    if result:
        return im, (left, top, width, height)
    else:
        return None, (0, 0)


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width) 将坐标限制在设定宽高内 (一般使用为原图宽高)
    boxes[:, 0] = np.clip(boxes[:, 0], 0, img_shape[1])  # x1
    boxes[:, 1] = np.clip(boxes[:, 1], 0, img_shape[0])  # y1
    boxes[:, 2] = np.clip(boxes[:, 2], 0, img_shape[1])  # x2
    boxes[:, 3] = np.clip(boxes[:, 3], 0, img_shape[0])  # y2


def draw_reginon(img, region):
    bgr_img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    x, y, w, h = region
    draw_img = cv2.rectangle(bgr_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return draw_img


def show_img(image):
    plt.imshow(image)
    plt.show()


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    '''
    将推理坐标还原到原图  有pad将x y pad 值减去，坐标映射回原图
    :param img1_shape:推理图片大小
    :param coords: 推理位置结果 x1y1x2y2
    :param img0_shape:原图尺寸 [h,w]
    :param ratio_pad:缩放比例， pad尺寸  [[x_ratio, y_ratio],[x_pad,y_pad]]
    :return:
    '''
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def NMS(prediction, conf_thres=0.5, iou_thres=0.5, method='or', iou_method="iou"):
    """
    将 （x,y,w,h,obj_conf,cls1_conf, cls2_conf， cls3_conf, cls4_conf）
    转  （x1,y1,x2,y2,cls_conf, cls）
    :param prediction:
    :param conf_thres:
    :param iou_thres:
    :return:
    """

    def nms_and_clsmerge(prediction, conf_thres=0.001, iou_thres=0.5, method='or', iou_method="iou"):
        # nc = 4  # number of classes
        nc = 15  # number of classes
        output = []
        for class_id in range(nc):
            idx = np.where(prediction[..., 5] == class_id)
            pred = prediction[idx]
            if len(pred):
                pred = nms.non_max_suppression(pred, conf_thres=conf_thres, iou_thres=iou_thres, method=method,
                                               iou_method=iou_method)
                if pred is not None:
                    output.append(pred)
        if not len(output):
            return None
        output = np.concatenate(output, axis=0)
        # 高重叠目标类间合并 （仅保留重叠目标置信度最高的目标）
        output = nms.merge_between_cls(output, iou=0.85, mode="union")
        return output

    nc = prediction[0].shape[1] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates  得到目标置信度大于置信度阈值的目标索引
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)
    # output = [None] * prediction.shape[0]
    output = []
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence  得到单个图片 置信度大于置信度阈值的推理结果 shape=(目标数，9)
        # If none remain process next image  如果该图没有推理结果，转处理下一张图
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf  更新分类置信度  新分类置信度 = 分类置信度 * 目标置信度

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)  注意 这里的conf是 新分类置信度 = 分类置信度 * 目标置信度
        if multi_label:
            # 置信度衰减 -----------------------------------------------------------------------------------------------
            # x_dealed = None
            # for id_b, _ in enumerate(x):
            #     id_c = (x[id_b][5:] > conf_thres).nonzero(as_tuple=False).T[0]  # 得到该推理框大于置信度阈值的类别id
            #     conf = torch.mean(torch.tensor([x[id_b][i+5] for i in id_c], device=x.device))  # 取大于置信度阈值的类比的置信度均值
            #     if torch.isnan(conf):
            #         continue
            #     cls = x[id_b][5:].max(0)[1]  # 该推理框类别  取最大置信度的类别
            #     box_mean = torch.cat((box[id_b], conf.unsqueeze(0), cls.unsqueeze(0))).unsqueeze(0)
            #     if x_dealed is None:
            #         x_dealed = box_mean
            #     else:
            #         x_dealed = torch.cat((x_dealed, box_mean), 0)
            # x = x_dealed
            # ----------------------------------------------------------------------------------------------------------
            # 取最大值方式
            conf = np.max(x[:, 5:], axis=1)
            j = np.argmax(x[:, 5:], axis=1)
            x = np.concatenate((box, conf[..., None], j[..., None]), axis=1)
            # 原处理方式
            # i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            # x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf = np.max(x[:, 5:], axis=1)
            j = np.argmax(x[:, 5:], axis=1)
            x = np.concatenate((box, conf[..., None], j[..., None]), axis=1)

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue
        # 所有类别一起做NMS ，无需类间合并
        # pred = nms.non_max_suppression(x, conf_thres=conf_thres, iou_thres=iou_thres, method=method, iou_method=iou_method)

        # 类别单独NMS后做类间合并
        pred = nms_and_clsmerge(x, conf_thres=conf_thres, iou_thres=iou_thres, method=method, iou_method=iou_method)
        output.append(pred)
    return output


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border  填充边缘
    # wh_ratio = new_unpad[0] / shape[1], new_unpad[1] / shape[0]
    return img, ratio, (dw, dh)


def main():
    # 检测游戏
    hwnd_name = "HLDDZ"
    device = torch.device('cpu')
    weights = "best.pt"
    conf_thres = 0.3
    iou_thres = 0.6
    img_size = 640
    model = torch.load(weights, map_location=torch.device(device))['model'].float().eval()
    names = model.module.names if hasattr(model, "module") else model.names
    print(names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    imgsz = check_img_size(img_size, s=model.stride.max())  # check img_size
    while 1:
        start_time = time.time()
        try:
            raw_im, region = screen_shot_by_hwnd_name(hwnd_name)
        except:
            print("清确保斗地主程序已打开")
            continue
        cv_img = cv2.cvtColor(np.asarray(raw_im), cv2.COLOR_RGB2BGR)
        # resize_img = cv2.resize(cv_img,(img_size,img_size))
        im_array = np.array(cv_img)
        letterbox_img = letterbox(im_array, new_shape=img_size, auto=False)[0]
        img = letterbox_img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img0 = np.ascontiguousarray(img)

        img = torch.from_numpy(img0).to(device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred, _ = model(img, augment=True)
        # Apply NMS
        predition = np.array(pred.to("cpu"))  # torch to numpy
        pred = NMS(predition, conf_thres=conf_thres, iou_thres=iou_thres, method='merge', iou_method="giou")
        # Process detections
        print("Process detections")
        for i, det in enumerate(pred):  # detections per image
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], cv_img.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    # print(xyxy)
                    label = "%s" % (names[int(cls)])
                    plot_one_box(
                        img_name="",
                        i=i,
                        x=xyxy,
                        img=cv_img,
                        label=label,
                        color=colors[int(cls)],
                        line_thickness=3,
                    )

        print("Process detections over")
        end_time = time.time()
        print(end_time-start_time)


def plot_one_box(img_name, i, x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    tl = 1
    img = cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:

        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        img = cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        img = cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    img = cv2.resize(img,(1080,640))
    cv2.imshow("1.png", img)
    if cv2.waitKey(2) & 0xFF == ord('q'):
        pass


if __name__ == '__main__':
    main()
