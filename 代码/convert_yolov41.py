from rknn.api import RKNN
import cv2
import numpy as np
from scipy.special import expit as sigmoid

# 更新参数
INPUT_SIZE = 416
CLASSES = ['mouth_opened', 'mouth_closed', 'eyes_open', 'eyes_closed', 'face']

# 分层锚框定义 (根据YOLOv4-tiny结构调整)
ANCHORS_PER_LAYER = [
    [[81, 82], [135, 169], [344, 319]],  # 13x13 层 (对应输出0)
    [[23, 27], [37, 58], [81, 82]]  # 26x26 层 (对应输出1)
]
STRIDES = [INPUT_SIZE // 13, INPUT_SIZE // 26]  # 32和16


# YOLO输出解码函数
def decode_yolo_output(outputs, confidence_thresh=0.5, iou_thresh=0.5):
    boxes, confidences, class_ids = [], [], []

    for i, output in enumerate(outputs):
        # 输出形状: (1, 30, grid, grid) -> 调整为 (grid, grid, 3, 10)
        grid_size = output.shape[2]
        predictions = output[0].transpose((1, 2, 0))  # (30, grid, grid) -> (grid, grid, 30)
        predictions = predictions.reshape((grid_size, grid_size, 3, 10))  # (grid, grid, 3, 10)

        anchors = ANCHORS_PER_LAYER[i]
        stride = STRIDES[i]

        for y in range(grid_size):
            for x in range(grid_size):
                for a in range(3):
                    # 获取预测值
                    tx, ty, tw, th, obj, *class_probs = predictions[y, x, a]

                    # 应用sigmoid激活
                    tx = sigmoid(tx)
                    ty = sigmoid(ty)
                    obj = sigmoid(obj)
                    class_probs = sigmoid(class_probs)

                    # 计算边界框坐标
                    bx = (tx + x) * stride
                    by = (ty + y) * stride
                    bw = anchors[a][0] * np.exp(tw)
                    bh = anchors[a][1] * np.exp(th)

                    # 转换为(x1, y1, x2, y2)格式
                    x1 = int(bx - bw / 2)
                    y1 = int(by - bh / 2)
                    x2 = int(bx + bw / 2)
                    y2 = int(by + bh / 2)

                    # 获取类别和置信度
                    class_id = np.argmax(class_probs)
                    confidence = obj * class_probs[class_id]

                    # 过滤低置信度检测
                    if confidence > confidence_thresh:
                        boxes.append([x1, y1, x2, y2])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

    # 应用NMS
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_thresh, iou_thresh)

    results = []
    for idx in indices:
        if isinstance(idx, np.ndarray) or isinstance(idx, list):
            for i in idx:
                results.append({
                    'box': boxes[i],
                    'confidence': confidences[i],
                    'class_id': class_ids[i],
                    'class_name': CLASSES[class_ids[i]]
                })
        else:
            results.append({
                'box': boxes[idx],
                'confidence': confidences[idx],
                'class_id': class_ids[idx],
                'class_name': CLASSES[class_ids[idx]]
            })

    return results


def main():
    rknn = RKNN()

    print('--> Configuring model')
    rknn.config(
        target_platform='rk3588',
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        quant_img_RGB2BGR=False,
        optimization_level=3
    )

    print('--> Loading Darknet model')
    ret = rknn.load_darknet(
        model='./yolov4-tiny_obj.cfg',
        weight='./yolov4-tiny_obj_best.weights',
    )
    if ret != 0:
        print('Load Darknet model failed!')
        rknn.release()
        exit(ret)

    print('--> Building model')
    ret = rknn.build(do_quantization=False)
    if ret != 0:
        print('Build model failed!')
        rknn.release()
        exit(ret)

    print('--> Export RKNN model')
    ret = rknn.export_rknn('./yolov4-tiny2.rknn')
    if ret != 0:
        print('Export RKNN model failed!')
        rknn.release()
        exit(ret)

    # 验证模型
    print('--> Init runtime')
    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime failed!')
        rknn.release()
        exit(ret)

    # 测试推理
    print('--> Test inference')
    img = cv2.imread('testt1.jpg')

    # 确保图片存在
    if img is None:
        print("Error: Could not load image")
        rknn.release()
        exit(1)

    orig_img = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))

    # 正确预处理输入 (添加batch维度)
    img_input = np.expand_dims(img, 0)

    outputs = rknn.inference(inputs=[img_input])
    print(f'Number of outputs: {len(outputs)}')

    # 解码输出
    detections = decode_yolo_output(outputs, confidence_thresh=0.5, iou_thresh=0.5)

    # 绘制检测结果
    scale_x = orig_img.shape[1] / INPUT_SIZE
    scale_y = orig_img.shape[0] / INPUT_SIZE

    for det in detections:
        x1, y1, x2, y2 = det['box']
        # 缩放回原始图像尺寸
        x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
        y1, y2 = int(y1 * scale_y), int(y2 * scale_y)

        # 绘制边界框和标签
        label = f"{det['class_name']}: {det['confidence']:.2f}"
        cv2.rectangle(orig_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(orig_img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 保存并显示结果
    cv2.imwrite('detection_result.jpg', orig_img)
    cv2.imshow('Detection Results', orig_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print('--> Release model')
    rknn.release()


if __name__ == '__main__':
    main()