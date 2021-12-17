import os
import sys
import subprocess

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

import cv2
import copy
import numpy as np
import time
import logging
from PIL import Image
import tools.infer.utility as utility
import tools.infer.predict_rec as predict_rec
import tools.infer.predict_det as predict_det
import tools.infer.predict_cls as predict_cls
from ppocr.utils.utility import get_image_file_list, check_and_read_gif
from ppocr.utils.logging import get_logger
from tools.infer.utility import draw_ocr_box_txt, get_rotate_crop_image
import json
logger = get_logger()

def draw_det_res(dt_boxes, img, img_name, save_path):
    if len(dt_boxes) > 0:
        import cv2
        src_im = img
        for box in dt_boxes:
            box = box.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(src_im, [box], True, color=(255, 255, 0), thickness=2)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_path = os.path.join(save_path, os.path.basename(img_name))
        cv2.imwrite(save_path, src_im)

class TextSystem(object):
    def __init__(self, args):
        if not args.show_log:
            logger.setLevel(logging.INFO)

        # self.text_detector = predict_det.TextDetector(args)
        self.text_recognizer = predict_rec.TextRecognizer(args)
        # self.use_angle_cls = args.use_angle_cls
        self.drop_score = args.drop_score
        # if self.use_angle_cls:
        #     self.text_classifier = predict_cls.TextClassifier(args)

    def print_draw_crop_rec_res(self, img_crop_list, rec_res):
        bbox_num = len(img_crop_list)
        for bno in range(bbox_num):
            cv2.imwrite("./output/img_crop_%d.jpg" % bno, img_crop_list[bno])
            # logger.info(bno, rec_res[bno])

    def __call__(self, img, img_file, cls=True):
        ori_im = img.copy()
        # dt_boxes, elapse = self.text_detector(img)

        # print(img_file)
        # logger.debug("dt_boxes num : {}, elapse : {}".format(
        #     len(dt_boxes), elapse))
        # if dt_boxes is None:
        #     return None, None
        box_file = os.path.basename(img_file)
        # box_file = box_file.split('.')[0]
        box_file = args.input  + box_file + '.txt'
        with open(box_file) as f:
          content = f.readlines()
        content = [x.strip() for x in content] 
        dt_boxes=[]
        for i in content:
          i = i.split(',')
          bb = np.array([[int(i[0]),int(i[1])],[int(i[2]),int(i[3])], [int(i[4]),int(i[5])],[int(i[6]),int(i[7])]],dtype=np.float32)
          min_axis = np.argmin(np.sum(bb, axis=1))
          bb = bb[[min_axis, (min_axis + 1) % 4,(min_axis + 2) % 4, (min_axis + 3) % 4]]
          if abs(bb[0, 0] - bb[1, 0]) < abs(bb[0, 1] - bb[1, 1]):
            bb=bb[[0, 3, 2, 1]]
          dt_boxes.append(bb)
        # print(dt_boxes)
        img_crop_list = []
        dt_boxes = np.array(dt_boxes)
        dt_boxes = sorted_boxes(dt_boxes)
        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            img_crop = get_rotate_crop_image(ori_im, tmp_box)
            img_crop_list.append(img_crop)
        # if self.use_angle_cls and cls:
        #     img_crop_list, angle_list, elapse = self.text_classifier(
        #         img_crop_list)
        #     logger.debug("cls num  : {}, elapse : {}".format(
        #         len(img_crop_list), elapse))
        rec_res, elapse = self.text_recognizer(img_crop_list)
        logger.debug("rec_res num  : {}, elapse : {}".format(
            len(rec_res), elapse))
        # self.print_draw_crop_rec_res(img_crop_list, rec_res)
        filter_boxes, filter_rec_res = [], []
        for box, rec_reuslt in zip(dt_boxes, rec_res):
            text, score = rec_reuslt
            if score >= self.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_reuslt)
        return filter_boxes, filter_rec_res
        


def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        if abs(_boxes[i + 1][0][1] - _boxes[i][0][1]) < 10 and \
                (_boxes[i + 1][0][0] < _boxes[i][0][0]):
            tmp = _boxes[i]
            _boxes[i] = _boxes[i + 1]
            _boxes[i + 1] = tmp
    return _boxes


def main(args):
    image_file_list = get_image_file_list(args.image_dir)
    image_file_list = image_file_list[args.process_id::args.total_process_num]
    text_sys = TextSystem(args)
    is_visualize = False
    font_path = args.vis_font_path
    drop_score = args.drop_score
    save_results = []
    
    # warm up 10 times
    # if args.warmup:
    #     img = np.random.uniform(0, 255, [640, 640, 3]).astype(np.uint8)
    #     for i in range(10):
    #         res = text_sys(img, img_file)

    total_time = 0
    cpu_mem, gpu_mem, gpu_util = 0, 0, 0
    _st = time.time()
    count = 0
    countBbox = 0
    for idx, image_file in enumerate(image_file_list):

        img, flag = check_and_read_gif(image_file)
        if not flag:
            img = cv2.imread(image_file)
        if img is None:
            logger.info("error in loading image:{}".format(image_file))
            continue
        starttime = time.time()
        try:
          dt_boxes, rec_res = text_sys(img,image_file)
        except:
          continue
        elapse = time.time() - starttime
        total_time += elapse
        
        # logger.info(
        #     str(idx) + "  Predict time of %s: %.3fs" % (image_file, elapse))
        for text, score in rec_res:
            logger.info("{}, {:.3f}".format(text, score))
            
        
        if is_visualize:
            image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            boxes = dt_boxes
            txts = [rec_res[i][0] for i in range(len(rec_res))]
            scores = [rec_res[i][1] for i in range(len(rec_res))]

            draw_img = draw_ocr_box_txt(
                image,
                boxes,
                txts,
                scores,
                drop_score=drop_score,
                font_path=font_path)
            draw_img_save = "./inference_results2"
            if not os.path.exists(draw_img_save):
                os.makedirs(draw_img_save)
            if flag:
                image_file = image_file[:-3] + "png"
            cv2.imwrite(
                os.path.join(draw_img_save, os.path.basename(image_file)),
                draw_img[:, :, ::-1])
            logger.info("The visualized image saved in {}".format(
                os.path.join(draw_img_save, os.path.basename(image_file))))
        txts = [rec_res[i][0] for i in range(len(rec_res))]
        save_pred = np.array(dt_boxes).astype(np.int32).tolist()
        label_text=open(args.output+os.path.basename(image_file)+'.txt', "a")
        for i in range(len(save_pred)):
          countBbox += 1
          label = str(save_pred[i][0][0])+','+str(save_pred[i][0][1])+','+str(save_pred[i][1][0])+','+str(save_pred[i][1][1])+','+str(save_pred[i][2][0])+','+str(save_pred[i][2][1])+','+str(save_pred[i][3][0])+','+str(save_pred[i][3][1])+','+txts[i]+"\n"
          label_text.write(label)
        label_text.close()
    logger.info("============ FINISHED #2 RECOGNITION (time elapsed: {}). TOTAL RECOGNIZED BBOX: {} ============".format(str(total_time), str(countBbox)))


if __name__ == "__main__":
    args = utility.parse_args()
    if args.use_mp:
        p_list = []
        total_process_num = args.total_process_num
        for process_id in range(total_process_num):
            cmd = [sys.executable, "-u"] + sys.argv + [
                "--process_id={}".format(process_id),
                "--use_mp={}".format(False)
            ]
            p = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stdout)
            p_list.append(p)
        for p in p_list:
            p.wait()
    else:
        main(args)
