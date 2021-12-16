from detectron2.engine.defaults import DefaultPredictor
from detectron2.data.detection_utils import read_image
from adet.config import get_cfg
import time
import glob
import os
import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--confidence_threshold", default=0.4)
    parser.add_argument("--weights", default="")
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--config-file",
        default="/aic2021/AdelaiDet/configs/BAText/TotalText/v2_attn_R_50.yaml",
        metavar="FILE",
        help="path to config file",
    )
    return parser

def prepare_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(['MODEL.WEIGHTS', args.weights])
    # Set score_threshold for builtin models
    threshold = float(args.confidence_threshold)
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    cfg.MODEL.FCOS.INFERENCE_TH_TEST = threshold
    cfg.MODEL.MEInst.INFERENCE_TH_TEST = threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = threshold
    cfg.freeze()
    return cfg

if __name__ == "__main__":
    args = get_parser().parse_args()

    print(args)
    cfg = prepare_cfg(args)
    predictor = DefaultPredictor(cfg)
    
    print('Read predictor successfully. We are ready for predictions')
    
    list_txts = []
    start_time = time.time()
    files = glob.glob(args.input + "*")
    countBbox = 0
    trace = {}
    
    for id, img_path in enumerate(files):
        img = read_image(img_path, format="BGR")
        filename = os.path.basename(img_path)
        predictions = predictor(img)
        instances = predictions['instances']
        
        beziers = predictions['instances'].beziers.cpu().detach().numpy()
        recs = instances.recs
    
        content = ""
        for p, rec in zip(beziers, recs):
            p = [list(map(int,p[i:i + 2])) for i in range(0, len(p), 2)]
            points = [p[0]] + [p[3]] + [p[4]] + [p[7]]
            for _, _p in enumerate(points):
                content += ','.join(list(map(str,list(map(int,_p)))))
                if not (_ == len(points) - 1):
                    content += ','
            content += "\n"
            countBbox += 1
        
        if filename in trace:
            print("nope", filename)
            break
        else:
            trace[filename] = 1
            f_submission = open(args.output + filename + ".txt", 'w', encoding="utf-8")
            f_submission.write(content)
            f_submission.close()
            print("Done", filename, "-", str(id) + "/" + str(len(files)))
    end_time = time.time()
    
    print("============ FINISHED DETECTION (time elapsed: {}). TOTAL DETECTED BBOX: {} ============".format(str(countBbox), str(end_time - start_time)))
