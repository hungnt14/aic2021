from detectron2.engine.defaults import DefaultPredictor
from detectron2.data.detection_utils import read_image
from adet.config import get_cfg
from tqdm import tqdm_notebook
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

dictionary = "aàáạảãâầấậẩẫăằắặẳẵAÀÁẠẢÃĂẰẮẶẲẴÂẦẤẬẨẪeèéẹẻẽêềếệểễEÈÉẸẺẼÊỀẾỆỂỄoòóọỏõôồốộổỗơờớợởỡOÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠiìíịỉĩIÌÍỊỈĨuùúụủũưừứựửữƯỪỨỰỬỮUÙÚỤỦŨyỳýỵỷỹYỲÝỴỶỸ"
 
 
def make_groups():
    # dictionary = 'aàáạảãâầấậẩẫăằắặẳẵAÀÁẠẢÃĂẰẮẶẲẴÂẦẤẬẨẪeèéẹẻẽêềếệểễEÈÉẸẺẼÊỀẾỆỂỄoòóọỏõôồốộổỗơờớợởỡOÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠiìíịỉĩIÌÍỊỈĨuùúụủũưừứựửữƯỪỨỰỬỮUÙÚỤỦŨyỳýỵỷỹYỲÝỴỶỸ'
    groups = []
    i = 0
    while i < len(dictionary) - 5:
        group = [c for c in dictionary[i : i + 6]]
        i += 6
        groups.append(group)
    return groups
 

groups = make_groups()
 
TONES = ["", "ˋ", "ˊ", "﹒", "ˀ", "˜"]
SOURCES = ["ă", "â", "Ă", "Â", "ê", "Ê", "ô", "ơ", "Ô", "Ơ", "ư", "Ư", "Đ", "đ"]
TARGETS = ["aˇ", "aˆ", "Aˇ", "Aˆ", "eˆ", "Eˆ", "oˆ", "o˒", "Oˆ", "O˒", "u˒", "U˒", "D‑", "d‑"]

def correct_tone_position(word):
    word = word[:-1]
    if len(word) < 2:
        pass
    first_ord_char = ""
    second_order_char = ""
    for char in word:
        for group in groups:
            if char in group:
                second_order_char = first_ord_char
                first_ord_char = group[0]
    if word[-1] == first_ord_char and second_order_char != "":
        pair_chars = ["qu", "Qu", "qU", "QU", "gi", "Gi", "gI", "GI"]
        for pair in pair_chars:
            if pair in word and second_order_char in ["u", "U", "i", "I"]:
                return first_ord_char
        return second_order_char
    return first_ord_char
  
def decoder(recognition):
    for char in TARGETS:
        recognition = recognition.replace(char, SOURCES[TARGETS.index(char)])
    if len(recognition) < 1:
        return recognition
    if recognition[-1] in TONES:
        if len(recognition) < 2:
            return recognition
        replace_char = correct_tone_position(recognition)
        tone = recognition[-1]
        recognition = recognition[:-1]
        for group in groups:
            if replace_char in group:
                recognition = recognition.replace(replace_char, group[TONES.index(tone)])
    return recognition
 
def _decode_recognition(rec):
    #        CTLABELS = [' ','!','"','#','$','%','&','\'','(',')','*','+',',','-','.','/','0','1','2','3','4','5','6','7','8','9',':',';','<','=','>','?','@','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','[','\\',']','^','_','`','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','{','|','}','~']
    # CTLABELS = ['^', '\\', '}', 'ỵ', '>', '<', '{', '~', '`', '°', '$', 'ẽ', 'ỷ', 'ẳ', '_', 'ỡ', ';', '=', 'Ẳ', 'j', '[', ']', 'ẵ', '?', 'ẫ', 'Ẵ', 'ỳ', 'Ỡ', 'ẹ', 'è', 'z', 'ỹ', 'ằ', 'õ', 'ũ', 'Ẽ', 'ỗ', 'ỏ', '@', 'Ằ', 'Ỳ', 'Ẫ', 'ù', 'ử', '#', 'Ẹ', 'Z', 'Õ', 'ĩ', 'Ỏ', 'È', 'Ỷ', 'ý', 'Ũ', '*', 'ò', 'é', 'q', 'ở', 'ổ', 'ủ', 'ẩ', 'ã', 'ẻ', 'J', 'ữ', 'ễ', 'ặ', '+', 'ứ', 'Ỹ', 'ự', 'ụ', 'Ỗ', '%', 'ắ', 'ồ', '"', 'ề', 'ể', 'ỉ', 'ợ', '!', 'Ẻ', 'ừ', 'ọ', '&', 'ì', 'É', 'ậ', 'Ù', 'Ặ', 'x', 'Ỉ', 'ú', 'í', 'ó', 'Ẩ', 'ị', 'ế', 'Ứ', 'â', 'ấ', 'ầ', 'ớ', 'ă', 'Ủ', 'Ĩ', '(', 'Ắ', 'Ừ', ')', 'ờ', 'Ý', 'Ễ', 'Ã', 'ô', 'ộ', 'Ữ', 'Ợ', 'ả', 'Ở', 'ệ', 'W', 'ơ', 'Ổ', 'ố', 'Ề', 'f', 'Ử', 'ạ', 'w', 'Ò', 'Ự', 'Ụ', 'Ú', 'Ồ', 'ê', 'Ó', 'Ì', 'b', 'Í', 'Ể', 'đ', 'Ớ', '/', 'k', 'Ă', 'v', 'Ị', 'Ậ', 'Ọ', 'd', 'Ầ', 'Ấ', 'ư', 'á', 'Ế', ' ', 'p', 'Ơ', 'F', 'Ả', 'Ộ', 'Ê', 'Ờ', 's', '-', 'à', 'y', 'Ố', 'l', 'Â', 'Q', ',', 'X', 'Ệ', 'Ạ', 'Ô', 'r', ':', '6', '7', 'u', '4', 'm', '5', 'e', '8', 'c', 'Ư', 'Á', '9', 'D', '3', 'o', '.', 'Y', 'g', 'K', 'a', 'À', 't', '2', 'B', 'E', 'V', 'R', '1', 'S', 'i', 'L', 'P', 'Đ', 'h', 'U', '0', 'M', 'O', 'n', 'A', 'G', 'I', 'C', 'T', 'H', 'N']
    CTLABELS = [' ','!','"','#','$','%','&',"\'",'(',')','*','+',',','-','.','/','0','1','2','3','4','5','6','7','8','9',':',';','<','=','>','?','@','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','[','\\',']','^','_','`','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','{','|','}','~','ˋ','ˊ','﹒','ˀ','˜','ˇ','ˆ','˒','‑']
    s = ""
    for c in rec:
        c = int(c)
        if c < 104:
            s += CTLABELS[c]
        elif c == 104:
            s += u"口"
    return decoder(s)
 
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
    
    trace = {}
    cnt = 0
    for img_path in files:
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
            text = _decode_recognition(rec)
            text = ""
            for _, _p in enumerate(points):
                content += ','.join(list(map(str,list(map(int,_p)))))
                if not (_ == len(points) - 1):
                    content += ','
            content += "\n"
        
        if filename in trace:
            print("nope", filename)
            break
        else:
            trace[filename] = 1
            f_submission = open(args.output + filename + ".txt", 'w', encoding="utf-8")
            f_submission.write(content)
            f_submission.close()
            cnt += 1
            print("Done", filename, " - " + str(cnt) + "/" + str(len(files)))
    
