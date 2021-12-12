cd /aic2021/abcnetv2/tools/
python predict.py --input="/data/test_data/" --output="/aic2021/output/abcnetv2/" --confidence_threshold=0.4 --weights="/aic2021/weights/abcnetv2/model_final.pth"
cd /aic2021/vietocr/
python crop_polygon_picture.py --detected_bboxes="/aic2021/output/abcnetv2/" --original_images="/data/test_data/" --output_cropped="/aic2021/output/vietocr/cropped/"
python recognition_module.py --cropped_bboxes="/aic2021/output/vietocr/cropped/" --original_detect="/aic2021/output/abcnetv2/" --original_images="/data/test_data/" --seperated="/aic2021/output/vietocr/seperated/" --seperated_null="/aic2021/output/vietocr/seperated_null/" --weights="/aic2021/weights/vietocr/transformerocr.pth" --drop_score=0.8
cd /aic2021/PaddleOCR/tools/infer/
python3 test1.py --use_gpu=True --rec_algorithm="SRN" --rec_model_dir="/aic2021/weights/SRN/" --image_dir="/data/test_data/" --input="/aic2021/output/vietocr/seperated_null/" --output="/aic2021/output/paddleocr/" --drop_score=0.93 --rec_image_shape="1, 64, 256" --rec_char_type="ch" --rec_char_dict_path="/aic2021/PaddleOCR/ppocr/utils/dict/vi_vietnam.txt"
cd /aic2021/tools/
python mergeResults.py --input1="/aic2021/output/vietocr/seperated/" --input2="/aic2021/output/paddleocr/" --output="/data/submission_output/"
