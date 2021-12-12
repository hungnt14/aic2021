cd /aic2021/abcnetv2/tools/
python predict.py --input="/data/test_data/" --output="/aic2021/output/abcnetv2/" --confidence_threshold=0.4 --weights="/aic2021/weights/abcnetv2/model_final.pth"
cd /aic2021/vietocr/
python crop_polygon_picture.py --detected_bboxes="/aic2021/output/abcnetv2/" --original_images="/data/test_data/" --output_cropped="/aic2021/output/vietocr/cropped/"
python recognition_module.py --cropped_bboxes="/aic2021/output/vietocr/cropped/" --original_detect="/aic2021/output/abcnetv2/" --original_images="/data/test_data/" --seperated="/aic2021/output/vietocr/seperated/" --seperated_null="/aic2021/output/vietocr/seperated_null/" --weights="/aic2021/weights/vietocr/transformerocr.pth" --drop_score=0.8