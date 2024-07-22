import os, time, cv2
import numpy as np
from ppocr.utils.utility import get_image_file_list, check_and_read
from ppocr.utils.logging import get_logger


logger = get_logger()

def text_sys_infer(args, text_sys, image_dir):
    image_file_list = get_image_file_list(image_dir)
    image_file_list = image_file_list[args.process_id :: args.total_process_num]
    draw_img_save_dir = args.draw_img_save_dir
    os.makedirs(draw_img_save_dir, exist_ok=True)
    save_results = []
    logger.info(
        "In PP-OCRv3, rec_image_shape parameter defaults to '3, 48, 320', "
        "if you are using recognition model with PP-OCRv2 or an older version, please set --rec_image_shape='3,32,320"
    )

    # warm up 10 times
    if args.warmup:
        img = np.random.uniform(0, 255, [640, 640, 3]).astype(np.uint8)
        for _ in range(10):
            res = text_sys(img)

    total_time = 0
    _st = time.time()
    for idx, image_file in enumerate(image_file_list):
        img, flag_gif, flag_pdf = check_and_read(image_file)
        if not flag_gif and not flag_pdf:
            img = cv2.imread(image_file)
        if not flag_pdf:
            if img is None:
                logger.debug("error in loading image:{}".format(image_file))
                continue
            imgs = [img]
        else:
            page_num = args.page_num
            if page_num > len(img) or page_num == 0:
                page_num = len(img)
            imgs = img[:page_num]
        
        print("length >>>>>>>>> ", len(imgs), np.array(imgs).shape)
        for _, img in enumerate(imgs):
            starttime = time.time()
            dt_boxes, rec_res, _ = text_sys(img)
            elapse = time.time() - starttime
            total_time += elapse
            res = [
                {
                    "transcription": rec_res[i][0],
                    "points": np.array(dt_boxes[i]).astype(np.int32).tolist(),
                }
                for i in range(len(dt_boxes))
            ]
            save_results.append(res)   
    
    logger.info("The predict total time is {}".format(time.time() - _st))
    if args.benchmark:
        text_sys.text_detector.autolog.report()
        text_sys.text_recognizer.autolog.report()

    return save_results