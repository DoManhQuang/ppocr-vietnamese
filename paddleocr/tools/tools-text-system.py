from tools.infer.predict_system import TextSystem
import tools.infer.utility as utility
from tools.engine import text_sys_infer

det_model_dir="/work/quang.domanh/ppocr/PaddleOCR-2.8.0/inference/dict/ch_PP-OCRv4_det_infer.onnx"
rec_model_dir="/work/quang.domanh/ppocr/PaddleOCR-2.8.0/inference/dict/ch_PP-OCRv4_rec_infer.onnx"
cls_model_dir="/work/quang.domanh/ppocr/PaddleOCR-2.8.0/inference/dict/ch_ppocr_mobile_v2.0_cls_infer.onnx"
image_dir="/work/quang.domanh/datasets/XFUND/zh_val/image/zh_val_2.jpg"
rec_char_dict_path="./ppocr/utils/ppocr_keys_v1.txt"

# results = infer_ocr(rec_model_dir=rec_model_dir, det_model_dir=det_model_dir, cls_model_dir=cls_model_dir, rec_char_dict_path=rec_char_dict_path,
#                     image_dir=image_dir, use_gpu=False, use_onnx=True)

args = utility.parse_args()
args.det_model_dir = det_model_dir
args.rec_model_dir = rec_model_dir
args.cls_model_dir = cls_model_dir
args.image_dir = image_dir
args.use_gpu = False
args.use_onnx = True
textsys = TextSystem(args)

results = text_sys_infer(args=args, text_sys=textsys, image_dir=image_dir)

print("results >>>>>>>>> ", results)