import argparse
from models import *
from bounding_box import bounding_box as bb
from utils.datasets import *
from utils.utils import *
import random
import pickle

def detect(save_img=False):
    imgsz = opt.img_size
    out, source, weights, half, view_img, save_txt = opt.output, opt.source, opt.weights, opt.half, opt.view_img, opt.save_txt
    # Initialize
    device = torch_utils.select_device(opt.device)
    # if os.path.exists(out):
        # shutil.rmtree(out)  # delete output folder
    # os.makedirs(out)  # make new output folder
    
    # Initialize model
    model = Darknet(opt.cfg, imgsz)

    # Load weights
    model.load_state_dict(torch.load(weights, map_location=device)['model'])
    # Eval mode
    model.to(device).eval()
    # load image
    save_img = True
    time_list = []
    with open(source, 'r') as f :
        for test_file in f :
            test_file = test_file.replace('\n', '')

            dataset = LoadImages(test_file, img_size=imgsz)
            
            # Get names and colors
            names = load_classes(opt.names)

            # Run inference
            t0 = time.time()
            img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
            _ = model(img.half() if half else img.float()) if device.type != 'cpu' else None  # run once
            for path, img, im0s, vid_cap in dataset:
                img = torch.from_numpy(img).to(device)
                img = img.half() if half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)

                # Inference
                t1 = torch_utils.time_synchronized()
                pred = model(img, augment=opt.augment)[0]
                t2 = torch_utils.time_synchronized()

                # to float
                if half:
                    pred = pred.float()

                # Apply NMS
                pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres,
                                        multi_label=False, classes=opt.classes, agnostic=opt.agnostic_nms)
                    
                # Process detections
                for i, det in enumerate(pred):  # detections for image i
                    p, s, im0 = path, '', im0s

                    # save_path = str(Path(out) / Path(p).name)
                    s += '%gx%g ' % img.shape[2:]  # print string
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
                    if det is not None and len(det):
                        # Rescale boxes from imgsz to im0 size
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                        # Print results
                        for c in det[:, -1].unique():
                            n = (det[:, -1] == c).sum()  # detections per class
                            s += '%g %ss, ' % (n, names[int(c)])  # add to string

                        # Write results
                        for *xyxy, conf, cls in reversed(det):
                            if save_txt:  # Write to file
                                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                                with open(save_path[:save_path.rfind('.')] + '.txt', 'a') as file:
                                    file.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                            if save_img or view_img:  # Add bbox to image
                                # label = '%s %.2f' % (names[int(cls)], conf)
                                label = names[int(cls)]

                                c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                                bb.add(im0, c1[0], c1[1], c2[0], c2[1], label)
                                print(c1[0], c1[1], c2[0], c2[1], label)

                    # Print time (inference + NMS)
                    save_path = './outputs_2/' + 'test_' + test_file.split('/')[-1]
                    time_list.append([test_file, t2 - t1])
                    cv2.imwrite(save_path, im0)
                    print('%sDone. (%.3fs)' % (s, t2 - t1))

    with open('time_list.txt', 'wb') as f :
        pickle.dump(time_list, f)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp-custom.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='data/class.names', help='*.names path')
    parser.add_argument('--weights', type=str, default='weights/best_cls62_test.pt', help='weights path')
    parser.add_argument('--source', type=str, default='data/cls62-test.txt', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=512, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    opt.names = check_file(opt.names)  # check file
    print(opt)


    with torch.no_grad():
        detect()

