import os
import os.path as osp
import argparse
import numpy as np
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch
import cv2
import datetime
from tqdm import tqdm
from pathlib import Path
from human_models.human_models import SMPLX
from ultralytics import YOLO
from main.base import Tester
from main.config import Config
from utils.data_utils import load_img, process_bbox, generate_patch_image
from utils.visualization_utils import render_mesh
from utils.inference_utils import non_max_suppression


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_gpus', type=int, dest='num_gpus')
    parser.add_argument('--file_name', type=str, default='test')
    parser.add_argument('--ckpt_name', type=str, default='model_dump')
    parser.add_argument('--start', type=str, default=1)
    parser.add_argument('--end', type=str, default=1)
    parser.add_argument('--multi_person', action='store_true')
    parser.add_argument('--calibration_npz', type=str, default=None,
                        help='Path to camera calibration .npz with keys K, dist, image_size. '
                             'Required for ArUco marker 3D estimation.')
    parser.add_argument('--assume_undistorted', action='store_true',
                        help='Use K from calibration but set dist=zeros (frames already undistorted).')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    cudnn.benchmark = True

    # init config
    time_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    root_dir = Path(__file__).resolve().parent.parent
    config_path = osp.join('./pretrained_models', args.ckpt_name, 'config_base.py')
    cfg = Config.load_config(config_path)
    checkpoint_path = osp.join('./pretrained_models', args.ckpt_name, f'{args.ckpt_name}.pth.tar')
    img_folder = osp.join(root_dir, 'demo', 'input_frames', args.file_name)
    output_folder = osp.join(root_dir, 'demo', 'output_frames', args.file_name)
    smplx_dir = osp.join(root_dir, 'demo', 'results', args.file_name, 'smplx')
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(smplx_dir, exist_ok=True)
    exp_name = f'inference_{args.file_name}_{args.ckpt_name}_{time_str}'

    new_config = {
        "model": {
            "pretrained_model_path": checkpoint_path,
        },
        "log":{
            'exp_name':  exp_name,
            'log_dir': osp.join(root_dir, 'outputs', exp_name, 'log'),  
            }
    }
    cfg.update_config(new_config)
    cfg.prepare_log()
    
    # init human models
    smpl_x = SMPLX(cfg.model.human_model_path)

    # init tester
    demoer = Tester(cfg)
    demoer.logger.info(f"Using 1 GPU.")
    demoer.logger.info(f'Inference [{args.file_name}] with [{cfg.model.pretrained_model_path}].')
    demoer._make_model()

    # init detector
    bbox_model = getattr(cfg.inference.detection, "model_path",
                        './pretrained_models/yolov8x.pt')
    detector = YOLO(bbox_model)

    # load camera calibration (physical camera intrinsics for ArUco 3D estimation)
    if args.calibration_npz:
        _calib = np.load(args.calibration_npz)
        camera_K    = _calib['K'].astype(np.float64)
        camera_dist = np.zeros(5, dtype=np.float64) if args.assume_undistorted \
                      else _calib['dist'].astype(np.float64)
        demoer.logger.info(f"Loaded camera calibration from {args.calibration_npz}")
    else:
        camera_K    = np.zeros((3, 3), dtype=np.float64)
        camera_dist = np.zeros(5, dtype=np.float64)

    # init ArUco detector (opencv-contrib-python >= 4.7 required)
    _aruco_dict   = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    _aruco_params = cv2.aruco.DetectorParameters()
    aruco_detector = cv2.aruco.ArucoDetector(_aruco_dict, _aruco_params)

    start = int(args.start)
    end = int(args.end) + 1

    for frame in tqdm(range(start, end)):
        
        # prepare input image
        img_path =osp.join(img_folder, f'{int(frame):06d}.jpg')

        transform = transforms.ToTensor()
        original_img = load_img(img_path)
        vis_img = original_img.copy()
        original_img_height, original_img_width = original_img.shape[:2]
        
        # ArUco detection — runs once per frame; saved to every per-person .npz for this frame
        _gray = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
        _corners_raw, _ids_raw, _ = aruco_detector.detectMarkers(_gray)
        if _ids_raw is not None:
            marker_ids    = _ids_raw.flatten().astype(np.int32)
            marker_corners = np.stack([c.squeeze(0) for c in _corners_raw])  # (N, 4, 2)
        else:
            marker_ids    = np.zeros((0,), dtype=np.int32)
            marker_corners = np.zeros((0, 4, 2), dtype=np.float32)

        # detection, xyxy
        yolo_bbox = detector.predict(original_img, 
                                device='cuda', 
                                classes=00, 
                                conf=cfg.inference.detection.conf, 
                                save=cfg.inference.detection.save, 
                                verbose=cfg.inference.detection.verbose
                                    )[0].boxes.xyxy.detach().cpu().numpy()

        if len(yolo_bbox)<1:
            # save original image if no bbox
            num_bbox = 0
        elif not args.multi_person:
            # only select the largest bbox
            num_bbox = 1
            # yolo_bbox = yolo_bbox[0]
        else:
            # keep bbox by NMS with iou_thr
            yolo_bbox = non_max_suppression(yolo_bbox, cfg.inference.detection.iou_thr)
            num_bbox = len(yolo_bbox)

        # loop all detected bboxes
        for bbox_id in range(num_bbox):
            yolo_bbox_xywh = np.zeros((4))
            yolo_bbox_xywh[0] = yolo_bbox[bbox_id][0]
            yolo_bbox_xywh[1] = yolo_bbox[bbox_id][1]
            yolo_bbox_xywh[2] = abs(yolo_bbox[bbox_id][2] - yolo_bbox[bbox_id][0])
            yolo_bbox_xywh[3] = abs(yolo_bbox[bbox_id][3] - yolo_bbox[bbox_id][1])
            
            # xywh
            bbox = process_bbox(bbox=yolo_bbox_xywh, 
                                img_width=original_img_width, 
                                img_height=original_img_height, 
                                input_img_shape=cfg.model.input_img_shape, 
                                ratio=getattr(cfg.data, "bbox_ratio", 1.25))                
            img, _, _ = generate_patch_image(cvimg=original_img, 
                                                bbox=bbox, 
                                                scale=1.0, 
                                                rot=0.0, 
                                                do_flip=False, 
                                                out_shape=cfg.model.input_img_shape)
                
            img = transform(img.astype(np.float32))/255
            img = img.cuda()[None,:,:,:]
            inputs = {'img': img}
            targets = {}
            meta_info = {}

            # mesh recovery
            with torch.no_grad():
                out = demoer.model(inputs, targets, meta_info, 'test')

            mesh = out['smplx_mesh_cam'].detach().cpu().numpy()[0]

            # rendered (virtual) camera intrinsics, matched to the bbox crop
            focal = [cfg.model.focal[0] / cfg.model.input_body_shape[1] * bbox[2],
                     cfg.model.focal[1] / cfg.model.input_body_shape[0] * bbox[3]]
            princpt = [cfg.model.princpt[0] / cfg.model.input_body_shape[1] * bbox[2] + bbox[0],
                       cfg.model.princpt[1] / cfg.model.input_body_shape[0] * bbox[3] + bbox[1]]

            # save shape parameters for downstream measurement
            # existing keys (betas, cam_trans, vertices) are unchanged — backward compatible
            np.savez(
                osp.join(smplx_dir, f'{int(frame):06d}_{bbox_id}.npz'),
                # --- existing (backward compatible) ---
                betas=out['smplx_shape'].detach().cpu().numpy()[0],
                cam_trans=out['cam_trans'].detach().cpu().numpy()[0],
                vertices=mesh,
                # --- pose params (needed for Path B optimizer initialization) ---
                body_pose=out['smplx_body_pose'].detach().cpu().numpy()[0],
                global_orient=out['smplx_root_pose'].detach().cpu().numpy()[0],
                # --- rendered (virtual) camera, per-bbox ---
                bbox=bbox,
                bbox_xyxy=yolo_bbox[bbox_id],
                focal=np.array(focal),
                princpt=np.array(princpt),
                # --- physical camera calibration (metric; zeros if not provided) ---
                camera_K=camera_K,
                camera_dist=camera_dist,
                # --- ArUco markers detected in this frame ---
                markers_ids=marker_ids,
                markers_corners=marker_corners,
                # --- provenance ---
                img_shape=np.array(original_img.shape[:2]),
            )
            
            # draw the bbox on img
            vis_img = cv2.rectangle(vis_img, (int(yolo_bbox[bbox_id][0]), int(yolo_bbox[bbox_id][1])), 
                                    (int(yolo_bbox[bbox_id][2]), int(yolo_bbox[bbox_id][3])), (0, 255, 0), 1)
            # draw mesh
            vis_img = render_mesh(vis_img, mesh, smpl_x.face, {'focal': focal, 'princpt': princpt}, mesh_as_vertices=False)

        # save rendered image
        frame_name = os.path.basename(img_path)
        cv2.imwrite(os.path.join(output_folder, frame_name), vis_img[:, :, ::-1])


if __name__ == "__main__":
    main()
