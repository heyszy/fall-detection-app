import os
import cv2
import time
import torch
import argparse
import numpy as np
import subprocess
import requests

from Detection.Utils import ResizePadding
from CameraLoader import CamLoader, CamLoaderForVideo
from DetectorLoader import TinyYOLOv3_onecls

from PoseEstimateLoader import SPPE_FastPose
from fn import draw_single

from Track.Tracker import Detection, Tracker
from ActionsEstLoader import TSSTG

from threading import Timer

source = './test.mp4'
# source = 'rtsp://192.168.31.58:8554/live1.h264'
# source = 'rtmp://localhost/live/JETSON_NANO'
# source = 'rtmp://localhost/live/test'
# source = 0
out = 'result.mp4'

# rtmp = r'rtmp://localhost/live/FD'

# https://wxpusher.zjiecode.com/ WxPusher 提供的微信消息推送服务
url = 'http://wxpusher.zjiecode.com/api/send/message/'
params = {
    'appToken': 'AT_Y3GB4WcwHP2NNoBT3pSqOK337ctxWlCP',
    'content': '检测到摔倒行为发生',
    'uid': 'UID_ABaGAlqE4qJnJ8f0qnATlTHUyz23',
}


def preproc(img):
    """preprocess function for CameraLoader.
    """
    img = resize_fn(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def kpt2bbox(kpt, ex=20):
    """Get bbox that hold on all of the keypoints (x,y)
    kpt: array of shape `(N, 2)`,
    ex: (int) expand bounding box,
    """
    return np.array((kpt[:, 0].min() - ex, kpt[:, 1].min() - ex,
                     kpt[:, 0].max() + ex, kpt[:, 1].max() + ex))


if __name__ == '__main__':
    par = argparse.ArgumentParser(description='Human Fall Detection Demo.')
    par.add_argument('-C', '--camera', default=source,  # required=True,  # default=2,
                     help='Source of camera or video file path.')
    par.add_argument('--detection_input_size', type=int, default=384,
                     help='Size of input in detection model in square must be divisible by 32 (int).')
    par.add_argument('--pose_input_size', type=str, default='224x160',
                     help='Size of input in pose model must be divisible by 32 (h, w)')
    par.add_argument('--pose_backbone', type=str, default='resnet50',
                     help='Backbone model for SPPE FastPose model.')
    par.add_argument('--show_detected', default=False, action='store_true',
                     help='Show all bounding box from detection.')
    par.add_argument('--show_skeleton', default=True, action='store_true',
                     help='Show skeleton pose.')
    par.add_argument('--save_out', type=str, default=out,
                     help='Save display to video file.')
    par.add_argument('--device', type=str, default='cuda',
                     help='Device to run model on cpu or cuda.')
    par.add_argument('--stream_out', type=str, default='',
                     help='推流输出地址')
    args = par.parse_args()

    device = args.device

    # DETECTION MODEL.
    inp_dets = args.detection_input_size
    detect_model = TinyYOLOv3_onecls(inp_dets, device=device)

    # POSE MODEL.
    inp_pose = args.pose_input_size.split('x')
    inp_pose = (int(inp_pose[0]), int(inp_pose[1]))
    pose_model = SPPE_FastPose(args.pose_backbone, inp_pose[0], inp_pose[1], device=device)

    # Tracker.
    max_age = 30
    tracker = Tracker(max_age=max_age, n_init=3)

    # Actions Estimate.
    action_model = TSSTG()

    resize_fn = ResizePadding(inp_dets, inp_dets)

    cam_source = args.camera
    if type(cam_source) is str and os.path.isfile(cam_source):
        # Use loader thread with Q for video file.
        cam = CamLoaderForVideo(cam_source, queue_size=3000, preprocess=preproc).start()
    else:
        # Use normal thread loader for webcam.
        cam = CamLoader(int(cam_source) if cam_source.isdigit() else cam_source, preprocess=preproc).start()

    # frame_size = cam.frame_size
    # scf = torch.min(inp_size / torch.FloatTensor([frame_size]), 1)[0]

    is_save_out = False
    if not cam_source.startswith('rtmp') and args.save_out != '':
        is_save_out = True
        codec = cv2.VideoWriter_fourcc(*'avc1')
        writer = cv2.VideoWriter(args.save_out, codec, 30, (inp_dets * 2, inp_dets * 2))

    fps_time = 0
    f = 0

    is_stream_out = False
    if args.stream_out != '':
        is_stream_out = True
        sizeStr = str(inp_dets * 2) + 'x' + str(inp_dets * 2)
        command = ['ffmpeg',
                   '-y', '-an',
                   '-f', 'rawvideo',
                   '-vcodec', 'rawvideo',
                   '-pix_fmt', 'bgr24',
                   '-s', sizeStr,
                   '-r', '15',
                   '-i', '-',
                   '-c:v', 'libx264',
                   '-pix_fmt', 'yuv420p',
                   '-preset', 'ultrafast',
                   '-f', 'flv',
                   args.stream_out]
        pipe = subprocess.Popen(command, shell=False, stdin=subprocess.PIPE)

    can_alert = True

    fall_detect_count = 0


    def set_can_alert():
        global can_alert
        can_alert = True


    while cam.grabbed():
        f += 1
        frame = cam.getitem()
        # image = frame.copy()

        # Detect humans bbox in the frame with detector model.
        detected = detect_model.detect(frame, need_resize=False, expand_bb=10)

        # Predict each tracks bbox of current frame from previous frames information with Kalman filter.
        tracker.predict()
        # Merge two source of predicted bbox together.
        for track in tracker.tracks:
            det = torch.tensor([track.to_tlbr().tolist() + [0.5, 1.0, 0.0]], dtype=torch.float32)
            detected = torch.cat([detected, det], dim=0) if detected is not None else det

        detections = []  # List of Detections object for tracking.
        if detected is not None:
            # detected = non_max_suppression(detected[None, :], 0.45, 0.2)[0]
            # Predict skeleton pose of each bboxs.
            poses = pose_model.predict(frame, detected[:, 0:4], detected[:, 4])

            # Create Detections object.
            detections = [Detection(kpt2bbox(ps['keypoints'].numpy()),
                                    np.concatenate((ps['keypoints'].numpy(),
                                                    ps['kp_score'].numpy()), axis=1),
                                    ps['kp_score'].mean().numpy()) for ps in poses]

            # VISUALIZE.
            if args.show_detected:
                for bb in detected[:, 0:5]:
                    frame = cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), (0, 0, 255), 1)

        # Update tracks by matching each track information of current and previous frame or
        # create a new track if no matched.
        tracker.update(detections)

        # Predict Actions of each track.
        for i, track in enumerate(tracker.tracks):
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            bbox = track.to_tlbr().astype(int)
            center = track.get_center().astype(int)

            action = 'pending..'
            clr = (0, 255, 0)
            # Use 30 frames time-steps to prediction.
            if len(track.keypoints_list) == 30:
                pts = np.array(track.keypoints_list, dtype=np.float32)
                out = action_model.predict(pts, frame.shape[:2])
                action_name = action_model.class_names[out[0].argmax()]
                action = '{}: {:.2f}%'.format(action_name, out[0].max() * 100)
                if action_name == 'Fall Down':
                    # print("Fall Detected!")
                    clr = (255, 0, 0)
                    if can_alert:
                        fall_detect_count += 1
                    if type(cam_source) is str and cam_source.startswith(
                            'rtmp') and fall_detect_count == 5 and can_alert:
                        requests.get(url, params)
                        can_alert = False
                        fall_detect_count = 0
                        t = Timer(30.0, set_can_alert)
                        t.start()
                elif action_name == 'Lying Down':
                    clr = (255, 200, 0)

            # VISUALIZE.
            if track.time_since_update == 0:
                if args.show_skeleton:
                    frame = draw_single(frame, track.keypoints_list[-1])
                frame = cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 1)
                frame = cv2.putText(frame, str(track_id), (center[0], center[1]), cv2.FONT_HERSHEY_COMPLEX,
                                    0.4, (255, 0, 0), 2)
                frame = cv2.putText(frame, action, (bbox[0] + 5, bbox[1] + 15), cv2.FONT_HERSHEY_COMPLEX,
                                    0.4, clr, 1)

        # Show Frame.
        frame = cv2.resize(frame, (0, 0), fx=2., fy=2.)
        frame = cv2.putText(frame, '%d, FPS: %f' % (f, 1.0 / (time.time() - fps_time)),
                            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        frame = frame[:, :, ::-1]
        fps_time = time.time()

        if is_save_out:
            writer.write(frame)

        cv2.imshow('frame', frame)

        if is_stream_out:
            pipe.stdin.write(frame.tostring())

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clear resource.
    cam.stop()

    if is_save_out:
        writer.release()
    cv2.destroyAllWindows()
    if is_stream_out:
        pipe.terminate()
