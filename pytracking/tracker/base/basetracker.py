import matplotlib
from pytracking.utils.config import ON_COLAB

matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2 as cv
import time
import os
import numpy as np

from google.colab.patches import cv2_imshow
from mtcnn import MTCNN

class BaseTracker:
    """Base class for all trackers."""

    def __init__(self, params):
        self.params = params

    def initialize(self, image, state, class_info=None):
        """Overload this function in your tracker. This should initialize the model."""
        raise NotImplementedError

    def track(self, image):
        """Overload this function in your tracker. This should track in the frame and update the model."""
        raise NotImplementedError

    def track_sequence(self, sequence):
        """Run tracker on a sequence."""

        # Initialize
        image = self._read_image(sequence.frames[0])

        times = []
        start_time = time.time()
        self.sequence_name = sequence.name
        self.initialize(image, sequence.init_state)
        init_time = getattr(self, 'time', time.time() - start_time)
        times.append(init_time)

        if self.params.visualization:
            self.init_visualization()
            self.visualize(image, sequence.init_state)

        # Track
        tracked_bb = [sequence.init_state]
        for frame in sequence.frames[1:]:
            image = self._read_image(frame)

            start_time = time.time()
            state = self.track(image)
            times.append(time.time() - start_time)

            tracked_bb.append(state)

            if self.params.visualization:
                self.visualize(image, state)

        return tracked_bb, times

    def track_sequence_w_face_recog(self, sequence, facerecog):
        """Run tracker on a sequence."""

        times = []
        tracked_bb = []
        self.sequence_name = sequence.name

        face_detect, face_recog = facerecog[0], facerecog[1]

        if self.params.visualization:
            self.init_visualization()
        
        face_found = False
        face_id = None

        for frame in sequence.frames:
            image = self._read_image(frame)

            start_time = time.time()

            # Face detect
            if not face_found:
                faces = face_detect.detect_faces(image)
                state = [0,0,0,0]
                if faces:
                    face_found = True

                    #identify faces
                    names = face_recog.face_identification(image,faces)

                    face_id, state = names[0], faces[0]['box']

                    # Enumerate all names, track the 1st unknown identity
                    for i, name in enumerate(names):
                        if name == "Unknown":
                            face_id,state = names[i], faces[i]['box']
                            break
                    
                    # Initialize tracking features
                    self.initialize(image, state)
            
            # Face track
            else :
                state = self.track(image)

                if state == None:
                    state = [0,0,0,0]
                    face_found = False
                    face_id = None

                if self.params.visualization:
                    self.visualize(image, state)
            
            state.append(face_id)
            
            times.append(time.time() - start_time)
            tracked_bb.append(state)

        return tracked_bb, times

    def live_track(self, videopath):
        
        # Start live capture
        cap = cv.VideoCapture(videopath)
        sign, frame = cap.read()
        frame_cp = frame.copy()

        # init detector
        detector = MTCNN()

        if sign is not True :
            print("Read frame from {} failed.".format(videopath))
            exit(-1)
        
        track_bb = []
        track_time = []
        while sign :
            # start detection
            start_time = time.time()
            bboxes = detector.detect_faces(frame_cp)
            end_time = time.time() - start_time

            if (bboxes == [] or bboxes is None):
                print("no face detected..")
                # read next frame
                sign, frame = cap.read()
                frame_cp = frame.copy()
                cv.waitKey(1)
                continue

            # process if face detected
            bbox = bboxes[0]['box']
            track_time.append(end_time)
            track_bb.append(bbox)

            # initialize tracker
            self.initialize(frame_cp, bbox)

            sign, frame = cap.read()
            frame_cp = frame.copy()

            # While the object is still detected
            while ((bbox is not None) and sign) :
                # Track object
                start_time = time.time()
                bbox = self.track(frame_cp)
                track_time.append(time.time() - start_time)

                track_bb.append(bbox)
                
                # read next frame
                sign, frame = cap.read()
                cv.waitKey(1)
        
        cap.release()

        return track_bb, track_time

    def imshow(self, display_name, frame, **kwargs):
        # if ON_COLAB:
        cv2_imshow(frame, **kwargs)
        # else:
        #     cv.imshow(display_name, frame, **kwargs)
    
    def track_videofile(self, videofilepath, optional_box=None):
        """Run track with a video file input."""

        assert os.path.isfile(videofilepath), "Invalid param {}".format(videofilepath)
        ", videofilepath must be a valid videofile"

        if hasattr(self, 'initialize_features'):
            self.initialize_features()

        cap = cv.VideoCapture(videofilepath)
        display_name = 'Display: ' + self.params.tracker_name
        # if not ON_COLAB:
        #     cv.namedWindow(display_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
        #     cv.resizeWindow(display_name, 960, 720)
        success, frame = cap.read()
        self.imshow(display_name, frame)
        if success is not True:
            print("Read frame from {} failed.".format(videofilepath))
            exit(-1)
        if optional_box is not None:
            assert isinstance(optional_box, (list, tuple))
            assert len(optional_box) == 4, "valid box's foramt is [x,y,w,h]"
            self.initialize(frame, optional_box)
        else:
            while True:
                # cv.waitKey()
                frame_disp = frame.copy()

                cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL,
                           1.5, (0, 0, 0), 1)

                x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
                init_state = [x, y, w, h]
                self.initialize(frame, init_state)
                break

        while True:
            ret, frame = cap.read()

            if frame is None:
                return

            frame_disp = frame.copy()

            # Draw box
            state = self.track(frame)
            state = [int(s) for s in state]
            cv.rectangle(frame_disp, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
                         (0, 255, 0), 5)

            font_color = (0, 0, 0)
            cv.putText(frame_disp, 'Tracking!', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)
            cv.putText(frame_disp, 'Press r to reset', (20, 55), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)
            cv.putText(frame_disp, 'Press q to quit', (20, 80), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)

            # Display the resulting frame
            self.imshow(display_name, frame_disp)
            key = cv.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('r'):
                ret, frame = cap.read()
                frame_disp = frame.copy()

                cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                           (0, 0, 0), 1)

                cv.imshow(display_name, frame_disp)
                x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
                init_state = [x, y, w, h]
                self.initialize(frame, init_state)

        # When everything done, release the capture
        cap.release()
        cv.destroyAllWindows()

    def track_webcam(self):
        """Run tracker with webcam."""

        class UIControl:
            def __init__(self):
                self.mode = 'init'  # init, select, track
                self.target_tl = (-1, -1)
                self.target_br = (-1, -1)
                self.mode_switch = False

            def mouse_callback(self, event, x, y, flags, param):
                if event == cv.EVENT_LBUTTONDOWN and self.mode == 'init':
                    self.target_tl = (x, y)
                    self.target_br = (x, y)
                    self.mode = 'select'
                    self.mode_switch = True
                elif event == cv.EVENT_MOUSEMOVE and self.mode == 'select':
                    self.target_br = (x, y)
                elif event == cv.EVENT_LBUTTONDOWN and self.mode == 'select':
                    self.target_br = (x, y)
                    self.mode = 'track'
                    self.mode_switch = True

            def get_tl(self):
                return self.target_tl if self.target_tl[0] < self.target_br[0] else self.target_br

            def get_br(self):
                return self.target_br if self.target_tl[0] < self.target_br[0] else self.target_tl

            def get_bb(self):
                tl = self.get_tl()
                br = self.get_br()

                bb = [min(tl[0], br[0]), min(tl[1], br[1]), abs(br[0] - tl[0]), abs(br[1] - tl[1])]
                return bb

        ui_control = UIControl()
        cap = cv.VideoCapture(0)
        display_name = 'Display: ' + self.params.tracker_name
        # if not ON_COLAB:
        #     cv.namedWindow(display_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
        #     cv.resizeWindow(display_name, 960, 720)
        #     cv.setMouseCallback(display_name, ui_control.mouse_callback)

        if hasattr(self, 'initialize_features'):
            self.initialize_features()

        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            frame_disp = frame.copy()

            if ui_control.mode == 'track' and ui_control.mode_switch:
                ui_control.mode_switch = False
                init_state = ui_control.get_bb()
                self.initialize(frame, init_state)

            # Draw box
            if ui_control.mode == 'select':
                cv.rectangle(frame_disp, ui_control.get_tl(), ui_control.get_br(), (255, 0, 0), 2)
            elif ui_control.mode == 'track':
                state = self.track(frame)
                state = [int(s) for s in state]
                cv.rectangle(frame_disp, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
                             (0, 255, 0), 5)

            # Put text
            font_color = (0, 0, 0)
            if ui_control.mode == 'init' or ui_control.mode == 'select':
                cv.putText(frame_disp, 'Select target', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, font_color, 1)
                cv.putText(frame_disp, 'Press q to quit', (20, 55), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                           font_color, 1)
            elif ui_control.mode == 'track':
                cv.putText(frame_disp, 'Tracking!', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                           font_color, 1)
                cv.putText(frame_disp, 'Press r to reset', (20, 55), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                           font_color, 1)
                cv.putText(frame_disp, 'Press q to quit', (20, 80), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                           font_color, 1)
            # Display the resulting frame
            self.imshow(display_name, frame_disp)
            key = cv.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('r'):
                ui_control.mode = 'init'

        # When everything done, release the capture
        cap.release()
        # if not ON_COLAB:
        #     cv.destroyAllWindows()

    def reset_tracker(self):
        pass

    def press(self, event):
        if event.key == 'p':
            self.pause_mode = not self.pause_mode
            print("Switching pause mode!")
        elif event.key == 'r':
            self.reset_tracker()
            print("Resetting target pos to gt!")

    def init_visualization(self):
        # plt.ion()
        self.pause_mode = False
        self.fig, self.ax = plt.subplots(1)
        # self.fig.canvas.manager.window.move(800, 50)
        # self.fig.canvas.manager.window.wm_geometry("+%d+%d" % (100, 50))

        self.fig.canvas.mpl_connect('key_press_event', self.press)
        plt.tight_layout()


    def visualize(self, image, state):
        self.ax.cla()
        self.ax.imshow(image)

        if len(state) == 4:
            pred = patches.Rectangle((state[0], state[1]), state[2], state[3], linewidth=2, edgecolor='r', facecolor='none')
        elif len(state) == 8:
            p_ = np.array(state).reshape((4, 2))
            pred = patches.Polygon(p_, linewidth=2, edgecolor='r', facecolor='none')
        else:
            print('Error: Unknown prediction region format.')
            exit(-1)

        self.ax.add_patch(pred)

        if hasattr(self, 'gt_state') and False:
            gt_state = self.gt_state
            rect = patches.Rectangle((gt_state[0], gt_state[1]), gt_state[2], gt_state[3], linewidth=1, edgecolor='g',
                                     facecolor='none')
            self.ax.add_patch(rect)
        self.ax.set_axis_off()
        self.ax.axis('equal')
        plt.draw()
        plt.pause(0.001)

        if self.pause_mode:
            plt.waitforbuttonpress()

    def _read_image(self, image_file: str):
        return cv.cvtColor(cv.imread(image_file), cv.COLOR_BGR2RGB)

