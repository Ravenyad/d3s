import numpy as np
import multiprocessing
import os
from itertools import product
from pytracking.evaluation import Sequence, Tracker
from mtcnn import MTCNN
from keras_vggface.vggface import VGGFace
from keras_vggface import utils


def run_sequence(seq: Sequence, tracker: Tracker, debug=False, face_detect=None):
    """Runs a tracker on a sequence."""

    base_results_path = '{}/{}'.format(tracker.results_dir, seq.name)
    results_path = '{}.txt'.format(base_results_path)
    times_path = '{}_time.txt'.format(base_results_path)

    if os.path.isfile(results_path) and not debug:
        return

    print('Tracker: {} {} {} ,  Sequence: {}'.format(tracker.name, tracker.parameter_name, tracker.run_id, seq.name))
    return
    if debug:
        tracked_bb, exec_times = tracker.run(seq, debug=debug, facerecog=face_detect)
    else:
        try:
            tracked_bb, exec_times = tracker.run(seq, debug=debug, facerecog=face_detect)
        except Exception as e:
            print(e)
            return

    tracked_bb = np.array(tracked_bb).astype(float)
    exec_times = np.array(exec_times).astype(float)

    print('FPS: {}'.format(len(exec_times) / exec_times.sum()))
    if not debug:
        np.savetxt(results_path, tracked_bb, delimiter=',', fmt='%f')
        np.savetxt(times_path, exec_times, delimiter='\t', fmt='%f')


def run_dataset(dataset, trackers, debug=False, threads=0, no_anno=False):
    """Runs a list of trackers on a dataset.
    args:
        dataset: List of Sequence instances, forming a dataset.
        trackers: List of Tracker instances.
        debug: Debug level.
        threads: Number of threads to use (default 0).
    """
    if threads == 0:
        mode = 'sequential'
    else:
        mode = 'parallel'

    if mode == 'sequential':
        # Cek apakah pakai face detector atau tidak
        if no_anno:
            print("Entering self-detecting tracker...")
            print("Configuring face detector & recognizer...")
            face_detector = MTCNN()
            # face_identify = VGGFace()
            face_recog = face_detector
            print("Configuration Done")
        else:
            face_detector = None
        
        for seq in dataset:
            for tracker_info in trackers:
                run_sequence(seq, tracker_info, debug=debug, face_detect=face_recog)

    elif mode == 'parallel':
        param_list = [(seq, tracker_info, debug) for seq, tracker_info in product(dataset, trackers)]
        with multiprocessing.Pool(processes=threads) as pool:
            pool.starmap(run_sequence, param_list)
    print('Done')


def run_stream(path, tracker:Tracker, debug=False, threads=0):
    if threads == 0:
        mode = 'sequential'
    else:
        mode = 'parallel'

    base_results_path = '{}/{}'.format(tracker.results_dir, path)
    results_path = '{}.txt'.format(base_results_path)
    times_path = '{}_time.txt'.format(base_results_path)

    if os.path.isfile(results_path) and not debug:
        return

    # Run detection and tracking online / nonstop
    if mode == 'sequential':
        print('Tracker: {} {} {} , Source: {}'.format(tracker.name, tracker.parameter_name, 
        tracker.run_id, path))
        
        try:
            bboxes, exec_times = tracker.run_vidseq(path, debug=debug)
        except Exception as e :
            print(e)
            return

    elif mode == 'parallel':
        param_list = [(seq, tracker_info, debug) for seq, tracker_info in product(dataset, trackers)]
        with multiprocessing.Pool(processes=threads) as pool:
            pool.starmap(run_sequence, param_list)
    
    bboxes = np.array(bboxes).astype(float)
    exec_times = np.array(exec_times).astype(float)

    print('FPS: {}'.format(len(exec_times) / exec_times.sum()))
    if not debug:
        np.savetxt(results_path, bboxes, delimiter=',', fmt='%f')
        np.savetxt(times_path, exec_times, delimiter='\t', fmt='%f')
    
    print('Done')

# def precompute_features():
#     resnet50_features = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3),
#                                 pooling='avg')  # pooling: None, avg or max
#     def image2x(image_path):
#         img = image.load_img(image_path, target_size=(224, 224))
#         x = image.img_to_array(img)
#         x = np.expand_dims(x, axis=0)
#         x = utils.preprocess_input(x, version=1)  # or version=2
#         return x

#     def cal_mean_feature(image_folder):
#         face_images = list(glob.iglob(os.path.join(image_folder, '*.*')))

#         def chunks(l, n):
#             """Yield successive n-sized chunks from l."""
#             for i in range(0, len(l), n):
#                 yield l[i:i + n]

#         batch_size = 32
#         face_images_chunks = chunks(face_images, batch_size)
#         fvecs = None
#         for face_images_chunk in face_images_chunks:
#             images = np.concatenate([image2x(face_image) for face_image in face_images_chunk])
#             batch_fvecs = resnet50_features.predict(images)
#             if fvecs is None:
#                 fvecs = batch_fvecs
#             else:
#                 fvecs = np.append(fvecs, batch_fvecs, axis=0)
#         return np.array(fvecs).sum(axis=0) / len(fvecs)
    
#     return
