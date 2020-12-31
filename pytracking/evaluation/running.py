import numpy as np
import multiprocessing
import os
from itertools import product
from pytracking.evaluation import Sequence, Tracker
from mtcnn import MTCNN
from keras_vggface.vggface import VGGFace
from keras_vggface import utils
from pytracking.face_identify import FaceIdentify


def run_sequence(seq: Sequence, tracker: Tracker, debug=False, face_detect=None):
    """Runs a tracker on a sequence."""

    base_results_path = '{}/{}'.format(tracker.results_dir, seq.name)
    results_path = '{}.txt'.format(base_results_path)
    times_path = '{}_time.txt'.format(base_results_path)

    if os.path.isfile(results_path) and not debug:
        return

    print('Tracker: {} {} {} ,  Sequence: {}'.format(tracker.name, tracker.parameter_name, tracker.run_id, seq.name))
    
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


def run_dataset(dataset, trackers, debug=False, threads=0, no_anno=False, pickledir=None):
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
            print("Load feature file...")
            face_identify = FaceIdentify(precompute_features_file=pickledir+"precompute_features.pickle")
            face_recog = face_detector
            print("Configuration Done")
        else:
            face_recog = None
        
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
