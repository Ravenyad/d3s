import os
import sys
import argparse

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

from pytracking.evaluation.otbdataset import OTBDataset
from pytracking.evaluation.nfsdataset import NFSDataset
from pytracking.evaluation.uavdataset import UAVDataset
from pytracking.evaluation.tpldataset import TPLDataset
from pytracking.evaluation.votdataset import VOTDataset
from pytracking.evaluation.vot18dataset import VOT18Dataset
from pytracking.evaluation.lasotdataset import LaSOTDataset
from pytracking.evaluation.trackingnetdataset import TrackingNetDataset
from pytracking.evaluation.got10kdataset import GOT10KDatasetTest, GOT10KDatasetVal, GOT10KDatasetLTRVal
from pytracking.evaluation.running import run_stream
from pytracking.evaluation import Tracker


def run_survey(tracker_name, tracker_param, run_id=None, vidpath=None, debug=0, threads=0):
    """Run tracker on sequence or dataset.
    args:
        tracker_name: Name of tracking method.
        tracker_param: Name of parameter file.
        run_id: The run id.
        dataset_name: Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).
        sequence: Sequence number or name.
        debug: Debug level.
        threads: Number of threads.
    """

    trackers = [Tracker(tracker_name, tracker_param, run_id)]

    run_stream(vidpath, trackers, debug, threads)


def main():
    parser = argparse.ArgumentParser(description='Run tracker on sequence or dataset.')
    parser.add_argument('tracker_name', type=str, help='Name of tracking method.')
    parser.add_argument('tracker_param', type=str, help='Name of parameter file.')
    parser.add_argument('--runid', type=int, default=None, help='The run id.')
    parser.add_argument('--videopath', type=str, default=None, help='Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).')
    parser.add_argument('--debug', type=int, default=0, help='Debug level.')
    parser.add_argument('--threads', type=int, default=0, help='Number of threads.')

    args = parser.parse_args()

    run_survey(args.tracker_name, args.tracker_param, args.runid, args.videopath, args.debug, args.threads)


if __name__ == '__main__':
    main()
