import numpy as np
import os
from pytracking.evaluation.data import Sequence, BaseDataset, SequenceList


def SkripsiDataset():
    return SkripsiDatasetClass().get_sequence_list()


class SkripsiDatasetClass(BaseDataset):
    """ Skripsi Dataset

    Kumpulan dataset untuk keperluan skripsi Ervan A Haryadi
    Dataset diambil dari PIROPO Dataset, ChokePoint Dataset, dan pengumpulan data sendiri

    PIROPO Dataset : 
    ChokePoint Dataset :
    """
    def __init__(self):
        super().__init__()
        self.base_path = "/content/drive/My Drive/DATASET/Image Sequences/SkripsiSet"
        self.sequence_info_list = self._get_sequence_info_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_info_list])

    def _construct_sequence(self, sequence_info):
        sequence_path = sequence_info['path']
        nz = sequence_info['nz']
        ext = sequence_info['ext']
        start_frame = sequence_info['startFrame']
        end_frame = sequence_info['endFrame']

        init_omit = 0
        if 'initOmit' in sequence_info:
            init_omit = sequence_info['initOmit']
        
        skip_frame = None
        if "skip_frame" in sequence_info:
            skip_frame = sequence_info['skip_frame']

        if isinstance(ext, list):
            for e in ext:
                first_frame_path = '{base_path}/{sequence_path}/{frame:0{nz}}.{ext}'.format(base_path=self.base_path,
                                                                                            sequence_path=sequence_path,
                                                                                            frame=start_frame + init_omit,
                                                                                            nz=nz,
                                                                                            ext=e)
                if os.path.isfile(first_frame_path):
                    ext = e
                    break

            if isinstance(ext, list):
                raise Exception('Sequence {} not found'.format(sequence_info['name']))

        frames = ['{base_path}/{sequence_path}/{frame:0{nz}}.{ext}'.format(base_path=self.base_path, 
        sequence_path=sequence_path, frame=frame_num, nz=nz, ext=ext) for frame_num in range(start_frame+init_omit, end_frame+1)]

        # anno_path = '{}/{}'.format(self.base_path, sequence_info['anno_path'])

        # try:
        #     ground_truth_rect = np.loadtxt(str(anno_path), dtype=np.float64)
        # except:
        #     ground_truth_rect = np.loadtxt(str(anno_path), delimiter=',', dtype=np.float64)

        return Sequence(sequence_info['name'], frames, skip_frame=skip_frame)

    def __len__(self):
        '''Overload this function in your evaluation. This should return number of sequences in the evaluation '''
        return len(self.sequence_info_list)


    def _get_sequence_info_list(self):
        sequence_info_list = [
            {"name":"Selftake_1", "path":"Selftake_1","startFrame":0, "endFrame":399, "nz":5, "ext":"jpg"},
            {"name":"Selftake_2", "path":"Selftake_2","startFrame":0, "endFrame":399, "nz":5, "ext":"jpg"},
            {"name":"Selftake_3", "path":"Selftake_3","startFrame":0, "endFrame":599, "nz":5, "ext":"jpg"},
            {"name":"Piropo2B", "path":"Piropo2B","startFrame":0, "endFrame":837, "nz":5, "ext":"jpg", "skip_frame":[90, 478, 563, 571, 608, 645, 663, 733, 829]},
            {"name":"Piropo4A", "path":"Piropo4A","startFrame":1, "endFrame":653, "nz":5, "ext":"jpg", "initOmit":490},
            {"name":"ChokeP1ES2", "path":"ChokeP1E_S2","startFrame":0, "endFrame":2295, "nz":5, "ext":"jpg", "initOmit":100}
        ]

        # Masukin nanti
        # {"name":"ChokeP2LS5", "path":"ChokeP2L_S5","startFrame":0, "endFrame":754, "nz":5, "ext":"jpg", "initOmit":100}

        #Format : {"name": "Woman", "path": "Woman/img", "startFrame": 1, "endFrame": 597, "nz": 4, "ext": "jpg", "anno_path": "Woman/groundtruth_rect.txt"}
        return sequence_info_list
