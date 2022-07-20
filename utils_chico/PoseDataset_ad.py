import os
import numpy as np
import pickle as pk

from torch.utils.data import Dataset


class PoseDataset(Dataset):
    def __init__(self, 
                 data_path='safeHR',
                 split='train',
                 input_time_frames=10,
                 output_time_frames=25,
                 win_stride=0,
                 actions=None):
        super(PoseDataset).__init__()
        
        self.data_path = data_path
        self.split = split
        self.in_tf = input_time_frames
        self.out_tf = output_time_frames
        self.win_size = input_time_frames + output_time_frames
        self.win_stride = self.win_size if win_stride == 0 else win_stride
        
        self.actions = []
        for act in actions:
            if act.endswith('.pkl'):
                self.actions.append(act)
            else:
                self.actions.append(act + '.pkl')
                
        self.subjects = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08']
    
        self.windows = self.build_dataset()
        
        if split == 'train':
            self.actions = [action for action in self.actions if '_CRASH' not in action]
        elif split == 'test':
            self.actions = [action for action in self.actions if '_CRASH' in action]
            
        
        
    def __getitem__(self, index):
        return self.gt_all_scales[list(self.gt_all_scales.keys())[0]].shape[0]
    
    def __len__(self):
        return self.windows.shape[0]
        
    def build_dataset(self):
        
        all_data = []        
        
        for subject in self.subjects:
            sub_path = os.path.join(self.data_path, subject)
            sub_actions_paths = [os.path.join(sub_path, act) for act in os.listdir(sub_path) if ((act.endswith('.pkl')) & (act in self.actions))]
            # print('subj: ', subject)
            # print(sub_actions_paths)
            for sub_actions_path in sub_actions_paths:
                data = self.retrieve_data(sub_actions_path)
                splitted_windows = self.split_single_pose(data)
                all_data.append(splitted_windows)
        
        return np.concatenate(all_data)
    
    def retrieve_data(self, action_path):
        with open(action_path, 'rb') as f:
            all_data = pk.load(f)
        
        human_related_data = [x[0] for x in all_data]
        
        single_hpose_np = np.stack(human_related_data, axis=0)
        
        return single_hpose_np
    
    def split_single_pose(self, single_pose_array):
        T, _, _ = single_pose_array.shape
        
        iterations = np.ceil((T-self.win_size)/self.win_stride).astype(int)
        all_windows = []
        for segment in range(iterations):
            start_index = self.win_stride * segment
            end_index = start_index + self.win_size
            
            curr_win = single_pose_array[start_index:end_index]
            all_windows.append(curr_win)
        
        return np.stack(all_windows, axis=0)