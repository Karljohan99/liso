from collections import defaultdict
from copy import deepcopy
from glob import glob
from inspect import getsourcefile
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from config_helper.config import parse_config
from liso.datasets.torch_dataset_commons import (
    LidarDataset,
    LidarSample,
    add_lidar_rows_to_kitti_sample,
    get_weighted_random_sampler_dropping_samples_without_boxes,
    lidar_dataset_collate_fn,
    recursive_npy_dict_to_torch,
    worker_init_fn,
)
from liso.kabsch.shape_utils import Shape

class TartuRawDataset(LidarDataset):
    def __init__(
        self,
        cfg,
        *,
        shuffle: bool,
        use_geom_augmentation: bool,
        use_skip_frames: str,
        mode="train",
        training_target="flow",
        size=None,
        verbose=False,
        pure_inference_mode=False,
        get_only_these_specific_samples=None,
        path_to_augmentation_db=None,
        path_to_mined_boxes_db=None,
        for_tracking=False,
        need_flow=True,
    ) -> None:
        super().__init__(
            cfg,
            shuffle=shuffle,
            mode=mode,
            use_geom_augmentation=use_geom_augmentation,
            use_skip_frames=use_skip_frames,
            path_to_augmentation_db=path_to_augmentation_db,
            path_to_mined_boxes_db=path_to_mined_boxes_db,
            for_tracking=for_tracking,
            need_flow=need_flow,
        )

        assert not shuffle, "can't do it - use dataloader"
        assert mode == "train", f"only mode train makes sense, but got {mode}"
        assert training_target in ("flow", "object"), training_target

        self.training_target = training_target
        self.verbose = verbose
        self.pure_inference_mode = pure_inference_mode
        dataset_root = Path(cfg.data.paths.tartu.local)

        # Import training data from dataset root
        dataset_root = dataset_root.joinpath("train")
        sample_files = sorted(glob(str(Path(dataset_root).joinpath("*.npy"))))

        if pure_inference_mode:
            assert not use_geom_augmentation
            self.sample_files = sample_files
        else:
            if size is not None: # Take a random subset of the dataset 
                if size < len(sample_files):
                    self.sample_files = np.random.choice(sample_files, size=size, replace=False).tolist()
                else:
                    self.sample_files = sample_files
                    print(f"Warning: Requested size={size} samples, but I only have {len(sample_files)}")
            else:
                self.sample_files = sample_files

        if get_only_these_specific_samples: # Take only specific samples
            filtered_samples = []
            for sample in self.sample_files:
                for special_sample in get_only_these_specific_samples:
                    if special_sample in sample:
                        filtered_samples.append(sample)
            self.sample_files = filtered_samples

            if size is not None:
                assert size == len(filtered_samples), "either stop requesting specific samples or request size=None"
            
            print(f"Warning: Only requested {len(self.sample_files)}")

        downsample_dataset_keep_ratio = self.cfg.data.setdefault("downsample_dataset_keep_ratio", 1.0)

        if self.mode == "train" and self.cfg.data.downsample_dataset_keep_ratio != 1.0: # Keep a specific ratio of the dataset
            self.dataset_sequence_is_messed_up = True
            print(f"Downsampling dataset by {downsample_dataset_keep_ratio}. From {len(sample_files)} to..")
            self.sample_files = np.random.choice(
                self.sample_files,
                size=int(downsample_dataset_keep_ratio * len(self.sample_files)),
                replace=False,
            ).tolist()
            print(f"... {len(sample_files)} samples.")

        seq_samples = [("_".join(Path(el).stem.split("_")[0:-1]), Path(el)) for el in self.sample_files]
        
        # everything must be numpy arrays:
        # https://github.com/pytorch/pytorch/issues/13246#issuecomment-715050814

        sequences_with_sample_names = defaultdict(list)
        self.max_sequence_len = 500
        for seq_name, sample_name in seq_samples:
            if len(sequences_with_sample_names[seq_name]) < self.max_sequence_len:
                sequences_with_sample_names[seq_name].append(sample_name)
            else:
                subsection_idx = 0

                while (len(sequences_with_sample_names[seq_name + f"chop{subsection_idx}"]) >= self.max_sequence_len):
                    subsection_idx += 1

                sequences_with_sample_names[seq_name + f"chop{subsection_idx}"].append(sample_name)

        for seq_name in sequences_with_sample_names:
            sequences_with_sample_names[seq_name] = sorted(sequences_with_sample_names[seq_name])
            
        # Put all sequences to list
        self.per_seq_sample_paths = []
        for seq_name in sequences_with_sample_names:
            self.per_seq_sample_paths.append(sequences_with_sample_names[seq_name])

        # Get sequence lengths
        self.sequence_lens = np.array([len(el) for el in self.per_seq_sample_paths])

        print("sequence lengths: ", self.sequence_lens)

        assert self.data_use_skip_frames in ("only", "both", "never")

        if self.mode == "train" and self.cfg.data.flow_source != "gt":
            pred_flow_path = Path(self.cfg.data.paths.tartu.slim_flow[self.cfg.data.flow_source]["local"])
            self.pred_flow_path = pred_flow_path
            print(f"Loading flow seperately from source {pred_flow_path}")

        self.sample_files = np.array(self.sample_files).astype(np.string_)

    def get_samples_for_sequence(self, sequence_idx: int, start_idx_in_sequence: int, sequence_length: int) -> List[LidarSample]:
        
        assert not self.dataset_sequence_is_messed_up

        global_start_idx_of_sequence = int(np.sum(self.sequence_lens[:sequence_idx]))
        global_start_idx = global_start_idx_of_sequence + start_idx_in_sequence
        # global_end_idx = global_start_idx + sequence_length
        chosen_samples = [LidarSample(
            idx=global_start_idx + idx,
            sample_name=str(self.per_seq_sample_paths[sequence_idx][start_idx_in_sequence + idx].stem),
            timestamp=0,
            full_path=str(self.per_seq_sample_paths[sequence_idx][start_idx_in_sequence + idx]))
            for idx in range(sequence_length)]

        return chosen_samples

    def get_scene_index_for_scene_name(self, seq_name: str) -> int:
        for idx in range(len(self.per_seq_sample_paths)):
            if seq_name in self.per_seq_sample_paths[idx][0].as_posix():
                return idx

    def get_consecutive_sample_idxs_for_sequence(self, sequence_idx: int) -> List[LidarSample]:

        assert not self.dataset_sequence_is_messed_up
        
        if sequence_idx >= len(self.sequence_lens):
            print("Ran out of sequences!")
            return None
        samples_in_sequence = self.get_samples_for_sequence(
            sequence_idx=sequence_idx,
            start_idx_in_sequence=0,
            sequence_length=self.sequence_lens[sequence_idx],
        )
        return samples_in_sequence

    def __getitem__(self, index):
        self.initialize_loader_saver_if_necessary()
        self.initialize_dbs_if_necessary()

        # Load sample
        fname = str(self.sample_files[index], encoding="utf-8")
        sample_content = self.loader_saver_helper.load_sample(fname, np.load, allow_pickle=True).item()

        # Drop lidar intensity if 'use_lidar_intensity' is set to false
        if not self.cfg.data.use_lidar_intensity:
            self.drop_intensities_from_pcls_in_sample(sample_content)

        if self.for_tracking:
            #src_key = "t0"
            #target_key = "t1"
            #src_trgt_time_delta_s = 0.1
            #self.drop_unused_timed_keys_from_sample(sample_content, "foo", "bar", "baz")
            raise ValueError("tracking not available")
        else:
            src_key, target_key, delete_target_key, src_trgt_time_delta_s = self.select_time_keys()
            self.drop_unused_timed_keys_from_sample(sample_content, src_key, target_key, delete_target_key)

        # KITTI (RAW) SPECIFIC: only needed to create tracking DB (raydrop augmentation)
        add_lidar_rows_to_kitti_sample(sample_content, time_keys=(src_key, target_key))
        sample_content = self.drop_points_on_kitti_vehicle(sample_content, src_key, target_key)

        # Restructure_sample
        sample_content = self.move_keys_to_subdict(sample_content, move_these_keys=("kiss_",), subdict_target_key="kiss_icp", drop_substr_from_moved_keys="kiss_")
        sample_content = self.move_keys_to_subdict(sample_content)

        self.add_reverse_odometry_to_sample(sample_content)

        if self.need_flow and self.training_target == "object":
            assert (self.cfg.data.flow_source != "gt"), "no gt flow available for TARTU Raw"
            self.load_add_flow_to_sample_content(fname, sample_content, src_key, target_key)

            if self.for_tracking:
                # also add flow from t1->t2
                assert src_key == "t0", src_key
                assert target_key == "t1", target_key
                self.load_add_flow_to_sample_content(fname, sample_content, src_key="t1", target_key="t2")

        if not self.pure_inference_mode:  # we only have slim flow for train dataset
            if self.mined_boxes_db is not None:
                self.load_add_mined_boxes_to_sample_content(Path(fname).stem, sample_content, )

        # Augment data
        if (not self.for_tracking  # WE CANT AUGMENT WHEN TRACKING!
            and not self.pure_inference_mode and self.use_geom_augmentation and self.cfg.data.augmentation.active):

            self.augment_sample_content(sample_content, src_key, target_key, "tartu")

        meta = {"sample_id": sample_content["name"]}
        del sample_content["name"]

        # Assemble data sample
        if self.for_tracking:
            sample_data_ta = self.assemble_sample_data(deepcopy(sample_content), "t0", "t1", src_trgt_time_delta_s)

            if self.need_reverse_time_sample_data:
                sample_data_tb = self.assemble_sample_data(sample_content, "t1", "t2", src_trgt_time_delta_s)
            else:
                sample_data_tb = {"gt": {}}

            for gt_source in {"gt", self.cfg.data.flow_source}:
                #t2 has been rempped
                self.drop_unused_timed_keys_from_sample(sample_data_ta[gt_source], "ta", "tb", "t2")
                #t0 has been rempped
                self.drop_unused_timed_keys_from_sample(sample_data_tb[gt_source], "ta", "tb", "t0")

            return recursive_npy_dict_to_torch(sample_data_ta), recursive_npy_dict_to_torch(sample_data_tb), {}, meta

        else:
            sample_data_ta = self.assemble_sample_data(deepcopy(sample_content), src_key, target_key, src_trgt_time_delta_s)

            if self.need_reverse_time_sample_data:
                sample_data_tb = self.assemble_sample_data(sample_content, target_key, src_key, src_trgt_time_delta_s)
                
            else:
                sample_data_tb = {"gt": {}}

        if self.verbose:
            print("Loaded sample: {0}".format(Path(fname).stem))

        # Convert arrays to torch tensors
        sample_data_ta = recursive_npy_dict_to_torch(sample_data_ta)
        if self.cfg.loss.supervised.centermaps.active:
            sample_data_ta["gt"].update(self.get_motion_based_centermaps(sample_data_ta))

        sample_data_tb = recursive_npy_dict_to_torch(sample_data_tb)
        if (self.need_reverse_time_sample_data and self.cfg.loss.supervised.centermaps.active):
            sample_data_tb["gt"].update(self.get_motion_based_centermaps(sample_data_tb))

        if self.mode == "train":
            augm_sample_ta = self.create_augmented_sample_from_flow_cluster_detector_and_box_snippet_db(src_trgt_time_delta_s, sample_data_ta)
        else:
            augm_sample_ta = {}

        return (sample_data_ta, sample_data_tb, augm_sample_ta, meta,)

    def extract_boxes_for_timestamp(self, _sample_content: Dict[str, np.ndarray], _src_key: str, _target_key: str) -> Shape:
        # Tartu Raw has no objects
        return Shape.createEmpty(), np.array([], dtype=str)

    def get_has_valid_scene_flow_label(self, sample_content, src_key):
        return np.zeros_like(sample_content[f"pcl_{src_key}"]["pcl"][:, 0], dtype=bool)

def get_tartu_train_dataset(
    cfg,
    use_skip_frames: str,
    use_geom_augmentation=True,
    shuffle=True,
    size=None,
    verbose=False,
    get_only_these_specific_samples=None,
    target="flow",
    path_to_augmentation_db: str = None,
    path_to_mined_boxes_db: str = None,
    need_flow_during_training: bool = True
    ):

    extra_loader_kwargs = {"shuffle": shuffle}

    # Initialize tartu raw dataset
    train_dataset = TartuRawDataset(
        shuffle=False,  # only needed for val datasets
        mode="train",
        cfg=cfg,
        use_geom_augmentation=use_geom_augmentation,
        use_skip_frames=use_skip_frames,
        size=size,
        verbose=verbose,
        get_only_these_specific_samples=get_only_these_specific_samples,
        training_target=target,
        path_to_augmentation_db=path_to_augmentation_db,
        path_to_mined_boxes_db=path_to_mined_boxes_db,
        need_flow=need_flow_during_training,
    )
    
    if path_to_mined_boxes_db is not None:
        sample_file_stems = [Path(str(sf, encoding="utf-8")).stem for sf in train_dataset.sample_files]
        weighted_random_sampler = get_weighted_random_sampler_dropping_samples_without_boxes(path_to_mined_boxes_db,
                                                                                              extra_loader_kwargs,
                                                                                              train_dataset,
                                                                                              sample_file_stems)
                                
        extra_loader_kwargs["sampler"] = weighted_random_sampler
    
    # Create torch dataloader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        pin_memory=True,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        collate_fn=lidar_dataset_collate_fn,
        worker_init_fn=worker_init_fn,
        **extra_loader_kwargs,
    )
    return train_loader, train_dataset

"""
# For debugging
def main():
    default_cfg_file = (
        Path(getsourcefile(lambda: 0)).parent.parent
        / Path("config")
        / Path("liso_config.yml")
    )

    cfg = parse_config(default_cfg_file, extra_cfg_args=("data_source_tartu",))
    train_ds = TartuRawDataset(cfg, allow_data_augmentation=True)
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=1,
        num_workers=0,
        collate_fn=lidar_dataset_collate_fn,
        worker_init_fn=worker_init_fn,
    )
    test_el = next(iter(train_loader))
    print(test_el)


if __name__ == "__main__":
    main()
"""