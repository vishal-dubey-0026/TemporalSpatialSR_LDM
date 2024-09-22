import cv2
import math
import pydicom as dicom
import numpy as np
import os
import os.path as osp
import random
import time
import torch
from pathlib import Path

import albumentations

import torch.nn.functional as F
from torch.utils import data as data

from basicsr.utils import DiffJPEG
from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
from basicsr.data.transforms import augment
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
from basicsr.utils.img_process_util import filter2D
from basicsr.data.transforms import paired_random_crop
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt

def readline_txt(txt_file):
    txt_file = [txt_file, ] if isinstance(txt_file, str) else txt_file
    out = []
    for txt_file_current in txt_file:
        with open(txt_file_current, 'r') as ff:
            out.extend([x[:-1] for x in ff.readlines()])

    return out

@DATASET_REGISTRY.register(suffix='basicsr')
class RealESRGANDataset(data.Dataset):
    """Dataset used for Real-ESRGAN model:
    Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    It loads gt (Ground-Truth) images, and augments them.
    It also generates blur kernels and sinc kernels for generating low-quality images.
    Note that the low-quality images are processed in tensors on GPUS for faster processing.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            meta_info (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
            Please see more options in the codes.
    """

    def __init__(self, opt, mode='training'):
        super(RealESRGANDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mode = mode
        self.rescale_gt = opt['rescale_gt']
        self.MAX_VAL = 1024
        # file client (lmdb io backend)
        self.paths = []
        self.paths_idx_to_key = {}
        root_path = self.opt['root_path']
        if 'dir_paths' in opt:
            for current_dir in opt['dir_paths']:
                for current_ext in opt['im_exts']:
                    self.paths.extend(sorted([str(x) for x in Path(current_dir).glob(f'**/*.{current_ext}')]))
        if 'txt_file_path' in opt:
            for current_txt in opt['txt_file_path']:
                videos_path = readline_txt(current_txt)
                print(f"loading dataset: {current_txt}")
                for ii, video_path in enumerate(videos_path):
                    print(f"{ii}/{len(videos_path)}")
                    video_path = f'{root_path}/{video_path}'
                    self.paths_idx_to_key[len(self.paths)] = video_path
                    self.paths.append(readline_txt(video_path)) # list of list, each list with one video frames
        if 'length' in opt:
            self.paths = random.sample(self.paths, opt['length'])
            
        print(f"[INFO] There are {len(self.paths)} samples in the {mode} dataset")
        if self.mode == 'training':
            # blur settings for the first degradation
            self.blur_kernel_size = opt['blur_kernel_size']
            self.kernel_list = opt['kernel_list']
            self.kernel_prob = opt['kernel_prob']  # a list for each kernel probability
            self.blur_sigma = opt['blur_sigma']
            self.betag_range = opt['betag_range']  # betag used in generalized Gaussian blur kernels
            self.betap_range = opt['betap_range']  # betap used in plateau blur kernels
            self.sinc_prob = opt['sinc_prob']  # the probability for sinc filters

            # blur settings for the second degradation
            self.blur_kernel_size2 = opt['blur_kernel_size2']
            self.kernel_list2 = opt['kernel_list2']
            self.kernel_prob2 = opt['kernel_prob2']
            self.blur_sigma2 = opt['blur_sigma2']
            self.betag_range2 = opt['betag_range2']
            self.betap_range2 = opt['betap_range2']
            self.sinc_prob2 = opt['sinc_prob2']

            # a final sinc filter
            self.final_sinc_prob = opt['final_sinc_prob']

            self.kernel_range1 = [x for x in range(3, opt['blur_kernel_size'], 2)]  # kernel size ranges from 7 to 21
            self.kernel_range2 = [x for x in range(3, opt['blur_kernel_size2'], 2)]  # kernel size ranges from 7 to 21
            # TODO: kernel range is now hard-coded, should be in the configure file
            # convolving with pulse tensor brings no blurry effect
            self.pulse_tensor = torch.zeros(opt['blur_kernel_size2'], opt['blur_kernel_size2']).float()
            self.pulse_tensor[opt['blur_kernel_size2']//2, opt['blur_kernel_size2']//2] = 1
        
        


    

    def read_scale_dcm(self, file_path):
        try:
            d = dicom.dcmread(file_path)
            img = np.array(d.pixel_array)
            
            if 'PixelSpacing' in d:
                (x, y) = d.PixelSpacing
                (x, y) = (float(x), float(y))
            else:
                # Set default PixelSpacing if not available
                (x, y) = (1.0, 1.0)
            img = cv2.resize(src = img, dsize=None, fx=x, fy=y)
            gt_size = self.opt['gt_size']
            img = cv2.resize(src = img, dsize=(gt_size, gt_size))
            img = ((img.astype(np.float32) / self.MAX_VAL) * 255).clip(0, 255).astype(np.uint8)
            return img
        except:
            return None



    # Function to compute intermediate frames using optical flow
    def interpolate_frames(self, frame1, frame2, time_idxs, num_interpolations):
        """
        Interpolates between two frames using optical flow.
        :param frame1: First frame (t=0).
        :param frame2: Second frame (t=1).
        :param num_interpolations: Number of frames to interpolate between frame1 and frame2.
        :return: List of interpolated frames.
        """

        
        # Convert to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY) if len(frame1.shape) == 3 else frame1
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY) if len(frame2.shape) == 3 else frame2

        # Calculate optical flow between frame1 and frame2
        flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 
                                            pyr_scale=0.5, 
                                            levels=3, 
                                            winsize=15, 
                                            iterations=3, 
                                            poly_n=5, 
                                            poly_sigma=1.2, 
                                            flags=0)

        # Prepare list of interpolated frames
        interpolated_frames = []

        # Generate intermediate frames
        for i in time_idxs:
            t = i / (num_interpolations + 1)  # Interpolation factor (0 < t < 1)

            # Interpolation for forward and backward flow
            h, w = gray1.shape
            # Create a meshgrid of pixel coordinates (x, y)
            x, y = np.meshgrid(np.arange(w), np.arange(h))

            # Stack the x and y grids to form the coordinate map (h, w, 2)
            flow_map = np.stack((x, y), axis=-1)

            # Compute the forward and backward flow
            flow_forward = flow_map + t * flow  # Warp towards frame2
            flow_backward = flow_map - (1 - t) * flow  # Warp towards frame1

            # Separate the flow fields into x and y coordinates for remap
            flow_forward_x = flow_forward[..., 0].astype(np.float32)
            flow_forward_y = flow_forward[..., 1].astype(np.float32)
            flow_backward_x = flow_backward[..., 0].astype(np.float32)
            flow_backward_y = flow_backward[..., 1].astype(np.float32)

            # Warp both frames based on the flow using cv2.remap
            frame1_warped = cv2.remap(frame1, flow_backward_x, flow_backward_y, interpolation=cv2.INTER_LINEAR)
            frame2_warped = cv2.remap(frame2, flow_forward_x, flow_forward_y, interpolation=cv2.INTER_LINEAR)

            # Blend the warped images based on the interpolation factor t
            interpolated_frame = cv2.addWeighted(frame1_warped, 1 - t, frame2_warped, t, 0)

            # Append the interpolated frame to the list
            interpolated_frames.append(interpolated_frame)

        return interpolated_frames




    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)
        retrial = True
        while retrial:
            retrial = False
            index = random.randint(0, len(self.paths) - 1) #random.sample(population = list(range(len(self.paths))), k = 1)
            video_path = self.paths[index]
            temp_frame_count = len(video_path)
            
            temporal_SR = 8 # for 8x temporal SR
            assert len(video_path) == temp_frame_count, f"{self.paths_idx_to_key[index]} has {len(video_path)}, but expecting {temp_frame_count}"
            # -------------------------------- Load gt images -------------------------------- #
            # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
            if (temp_frame_count - temporal_SR - 1) < 0:
                retrial = True; continue
            start_time_step = random.randint(0, temp_frame_count - temporal_SR - 1) 
            end_time_step = start_time_step + temporal_SR
            start_path, end_path = (video_path[start_time_step], video_path[end_time_step])
            root_path = self.opt['root_path']
            start_path = f'{root_path}/{start_path}'
            end_path = f'{root_path}/{end_path}'
            start_frame, end_frame = self.read_scale_dcm(start_path), self.read_scale_dcm(end_path)
            if not (isinstance(start_frame, np.ndarray) and isinstance(end_frame, np.ndarray)):
                retrial = True; continue
            mid_frame_idx = random.randint(start_time_step + 1,  end_time_step- 1)

            

            mid_time_idx = mid_frame_idx - start_time_step # [1 to 7]
            if (start_time_step + 1) == mid_frame_idx: 
                interp_frames = self.interpolate_frames(frame1 = start_frame, frame2 = end_frame, time_idxs = [mid_time_idx, mid_time_idx+1], num_interpolations = temporal_SR - 1)
                interp_frames = [start_frame.copy()] + interp_frames
                left = start_frame
                mid = self.read_scale_dcm(f'{root_path}/{video_path[mid_frame_idx]}') 
                right = self.read_scale_dcm(f'{root_path}/{video_path[mid_frame_idx + 1]}') 
            elif (end_time_step - 1) == mid_frame_idx:
                interp_frames = self.interpolate_frames(frame1 = start_frame, frame2 = end_frame, time_idxs = [mid_time_idx-1, mid_time_idx], num_interpolations = temporal_SR - 1)
                interp_frames = interp_frames + [end_frame.copy()]
                left = self.read_scale_dcm(f'{root_path}/{video_path[mid_frame_idx - 1]}') 
                mid = self.read_scale_dcm(f'{root_path}/{video_path[mid_frame_idx]}') 
                right = end_frame
            else:
                interp_frames = self.interpolate_frames(frame1 = start_frame, frame2 = end_frame, time_idxs = [mid_time_idx-1, mid_time_idx, mid_time_idx+1], num_interpolations = temporal_SR - 1)
                left = self.read_scale_dcm(f'{root_path}/{video_path[mid_frame_idx - 1]}') 
                mid = self.read_scale_dcm(f'{root_path}/{video_path[mid_frame_idx]}') 
                right = self.read_scale_dcm(f'{root_path}/{video_path[mid_frame_idx + 1]}') 
            
            if not (isinstance(left, np.ndarray) and isinstance(mid, np.ndarray) and isinstance(right, np.ndarray)):
                retrial = True; continue
            
        interp_image = np.stack(interp_frames, axis = -1) # use this as spaital GT, apply spatial SR on this image, spatially degrade this GT
        GT_image = np.stack([left, mid, right], axis = -1) # use this for final loss calculation as GT, model should output this as GT
        
        img_gt = np.concatenate([interp_image, GT_image], axis = -1)  # HxWxC with C = 6
        img_gt = img_gt.astype(np.float32) / 255 #[0 - 1]


        # avoid errors caused by high latency in reading files
        '''
        retry = 3
        while retry > 0:
            try:    
                img_bytes = self.file_client.get(gt_path, 'gt')
                img_gt = imfrombytes(img_bytes, float32=True)
            # except (IOError, OSError, AttributeError) as e:
            except:
                # logger = get_root_logger()
                # logger.warn(f'File client error: {e}, remaining retry times: {retry - 1}')
                # change another file to read
                index = random.randint(0, self.__len__())
                gt_path = self.paths[index]
                time.sleep(1)  # sleep 1s for occasional server congestion
            # else:
                # break
            finally:
                retry -= 1
        '''
        if self.mode == 'testing':
            if not hasattr(self, 'test_aug'):
                self.test_aug = albumentations.Compose([
                    albumentations.SmallestMaxSize(max_size=self.opt['gt_size']),
                    albumentations.CenterCrop(self.opt['gt_size'], self.opt['gt_size']),
                    ])
            img_gt = self.test_aug(image=img_gt)['image'] #
        elif self.mode == 'training':
            pass
        else:
            raise ValueError(f'Unexpected value {self.mode} for mode parameter')

        if self.mode == 'training':
            # -------------------- Do augmentation for training: flip, rotation -------------------- #
            img_gt = augment(img_gt, self.opt['use_hflip'], self.opt['use_rot'])

            # crop or pad to 400
            # TODO: 400 is hard-coded. You may change it accordingly
            h, w = img_gt.shape[0:2]
            if self.rescale_gt:
                crop_pad_size = max(min(h, w), self.opt['gt_size'])
            else:
                crop_pad_size = self.opt['crop_pad_size']
            # pad
            # if h < crop_pad_size or w < crop_pad_size:
                # pad_h = max(0, crop_pad_size - h)
                # pad_w = max(0, crop_pad_size - w)
                # img_gt = cv2.copyMakeBorder(img_gt, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
            while h < crop_pad_size or w < crop_pad_size:
                pad_h = min(max(0, crop_pad_size - h), h)
                pad_w = min(max(0, crop_pad_size - w), w)
                img_gt = cv2.copyMakeBorder(img_gt, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
                h, w = img_gt.shape[0:2]
            # crop
            if img_gt.shape[0] > crop_pad_size or img_gt.shape[1] > crop_pad_size:
                h, w = img_gt.shape[0:2]
                # randomly choose top and left coordinates
                top = random.randint(0, h - crop_pad_size)
                left = random.randint(0, w - crop_pad_size)
                img_gt = img_gt[top:top + crop_pad_size, left:left + crop_pad_size, ...]
            if self.rescale_gt and crop_pad_size != self.opt['gt_size']:
                img_gt = cv2.resize(img_gt, dsize=(self.opt['gt_size'],)*2, interpolation=cv2.INTER_AREA)
        elif self.mode == 'testing':
            pass
        else:
            raise ValueError(f'Unexpected value {self.mode} for mode parameter')

        # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range1)
        if np.random.uniform() < self.opt['sinc_prob']:
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                kernel_size,
                self.blur_sigma,
                self.blur_sigma, [-math.pi, math.pi],
                self.betag_range,
                self.betap_range,
                noise_range=None)
        # pad kernel
        pad_size = (self.blur_kernel_size - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range2)
        if np.random.uniform() < self.opt['sinc_prob2']:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = random_mixed_kernels(
                self.kernel_list2,
                self.kernel_prob2,
                kernel_size,
                self.blur_sigma2,
                self.blur_sigma2, [-math.pi, math.pi],
                self.betag_range2,
                self.betap_range2,
                noise_range=None)

        # pad kernel
        pad_size = (self.blur_kernel_size2 - kernel_size) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------------------- the final sinc kernel ------------------------------------- #
        if np.random.uniform() < self.opt['final_sinc_prob']:
            kernel_size = random.choice(self.kernel_range2)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=self.blur_kernel_size2)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = self.pulse_tensor

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt = img2tensor([img_gt], bgr2rgb=True, float32=True)[0] #CxHxW with C = 6 [0 - 1]
        kernel = torch.FloatTensor(kernel)
        kernel2 = torch.FloatTensor(kernel2)
        #gt_path = None
        return_d = {'gt': img_gt, 'kernel1': kernel, 'kernel2': kernel2, 'sinc_kernel': sinc_kernel}
        return return_d

    def __len__(self):
        return int(1e9) ###len(self.paths)

    def degrade_fun(self, conf_degradation, im_gt, kernel1, kernel2, sinc_kernel):
        if not hasattr(self, 'jpeger'):
            self.jpeger = DiffJPEG(differentiable=False)  # simulate JPEG compression artifacts

        ori_h, ori_w = im_gt.size()[2:4]
        sf = conf_degradation.sf

        # ----------------------- The first degradation process ----------------------- #
        # blur
        out = filter2D(im_gt, kernel1)
        # random resize
        updown_type = random.choices(
                ['up', 'down', 'keep'],
                conf_degradation['resize_prob'],
                )[0]
        if updown_type == 'up':
            scale = random.uniform(1, conf_degradation['resize_range'][1])
        elif updown_type == 'down':
            scale = random.uniform(conf_degradation['resize_range'][0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(out, scale_factor=scale, mode=mode)
        # add noise
        gray_noise_prob = conf_degradation['gray_noise_prob']
        if random.random() < conf_degradation['gaussian_noise_prob']:
            out = random_add_gaussian_noise_pt(
                out,
                sigma_range=conf_degradation['noise_range'],
                clip=True,
                rounds=False,
                gray_prob=gray_noise_prob,
                )
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=conf_degradation['poisson_scale_range'],
                gray_prob=gray_noise_prob,
                clip=True,
                rounds=False)
        # JPEG compression
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*conf_degradation['jpeg_range'])
        out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
        out = self.jpeger(out, quality=jpeg_p)

        # ----------------------- The second degradation process ----------------------- #
        # blur
        if random.random() < conf_degradation['second_order_prob']:
            if random.random() < conf_degradation['second_blur_prob']:
                out = filter2D(out, kernel2)
            # random resize
            updown_type = random.choices(
                    ['up', 'down', 'keep'],
                    conf_degradation['resize_prob2'],
                    )[0]
            if updown_type == 'up':
                scale = random.uniform(1, conf_degradation['resize_range2'][1])
            elif updown_type == 'down':
                scale = random.uniform(conf_degradation['resize_range2'][0], 1)
            else:
                scale = 1
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(
                    out,
                    size=(int(ori_h / sf * scale), int(ori_w / sf * scale)),
                    mode=mode,
                    )
            # add noise
            gray_noise_prob = conf_degradation['gray_noise_prob2']
            if random.random() < conf_degradation['gaussian_noise_prob2']:
                out = random_add_gaussian_noise_pt(
                    out,
                    sigma_range=conf_degradation['noise_range2'],
                    clip=True,
                    rounds=False,
                    gray_prob=gray_noise_prob,
                    )
            else:
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range=conf_degradation['poisson_scale_range2'],
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False,
                    )

        # JPEG compression + the final sinc filter
        # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
        # as one operation.
        # We consider two orders:
        #   1. [resize back + sinc filter] + JPEG compression
        #   2. JPEG compression + [resize back + sinc filter]
        # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
        if random.random() < 0.5:
            # resize back + the final sinc filter
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(
                    out,
                    size=(ori_h // sf, ori_w // sf),
                    mode=mode,
                    )
            out = filter2D(out, sinc_kernel)
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*conf_degradation['jpeg_range2'])
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)
        else:
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*conf_degradation['jpeg_range2'])
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)
            # resize back + the final sinc filter
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(
                    out,
                    size=(ori_h // sf, ori_w // sf),
                    mode=mode,
                    )
            out = filter2D(out, sinc_kernel)

        # clamp and round
        im_lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.

        return {'lq':im_lq.contiguous(), 'gt':im_gt}
