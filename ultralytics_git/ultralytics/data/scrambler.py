from albumentations.core.transforms_interface import ImageOnlyTransform

import math

import numpy as np
       
    
class Scrambler:
    """
    Blockwise image scrambling transformation
    """
    def __init__(self, p=1.0, block_size=4, num_blocks=4,
                 seed=42, channels=3, pixel_permutation=None, block_permutation=None,
                 inversion=None, inversion_percentage=0.0, shift=0, shift_dir='left',
                 xor=None, unique_blocks=False) -> None:
        self.prob = p
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.pixels_in_group = self.block_size * self.num_blocks
        self.seed = seed
        self.channels = channels
        self.unique_blocks = unique_blocks
        self.shift = shift
        self.shift_dir = shift_dir
        self.xor = xor

        self.rng = np.random.default_rng(seed=self.seed)

        if pixel_permutation is None:
            if not self.unique_blocks:
                self.pixel_permutation = np.arange(0, self.block_size * self.block_size * self.channels)
                self.rng.shuffle(self.pixel_permutation)
            else:
                pixel_permutations = []
                for _ in range(self.num_blocks * self.num_blocks):
                    pixel_permutation = np.arange(0, self.block_size * self.block_size * self.channels)
                    self.rng.shuffle(pixel_permutation)
                    pixel_permutations.append(pixel_permutation)
                self.pixel_permutation = np.array(pixel_permutations, dtype=np.int32)
        else:
            self.pixel_permutation = np.array(pixel_permutation, dtype=np.int32)
        # self.pixel_permutation = [1, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 33, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]
        print(f"Pixel arangement ({len(self.pixel_permutation)}):")
        print(self.pixel_permutation)

        if block_permutation is None:
            self.block_permutation = np.arange(0, self.num_blocks * self.num_blocks)
            self.rng.shuffle(self.block_permutation)
        else:
            self.block_permutation = np.array(block_permutation, dtype=np.int32)
        # self.block_permutation = [1, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]
        print(f"Block arangement ({len(self.block_permutation)}):")
        print(self.block_permutation)
        
        if inversion is None:
            if not self.unique_blocks:
                self.inversion = np.arange(0, self.block_size * self.block_size * self.channels)
                self.rng.shuffle(self.inversion)
                cut_off_index = math.floor(len(self.inversion) * inversion_percentage)
                self.inversion = self.inversion[:cut_off_index]
            else:
                inversions = []
                for _ in range(self.num_blocks * self.num_blocks):
                    inversion = np.arange(0, self.block_size * self.block_size * self.channels)
                    self.rng.shuffle(inversion)
                    cut_off_index = math.floor(len(inversion) * inversion_percentage)
                    inversion = inversion[:cut_off_index]
                    inversions.append(inversion)
                self.inversion = np.array(inversions, dtype=np.int32)
        else:
            self.inversion = np.array(inversion, dtype=np.int32)
        # self.block_permutation = [1, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]
        print(f"Inverse arangement ({len(self.inversion)}):")
        print(self.inversion)
        
        print(f"Bit shift: {self.shift} to {self.shift_dir}")
        
        if xor is not None:
            if isinstance(xor, bool) and xor == True:
                if not self.unique_blocks:
                    self.xor = self.rng.integers(0, 255, size=self.block_size * self.block_size * self.channels, dtype=np.uint8)
                else:
                    xors = []
                    for _ in range(self.num_blocks * self.num_blocks):
                        xor = self.rng.integers(0, 255, size=self.block_size * self.block_size * self.channels, dtype=np.uint8)
                        xors.append(xor)
                    self.xor = np.array(xors, dtype=np.int8)
            else:
                self.xor = np.array(xor, dtype=np.int8)
            print(f'XORing with {self.xor}')
        else:
            self.xor = None
        
        if self.unique_blocks and self.pixel_permutation.shape[0] != self.inversion.shape[0]:
            raise ValueError("If `unique_blocks` is true then pixel_permutation and inversion must have same number of rows")

    def negative_positive_transform(self, img, code, channel=None, block=None, copy=False):
        if copy:
            img = img.copy()
        if channel is None:
            # Unique code across all channels
            if block is None:
                # same code in every block
                img[:, :, :, code] = 255 - img[:, :, :, code]
            else:
                # unique code in each block
                img[:, :, block, code] = 255 - img[:, :, block, code]
        else:
            # Same code for every channel
            if block is None:
                # same code in every block
                img[:, :, :, channel, code] = 255 - img[:, :, :, channel, code]
            else:
                # unique code in each block
                img[:, :, block, channel, code] = 255 - img[:, :, block, channel, code]
        return img
    
    def do_xor(self, img, code, channel=None, block=None, copy=False):
        if copy:
            img = img.copy()
        if channel is None:
            # Unique code across all channels
            if block is None:
                # same code in every block
                img = np.bitwise_xor(img, code)
            else:
                # unique code in each block
                img[:, :, block, :] = np.bitwise_xor(img[:, :, block, :], code)
        else:
            # Same code for every channel
            if block is None:
                # same code in every block
                img[:, :, :, channel, :] = np.bitwise_xor(img[:, :, :, channel, :], code)
            else:
                # unique code in each block
                img[:, :, block, channel, :] = np.bitwise_xor(img[:, :, block, channel, :], code)
        return img
    
    def bitshift(self, img, num, dir='right', copy=False):
        if copy:
            img = img.copy()
            
        def circular_rshift_np(x, shift, bits=8):
            shift %= bits
            return np.bitwise_or(np.right_shift(x, shift), np.left_shift(x, bits - shift)) & ((1 << bits) - 1)

        def circular_lshift_np(x, shift, bits=8):
            shift %= bits
            return np.bitwise_or(np.left_shift(x, shift), np.right_shift(x, bits - shift)) & ((1 << bits) - 1)

        dir = dir.lower().strip()
        
        if dir == 'left':
            img = circular_lshift_np(img, num)
        elif dir == 'right':
            img = circular_rshift_np(img, num)
        else:
            raise ValueError('Unknown direction')
        
        return img
    
    def __call__(self, img, copy=True):
        h, w, c = [*img.shape, 1] if len(img.shape) == 2 else img.shape
        # print(h, w, c)
        if h % self.pixels_in_group != 0 or w % self.pixels_in_group != 0:
            raise ValueError(f"For scrambling, image width and height must be a multiple of {self.pixels_in_group}")
        
        if copy:
            img = img.copy()

        num_groups_h, num_groups_w = h // self.pixels_in_group, w // self.pixels_in_group
        
        img = img.transpose((2, 0, 1)) # # transpose to channels-first
        img = img.reshape((c, num_groups_h, self.num_blocks, self.block_size,  num_groups_w, self.num_blocks, self.block_size)) # reshape to groups of blocks
        img = img.transpose((1, 4, 2, 5, 0, 3, 6)) # transpose to pixels-last

        # scramble pixels inside each block and each channel
        if c > self.channels and self.channels == 1:
            # if image has more channels than scrambler
            img = img.reshape((num_groups_h, num_groups_w, self.num_blocks * self.num_blocks, c, self.block_size * self.block_size)) # reshape to linear blocks

            for i in range(c):
                if self.unique_blocks:
                    for j in range(self.pixel_permutation.shape[0]):
                        img[:, :, j, i, :] = img[:, :, j, i, self.pixel_permutation[j]]
                        if len(self.inversion) > 0:
                            img = self.negative_positive_transform(img, code=self.inversion[j], channel=i, block=j)
                        if self.xor is not None:
                            img = self.do_xor(img, code=self.xor[j], channel=i, block=j)
                else:
                    img[:, :, :, i, :] = img[:, :, :, i, self.pixel_permutation]
                    if len(self.inversion) > 0:
                        img = self.negative_positive_transform(img, code=self.inversion[j], channel=i, block=None)
                    if self.xor is not None:
                        img = self.do_xor(img, code=self.xor[j], channel=i, block=None)
                
            img = img.reshape((num_groups_h, num_groups_w, self.num_blocks * self.num_blocks, c * self.block_size * self.block_size))
        elif c == self.channels:
            # if image has the same number of channels as scrambler
            img = img.reshape((num_groups_h, num_groups_w, self.num_blocks * self.num_blocks, c * self.block_size * self.block_size))
            if self.unique_blocks:
                for j in range(self.pixel_permutation.shape[0]):
                    img[:, :, j, :] = img[:, :, j, self.pixel_permutation[j]]
                    if len(self.inversion) > 0:
                        img = self.negative_positive_transform(img, code=self.inversion[j], channel=None, block=j)
                    if self.xor is not None:
                        img = self.do_xor(img, code=self.xor[j], channel=None, block=j)
            else:
                img = img[:, :, :, self.pixel_permutation]
                if len(self.inversion) > 0:
                    img = self.negative_positive_transform(img, code=self.inversion, channel=None, block=None)
                if self.xor is not None:
                    img = self.do_xor(img, code=self.xor, channel=None, block=None)
            
        else:
            raise ValueError(f"Number of channels in image is smaller than specified for scrambling ({c} -> {self.channels}")
        
        if self.shift != 0:
            img = self.bitshift(img, num=self.shift, dir=self.shift_dir)

        # scramble blocks inside each group
        img = img[:, :, self.block_permutation, :]

        # restore image
        img = img.reshape((num_groups_h, num_groups_w, self.num_blocks, self.num_blocks, c, self.block_size, self.block_size))
        img = img.transpose((4, 0, 2, 5, 1, 3, 6))
        img = img.reshape((c, h, w))
        img = img.transpose((1, 2, 0))

        if c == 1:
            img = img[:, :, 0]
        
        return img
    
class ScramblerAlbumentation(ImageOnlyTransform):
    def __init__(self, p=1.0, block_size=4, num_blocks=4,
                 seed=42, channels=3, pixel_permutation=None, block_permutation=None,
                 inversion=None, inversion_percentage=0.0, shift=0, shift_dir='left',
                 xor=None, unique_blocks=False) -> None:
        super().__init__(p=p)

        self.scrambler = Scrambler(block_size=block_size, num_blocks=num_blocks,
                                   seed=seed, channels=channels, pixel_permutation=pixel_permutation,
                                   block_permutation=block_permutation, inversion=inversion,
                                   inversion_percentage=inversion_percentage, shift=shift, shift_dir=shift_dir,
                                   xor=xor, unique_blocks=unique_blocks)
        

    def apply(self, img, copy=True, **params):
        return self.scrambler(img, copy)
    
class ScramblerTransform:
    def __init__(self, p=1.0, block_size=4, num_blocks=4,
                 seed=42, channels=3, pixel_permutation=None, block_permutation=None,
                 inversion=None, inversion_percentage=0.0, shift=0, shift_dir='left',
                 xor=None, unique_blocks=False, copy=True) -> None:

        self.p = p
        self.copy = copy
        self.scrambler = Scrambler(block_size=block_size, num_blocks=num_blocks,
                                   seed=seed, channels=channels, pixel_permutation=pixel_permutation,
                                   block_permutation=block_permutation, inversion=inversion,
                                   inversion_percentage=inversion_percentage, shift=shift, shift_dir=shift_dir,
                                   xor=xor, unique_blocks=unique_blocks)

    def __call__(self, labels):
        img = labels['img']
        img = self.scrambler(img, self.copy)
        labels['img'] = img
        return labels