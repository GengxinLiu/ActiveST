import random
import logging
import numpy as np
import scipy
import scipy.ndimage
import scipy.interpolate
import torch
import MinkowskiEngine as ME
import collections
from numpy import cross
from scipy.linalg import expm, norm


class RandomDropout(object):

    def __init__(self, dropout_ratio=0.2, dropout_application_ratio=0.5):
        """
        upright_axis: axis index among x,y,z, i.e. 2 for z
        """
        self.dropout_ratio = dropout_ratio
        self.dropout_application_ratio = dropout_application_ratio

    def __call__(self, coords, feats, labels):
        if random.random() < self.dropout_ratio:
            N = len(coords)
            inds = np.random.choice(N, int(N * (1 - self.dropout_ratio)), replace=False)
            return coords[inds], feats[inds], labels[inds], inds
        return coords, feats, labels, None


class RandomHorizontalFlip(object):

    def __init__(self, upright_axis, is_temporal):
        """
        upright_axis: axis index among x,y,z, i.e. 2 for z
        """
        self.is_temporal = is_temporal
        self.D = 4 if is_temporal else 3
        self.upright_axis = {'x': 0, 'y': 1, 'z': 2}[upright_axis.lower()]
        # Use the rest of axes for flipping.
        self.horz_axes = set(range(self.D)) - set([self.upright_axis])

    def __call__(self, coords, feats, labels):
        if random.random() < 0.95:
            for curr_ax in self.horz_axes:
                if random.random() < 0.5:
                    coord_max = np.max(coords[:, curr_ax])
                    coords[:, curr_ax] = coord_max - coords[:, curr_ax]
        return coords, feats, labels


class ChromaticTranslation(object):
    """Add random color to the image, input must be an array in [0,255] or a PIL image"""

    def __init__(self, trans_range_ratio=1e-1):
        """
        trans_range_ratio: ratio of translation i.e. 255 * 2 * ratio * rand(-0.5, 0.5)
        """
        self.trans_range_ratio = trans_range_ratio

    def __call__(self, coords, feats, labels):
        if random.random() < 0.95:
            tr = (np.random.rand(1, 3) - 0.5) * 255 * 2 * self.trans_range_ratio
            feats[:, :3] = np.clip(tr + feats[:, :3], 0, 255)
        return coords, feats, labels


class ChromaticAutoContrast(object):

    def __init__(self, randomize_blend_factor=True, blend_factor=0.5):
        self.randomize_blend_factor = randomize_blend_factor
        self.blend_factor = blend_factor

    def __call__(self, coords, feats, labels):
        if random.random() < 0.2:
            # mean = np.mean(feats, 0, keepdims=True)
            # std = np.std(feats, 0, keepdims=True)
            # lo = mean - std
            # hi = mean + std
            lo = np.min(feats, 0, keepdims=True)
            hi = np.max(feats, 0, keepdims=True)

            scale = 255 / (hi - lo)

            contrast_feats = (feats - lo) * scale

            blend_factor = random.random() if self.randomize_blend_factor else self.blend_factor
            feats = (1 - blend_factor) * feats + blend_factor * contrast_feats
        return coords, feats, labels


class ChromaticJitter(object):

    def __init__(self, std=0.01):
        self.std = std

    def __call__(self, coords, feats, labels):
        if random.random() < 0.95:
            noise = np.random.randn(feats.shape[0], 3)
            noise *= self.std * 255
            feats[:, :3] = np.clip(noise + feats[:, :3], 0, 255)
        return coords, feats, labels


class HueSaturationTranslation(object):

    @staticmethod
    def rgb_to_hsv(rgb):
        # Translated from source of colorsys.rgb_to_hsv
        # r,g,b should be a numpy arrays with values between 0 and 255
        # rgb_to_hsv returns an array of floats between 0.0 and 1.0.
        rgb = rgb.astype('float')
        hsv = np.zeros_like(rgb)
        # in case an RGBA array was passed, just copy the A channel
        hsv[..., 3:] = rgb[..., 3:]
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        maxc = np.max(rgb[..., :3], axis=-1)
        minc = np.min(rgb[..., :3], axis=-1)
        hsv[..., 2] = maxc
        mask = maxc != minc
        hsv[mask, 1] = (maxc - minc)[mask] / maxc[mask]
        rc = np.zeros_like(r)
        gc = np.zeros_like(g)
        bc = np.zeros_like(b)
        rc[mask] = (maxc - r)[mask] / (maxc - minc)[mask]
        gc[mask] = (maxc - g)[mask] / (maxc - minc)[mask]
        bc[mask] = (maxc - b)[mask] / (maxc - minc)[mask]
        hsv[..., 0] = np.select([r == maxc, g == maxc], [bc - gc, 2.0 + rc - bc], default=4.0 + gc - rc)
        hsv[..., 0] = (hsv[..., 0] / 6.0) % 1.0
        return hsv

    @staticmethod
    def hsv_to_rgb(hsv):
        # Translated from source of colorsys.hsv_to_rgb
        # h,s should be a numpy arrays with values between 0.0 and 1.0
        # v should be a numpy array with values between 0.0 and 255.0
        # hsv_to_rgb returns an array of uints between 0 and 255.
        rgb = np.empty_like(hsv)
        rgb[..., 3:] = hsv[..., 3:]
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        i = (h * 6.0).astype('uint8')
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        i = i % 6
        conditions = [s == 0.0, i == 1, i == 2, i == 3, i == 4, i == 5]
        rgb[..., 0] = np.select(conditions, [v, q, p, p, t, v], default=v)
        rgb[..., 1] = np.select(conditions, [v, v, v, q, p, p], default=t)
        rgb[..., 2] = np.select(conditions, [v, p, t, v, v, q], default=p)
        return rgb.astype('uint8')

    def __init__(self, hue_max, saturation_max):
        self.hue_max = hue_max
        self.saturation_max = saturation_max

    def __call__(self, coords, feats, labels):
        # Assume feat[:, :3] is rgb
        hsv = HueSaturationTranslation.rgb_to_hsv(feats[:, :3])
        hue_val = (random.random() - 0.5) * 2 * self.hue_max
        sat_ratio = 1 + (random.random() - 0.5) * 2 * self.saturation_max
        hsv[..., 0] = np.remainder(hue_val + hsv[..., 0] + 1, 1)
        hsv[..., 1] = np.clip(sat_ratio * hsv[..., 1], 0, 1)
        feats[:, :3] = np.clip(HueSaturationTranslation.hsv_to_rgb(hsv), 0, 255)

        # pcd = o3d.PointCloud()
        # pcd.points = o3d.Vector3dVector(coords)
        # pcd.colors = o3d.Vector3dVector(feats / 255)
        # o3d.draw_geometries([pcd])

        return coords, feats, labels


class HeightTranslation(object):

    def __init__(self, std=0.01):
        self.std = std

    def __call__(self, coords, feats, labels):
        if feats.shape[1] > 3 and random.random() < 0.95:
            feats[:, -1] += np.random.randn(1) * self.std
        return coords, feats, labels


class HeightJitter(object):

    def __init__(self, std):
        self.std = std

    def __call__(self, coords, feats, labels):
        if feats.shape[1] > 3 and random.random() < 0.95:
            feats[:, -1] += np.random.randn(feats.shape[0]) * self.std
        return coords, feats, labels


class NormalJitter(object):

    def __init__(self, std):
        self.std = std

    def __call__(self, coords, feats, labels):
        # normal jitter
        if feats.shape[1] > 6 and random.random() < 0.95:
            feats[:, 3:6] += np.random.randn(feats.shape[0], 3) * self.std
        return coords, feats, labels


class Compose(object):
    """Composes several transforms together."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        for t in self.transforms:
            args = t(*args)
        return args


def elastic_distortion(pointcloud, granularity, magnitude):
    """Apply elastic distortion on sparse coordinate space.

      pointcloud: numpy array of (number of points, at least 3 spatial dims)
      granularity: size of the noise grid (in same scale[m/cm] as the voxel grid)
      magnitude: noise multiplier
    """
    blurx = np.ones((3, 1, 1, 1)).astype('float32') / 3
    blury = np.ones((1, 3, 1, 1)).astype('float32') / 3
    blurz = np.ones((1, 1, 3, 1)).astype('float32') / 3
    coords = pointcloud[:, :3]
    coords_min = coords.min(0)

    # Create Gaussian noise tensor of the size given by granularity.
    noise_dim = ((coords - coords_min).max(0) // granularity).astype(int) + 3
    noise = np.random.randn(*noise_dim, 3).astype(np.float32)

    # Smoothing.
    for _ in range(2):
        noise = scipy.ndimage.filters.convolve(noise, blurx, mode='constant', cval=0)
        noise = scipy.ndimage.filters.convolve(noise, blury, mode='constant', cval=0)
        noise = scipy.ndimage.filters.convolve(noise, blurz, mode='constant', cval=0)

    # Trilinear interpolate noise filters for each spatial dimensions.
    ax = [
        np.linspace(d_min, d_max, d)
        for d_min, d_max, d in zip(coords_min - granularity, coords_min + granularity *
                                   (noise_dim - 2), noise_dim)
    ]
    interp = scipy.interpolate.RegularGridInterpolator(ax, noise, bounds_error=0, fill_value=0)
    pointcloud[:, :3] = coords + interp(coords) * magnitude
    return pointcloud


# gengxin
class ElasticDistortion(object):
    def __init__(self, ELASTIC_DISTORT_PARAMS=((0.2, 0.4), (0.8, 1.6))):
        self.ELASTIC_DISTORT_PARAMS = ELASTIC_DISTORT_PARAMS

    def __call__(self, coords, colors, labels):
        if random.random() < 0.95:
            for granularity, magnitude in self.ELASTIC_DISTORT_PARAMS:
                coords = elastic_distortion(coords, granularity, magnitude)
        return coords, colors, labels


def M(axis, theta):
    return expm(cross(np.eye(3), axis / norm(axis) * theta))


class RandomRotation(object):
    def __init__(self, ROTATION_AUGMENTATION_BOUND=(-np.pi, np.pi)):
        self.ROTATION_AUGMENTATION_BOUND = ROTATION_AUGMENTATION_BOUND

    def __call__(self, coords, colors, labels):
        if self.ROTATION_AUGMENTATION_BOUND is not None:
            if isinstance(self.ROTATION_AUGMENTATION_BOUND, collections.Iterable):
                rot_mats = []
                for axis_ind, rot_bound in enumerate(self.ROTATION_AUGMENTATION_BOUND):
                    theta = 0
                    axis = np.zeros(3)
                    axis[axis_ind] = 1
                    if rot_bound is not None:
                        theta = np.random.uniform(*rot_bound)
                    rot_mats.append(M(axis, theta))
                # Use random order
                np.random.shuffle(rot_mats)
                rot_mat = rot_mats[0] @ rot_mats[1] @ rot_mats[2]
                coords_aug = coords @ rot_mat
                return coords_aug, colors, labels
            else:
                raise ValueError()


class Voxelize(object):
    def __init__(self, voxel_size=0.02, random_scale=True, SCALE_AUGMENTATION_BOUND=(0.9, 1.1), ignore_label=-100):
        self.voxel_size = voxel_size
        self.random_scale = random_scale
        self.ignore_label = ignore_label
        self.SCALE_AUGMENTATION_BOUND = SCALE_AUGMENTATION_BOUND

    def __call__(self, coords, colors, labels):
        scale = 1 / self.voxel_size
        if self.random_scale and self.SCALE_AUGMENTATION_BOUND is not None:
            scale *= np.random.uniform(*self.SCALE_AUGMENTATION_BOUND)
        coords_aug = np.floor(coords * scale)
        map_index, remap_index = ME.utils.sparse_quantize(
            coords_aug, ignore_label=self.ignore_label, return_index=True, return_inverse=True, return_maps_only=True)
        coords_aug, feats, labels = coords_aug[map_index], colors[map_index], labels[map_index]

        return coords_aug, feats, labels, remap_index
