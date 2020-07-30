"""Run offline patching from GDC slides.

Generate WSI patches and save to disk as PNG files.
"""

import os
import threading
import uuid

from PIL import Image
from wsipre import slide


class PatchGenerator(object):
    """Generator of GDC WSI patches."""

    def __init__(self, slide_files, slide_level=0, random_tissue_patch=False,
                 patch_size=(299, 299), return_annotation=False):
        """
        Parameters
        ----------
        slide_files: list of 2-tuples
            WSI and .XML annotation file path pairs.
        slide_level: int
            Slide level to get patch from.
        random_tissue_patch: bool
            Whether to get random patch from tissue regions, ignoring
            annotations.
        patch_size: 2-tuple
            Patch size.
        return_annotation: bool
            Whether to output patch annotation.
        """
        self.slide_files = slide_files
        self.slide_level = slide_level
        self.random_tissue_patch = random_tissue_patch
        self.patch_size = patch_size
        self.return_annotation = return_annotation
        self.lock = threading.Lock()
        self.reset()
        self.n = len(slide_files)

    def _get_random_patch(self, selected_slide):
        wsi_file, xml_file = selected_slide

        wsi = slide.Slide(wsi_file, xml_file, 'asap')

        # Some slides have no detected tumor regions (label list is empty)
        # Just skip them
        if not wsi.labels:
            return 'No tumor annotations found.'

        patch, annotation = wsi.read_random_patch(
            level=self.slide_level, size=self.patch_size, target_class=1,
            min_class_area_ratio=0, polygon_type='area')

        if self.return_annotation:
            return patch, annotation, os.path.basename(wsi_file)
        else:
            return patch, os.path.basename(wsi_file)

    def _get_random_tissue_patch(self, selected_slide):
        if isinstance(selected_slide, (list, tuple)):
            wsi_file, _ = selected_slide
        else:
            wsi_file = selected_slide

        wsi = slide.Slide(wsi_file)
        patch = wsi.read_random_tissue_patch(
            level=self.slide_level, size=self.patch_size)

        return patch, os.path.basename(wsi_file)

    def reset(self):
        """Reset generator."""
        self.i = 0

    def __next__(self):
        with self.lock:
            if self.i >= self.n:
                self.reset()

            if self.random_tissue_patch:
                result = self._get_random_tissue_patch(
                    self.slide_files[self.i])
            else:
                result = self._get_random_patch(self.slide_files[self.i])

            self.i += 1

            return result


class OfflinePatcher(object):
    """Run offline patching."""

    def __init__(self, slide_files, target_dir, patch_size, slide_level=0,
                 get_random_tissue_patch=False):
        self.slide_files = slide_files
        self.target_dir = target_dir
        self.patch_size = patch_size
        self.slide_level = slide_level
        self.file_format = 'png'  # to preserve pixel values (unlike JPG...)
        self.filename = None

        self.patch_gen = PatchGenerator(
            slide_files=self.slide_files, slide_level=self.slide_level,
            random_tissue_patch=get_random_tissue_patch,
            patch_size=self.patch_size)

        # Make sure target directory exists
        if not os.path.isdir(self.target_dir):
            os.makedirs(self.target_dir)

    def _compose_path(self):
        # Make sure filename is unique
        unique_id = str(uuid.uuid4().hex)[:5]
        slide_file_name = os.path.splitext(self.filename)[0]
        # Remove 2nd part of name
        slide_file_name = os.path.splitext(slide_file_name)[0]
        unique_name = slide_file_name + '_' + unique_id

        unique_name += '.' + self.file_format.lower()  # Add file extension
        path = os.path.join(self.target_dir, unique_name)

        return path

    def _save(self, path):
        """Save WSI patch to disk.

        Save image to PNG format, in order to preserve the numpy array pixel
        values. There are many options to do this:
            - matplotlib.image.imsave
            - cv2.imwrite
            - skimage.io.imsave
            - PIL.Image.fromarray(patch).save
        Decided to use PIL.
        """
        self.patch.save(path)

    def _make_patch(self):
        self.patch, self.filename = next(self.patch_gen)
        file_path = self._compose_path()
        self._save(file_path)

    def run(self, n):
        """Generate and save indicated number of image patches per slide.

        Parameters
        ----------
        n: int
            Number of patches to generate (slides are selected in sequence).
        """
        # Upon keyboard interrupt save last patch to make sure it is not
        # corrupted
        print('Generating WSI patches')
        print('----------------------')
        try:
            for patch in range(n):
                print('\r' + f'{str(patch + 1)}/{str(n)}', end='')
                # Skip slides with no detected tumor regions
                result = next(self.patch_gen)
                if result == 'No tumor annotations found.':
                    continue
                self.patch, self.filename = result
                file_path = self._compose_path()
                self._save(file_path)
        except KeyboardInterrupt:
            file_path = self._compose_path()
            self._save(file_path)

        print()
