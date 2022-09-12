'''solver.py

by Umair Khan, from the Portland State Aerospace Society
based on OpenStarTracker from Andrew Tennenbaum at the University of Buffalo
openstartracker.org
'''
import uuid
import sys
import time
import datetime

from  datetime import datetime
from os.path import abspath, dirname

import numpy as np
import cv2

from olaf import logger

from .beast import beast


class SolverError(Exception):
    '''An erro has occur for the :py:class:`solver`'''


class Solver:
    '''Solve star trackr images'''

    def __init__(self, db_path=None, config_path=None, median_path=None, blur_kernel_size=None):
        # Prepare constants
        self.P_MATCH_THRESH = 0.99
        self.YEAR = 1991.25
        self.MEDIAN_IMAGE = None
        self.S_DB = None
        self.SQ_RESULTS = None
        self.S_FILTERED = None
        self.C_DB = None

        self.data_dir = dirname(abspath(__file__)) + '/data'
        self.median_path = median_path if median_path else f'{self.data_dir}/median-image.png'
        self.config_path = config_path if config_path else f'{self.data_dir}/configuration.txt'
        self.db_path = db_path if db_path else f'{self.data_dir}/hipparcos.dat'
        # Load median image
        self.MEDIAN_IMAGE = cv2.imread(self.median_path)

        # Load configuration
        beast.load_config(self.config_path)

        # Enable blur kernel
        if blur_kernel_size:
            self.blur_kernel_size = blur_kernel_size

        logger.debug(f'__init__:Solver \n Median Path: {self.median_path}\n DB Path:{self.db_path}\n Config Path:{self.config_path}')


    def startup(self):
        '''Start up sequence. Loads median image, loads config file, and setups database.

        Seperate from :py:func:`__init__` as it take time to setup database.

        Raises
        -------
        SolverError
            start up failed
        '''

        data_dir = dirname(abspath(__file__)) + '/data'

        try:
            # Load median image
            self.MEDIAN_IMAGE = cv2.imread(self.median_path)

            # Load configuration
            beast.load_config(self.config_path)

            # Load star database
            self.S_DB = beast.star_db() # 0 seconds
            self.S_DB.load_catalog(self.db_path, self.YEAR) # 7 seconds

            # Filter stars
            self.SQ_RESULTS = beast.star_query(self.S_DB) # 1 sec
            self.SQ_RESULTS.kdmask_filter_catalog() # 8 seconds

            self.SQ_RESULTS.kdmask_uniform_density(beast.cvar.REQUIRED_STARS) # 23 seconds!

            self.S_FILTERED = self.SQ_RESULTS.from_kdmask()

            # Set up constellation database
            self.C_DB = beast.constellation_db(self.S_FILTERED, 2 + beast.cvar.DB_REDUNDANCY, 0) # 1 second

        except Exception as exc:
            raise SolverError(f'Startup sequence failed with {exc}')


    def _preprocess_img(self, orig_img, guid=None):
        if not guid:
            guid = str(uuid.uuid4())
        cv2.imwrite(f'/tmp/solver-original-{guid}.png', orig_img)

        # Ensure images are always processed on calibration size.
        orig_img = cv2.resize(orig_img, (beast.cvar.IMG_X, beast.cvar.IMG_Y))
        cv2.imwrite(f'/tmp/solver-resized-{guid}.png', orig_img)

        # Blur the image if a blur is specified.
        if self.blur_kernel_size:
            orig_img = cv2.blur(orig_img,(self.blur_kernel_size, self.blur_kernel_size))
            cv2.imwrite(f'/tmp/solver-blurred-{guid}.png', orig_img)


        # Process the image for solving
        logger.info(f"start image pre-processing- {guid}")
        tmp = orig_img.astype(np.int16) - self.MEDIAN_IMAGE
        img = np.clip(tmp, a_min=0, a_max=255).astype(np.uint8)
        img_grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        cv2.imwrite(f'/tmp/solver-grey-{guid}.png', img_grey)

        return img_grey

    def _find_contours(self, img_grey, guid=None):
        if not guid: guid = str(uuid.uuid4())
        logger.info(f'entry: solve():{beast.cvar.IMG_X}, {beast.cvar.IMG_Y}')

        # Remove areas of the image that don't meet our brightness threshold and then extract
        # contours
        ret, thresh = cv2.threshold(img_grey, beast.cvar.THRESH_FACTOR * beast.cvar.IMAGE_VARIANCE,
                                    255, cv2.THRESH_BINARY)
        cv2.imwrite(f'/tmp/solver-thresh-{guid}.png', thresh)
        logger.info("finished image pre-processing")

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # contours_img = cv2.drawContours(img_grey, contours, -1, (0,255,0), 1)
        # cv2.imwrite(f'/tmp/solver-countours-{guid}.png', contours_img)
        logger.info(f"Number of  countours: {len(contours)}")

        return contours

    def _find_stars(self, img_grey, contours, guid = None):
        if not guid:
            guid = str(uuid.uuid4())
        star_list = []
        for c in contours:
            M = cv2.moments(c)
            if M['m00'] > 0:
                # this is how the x and y position are defined by cv2
                cx = M['m10'] / M['m00']
                cy = M['m01'] / M['m00']
                flux = float(cv2.getRectSubPix(img_grey, (1, 1), (cx, cy))[0, 0])

                # Add the list to star_list
                star_list.append([cx, cy,flux])
        return np.array(star_list)

    def _star_list_to_beast_stars_db(self, star_list):
        img_stars = beast.star_db()
        image_center = (beast.cvar.IMG_X / 2.0, beast.cvar.IMG_Y / 2.0)
        number_of_stars = star_list.shape[0]

        for idx in range(number_of_stars):
            cx, cy, flux = star_list[idx]
            cx_center, cy_center  = image_center
            # The center pixel is used as the approximation of the brightest pixel
            img_stars += beast.star(cx - cx_center, cy - cy_center, flux, -1)
        return img_stars

    def _generate_match(self, lis, img_stars):
            x = lis.winner.R11
            y = lis.winner.R21
            z = lis.winner.R31

            r = beast.cvar.MAXFOV / 2

            self.SQ_RESULTS.kdsearch(x, y, z, r,
                                     beast.cvar.THRESH_FACTOR * beast.cvar.IMAGE_VARIANCE)

            # Estimate density for constellation generation
            self.C_DB.results.kdsearch(x, y, z, r,
                                       beast.cvar.THRESH_FACTOR * beast.cvar.IMAGE_VARIANCE)

            fov_stars = self.SQ_RESULTS.from_kdresults()
            fov_db = beast.constellation_db(fov_stars, self.C_DB.results.r_size(), 1)

            self.C_DB.results.clear_kdresults()
            self.SQ_RESULTS.clear_kdresults()

            img_const = beast.constellation_db(img_stars, beast.cvar.MAX_FALSE_STARS + 2, 1)

            nearest_match = beast.db_match(fov_db, img_const)

            if nearest_match.p_match > self.P_MATCH_THRESH:
                return nearest_match

            return None

    def _extract_orientation(self, match):
        if match is None:
            return None

        match.winner.calc_ori()
        dec = match.winner.get_dec()
        ra = match.winner.get_ra()
        ori = match.winner.get_ori()
        return dec, ra, ori

    def _solve_orientation(self, star_list):
        '''
        _solve_orientation
        '''
        img_stars = self._star_list_to_beast_stars_db(star_list)

        # Find the constellation matches
        img_stars_n_brightest = img_stars.copy_n_brightest(
            beast.cvar.MAX_FALSE_STARS + beast.cvar.REQUIRED_STARS)

        img_const_n_brightest = beast.constellation_db(img_stars_n_brightest,
                                                       beast.cvar.MAX_FALSE_STARS + 2, 1)

        lis = beast.db_match(self.C_DB, img_const_n_brightest)

        # Generate the match
        match = None
        if lis.p_match > self.P_MATCH_THRESH and lis.winner.size() >= beast.cvar.REQUIRED_STARS:
            match = self._generate_match(lis, img_stars)

        orientation = self._extract_orientation(match)

        if orientation is None:
            logger.info("Unable to find orientation for image!")
            raise SolverError('Solution failed for image')

        return orientation


    def solve(self, orig_img) -> (float, float, float):
        '''
        Return
        ------
        float, float, float
            dec - rotation about the y-axis,
            ra  - rotation about the z-axis,
            ori - rotation about the camera axis

        Raises
        -------
        SolverError
            No matches found.
        '''
        guid = str(uuid.uuid4())

        # Preprocess the image for solving
        img_grey  = self._preprocess_img(orig_img, guid=guid)

        # Find the countours of the stars from binary image.
        contours = self._find_contours(img_grey, guid=guid)

        # Find most promising stars to search with.
        star_list = self._find_stars(img_grey, contours, guid)

        # Find orientation using given stars.
        orientation  = self._solve_orientation(star_list)

        return orientation

