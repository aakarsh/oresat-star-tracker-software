'''solver.py

by Umair Khan, from the Portland State Aerospace Society
based on OpenStarTracker from Andrew Tennenbaum at the University of Buffalo
openstartracker.org
'''


import time

import numpy as np
import cv2
from os.path import abspath, dirname

import datetime
from  datetime import datetime

from olaf import logger

from .beast import beast

import sys

#logger.add(sys.stdout, level="DEBUG")

class SolverError(Exception):
    '''An erro has occur for the :py:class:`solver`'''


class Solver:
    '''Solve star trackr images'''

    def __init__(self, db_path=None, config_path=None, median_path=None):
        logger.debug("__init__:Solver")
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

            logger.info(" Entry beast.star_db()")
            self.S_DB = beast.star_db() # 0 seconds
            logger.info(" Entry load_catalog()")
            self.S_DB.load_catalog(self.db_path, self.YEAR) # 7 seconds
            logger.info(" Exit load_catalog()")

            # Filter stars
            logger.info(" Entry star_query()")
            self.SQ_RESULTS = beast.star_query(self.S_DB) # 1 sec
            logger.info(" Exit star_query()")
            self.SQ_RESULTS.kdmask_filter_catalog() # 8 secons

            logger.info(" Entry kdmask_uniform_density()")
            self.SQ_RESULTS.kdmask_uniform_density(beast.cvar.REQUIRED_STARS) # 23 seconds!
            logger.info(" Exit kdmask_uniform_density()")
            self.S_FILTERED = self.SQ_RESULTS.from_kdmask()

            # Set up constellation database
            logger.info(" Entry constallation_db()")
            self.C_DB = beast.constellation_db(self.S_FILTERED, 2 + beast.cvar.DB_REDUNDANCY, 0) # 1 second
            logger.info(" Exit constallation_db()")

        except Exception as exc:
            raise SolverError(f'Startup sequence failed with {exc}')

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
            start up failed
        '''
        logger.info(" ENTRY: solve()")

        # Create and initialize variables
        img_stars = beast.star_db()
        match = None
        fov_db = None

        # Process the image for solving
        logger.info("Start image pre-processing")
        tmp = orig_img.astype(np.int16) - self.MEDIAN_IMAGE
        img = np.clip(tmp, a_min=0, a_max=255).astype(np.uint8)
        img_grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Remove areas of the image that don't meet our brightness threshold and then extract
        # contours
        ret, thresh = cv2.threshold(img_grey, beast.cvar.THRESH_FACTOR * beast.cvar.IMAGE_VARIANCE,
                                    255, cv2.THRESH_BINARY)
        logger.info("Finished image pre-processing")
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Process the contours
        for c in contours:
            M = cv2.moments(c)

            if M['m00'] > 0:

                # this is how the x and y position are defined by cv2
                cx = M['m10'] / M['m00']
                cy = M['m01'] / M['m00']

                # see https://alyssaq.github.io/2015/computing-the-axes-or-orientation-of-a-blob/
                # for how to convert these into eigenvectors/values
                u20 = M['m20'] / M['m00'] - cx ** 2
                u02 = M['m02'] / M['m00'] - cy ** 2
                u11 = M['m11'] / M['m00'] - cx * cy

                # The center pixel is used as the approximation of the brightest pixel
                img_stars += beast.star(cx - beast.cvar.IMG_X / 2.0,
                                        cy - beast.cvar.IMG_Y / 2.0,
                                        float(cv2.getRectSubPix(img_grey, (1, 1), (cx, cy))[0, 0]),
                                        -1)

        # We only want to use the brightest MAX_FALSE_STARS + REQUIRED_STARS
        img_stars_n_brightest = img_stars.copy_n_brightest(
            beast.cvar.MAX_FALSE_STARS + beast.cvar.REQUIRED_STARS)
        img_const_n_brightest = beast.constellation_db(img_stars_n_brightest,
                                                       beast.cvar.MAX_FALSE_STARS + 2, 1)
        lis = beast.db_match(self.C_DB, img_const_n_brightest)

        # Generate the match
        if lis.p_match > self.P_MATCH_THRESH and lis.winner.size() >= beast.cvar.REQUIRED_STARS:
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
            near = beast.db_match(fov_db, img_const)

            if near.p_match > self.P_MATCH_THRESH:
                match = near

        if match is not None:
            match.winner.calc_ori()
            dec = match.winner.get_dec()
            ra = match.winner.get_ra()
            ori = match.winner.get_ori()
        else:
            logger.info(" SOLVING FAIL!!! ")
            raise SolverError('Solution failed for image')

        logger.info(f'EXIT: solve() result - {dec} {ra}, {ori}')

        return dec, ra, ori
