'''solver.py

by Umair Khan, from the Portland State Aerospace Society
based on OpenStarTracker from Andrew Tennenbaum at the University of Buffalo
openstartracker.org
'''


import time
import logging
import numpy as np
import cv2

from .beast import beast


class SolverError(Exception):
    '''An erro has occur for the :py:class:`solver`'''


class Solver:
    '''Solve star trackr images'''

    def __init__(self):

        # Prepare constants
        self.P_MATCH_THRESH = 0.99
        self.YEAR = 1991.25
        self.MEDIAN_IMAGE = None
        self.S_DB = None
        self.SQ_RESULTS = None
        self.S_FILTERED = None
        self.C_DB = None

    def startup(self, median_path: str, config_path: str, db_path: str):
        '''Start up sequence. Loads median image, loads config file, and setups database.

        Seperate from :py:func:`__init__` as it take time to setup database.

        Paramaters
        ----------
        median_path: str
            file path to median image
        config_path: str
            file path to config_file
        db_path: str
            file path to database file

        Raises
        -------
        SolverError
            start up failed
        '''

        try:
            # Load median image
            self.MEDIAN_IMAGE = cv2.imread(median_path)

            # Load configuration
            beast.load_config(config_path)

            # Load star database
            self.S_DB = beast.star_db()
            self.S_DB.load_catalog(db_path, self.YEAR)

            # Filter stars
            self.SQ_RESULTS = beast.star_query(self.S_DB)
            self.SQ_RESULTS.kdmask_filter_catalog()
            self.SQ_RESULTS.kdmask_uniform_density(beast.cvar.REQUIRED_STARS)
            self.S_FILTERED = self.SQ_RESULTS.from_kdmask()

            # Set up constellation database
            self.C_DB = beast.constellation_db(self.S_FILTERED, 2 + beast.cvar.DB_REDUNDANCY, 0)
        except Exception as exc:
            raise SolverError(f'Startup sequence failed with {exc}')

    def solve(self, img_path: str) -> (float, float, float):
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

        # Create and initialize variables
        img_stars = beast.star_db()
        match = None
        fov_db = None

        # Process the image for solving
        orig_img = cv2.imread(img_path)
        img = np.clip(orig_img.astype(np.int16) - self.MEDIAN_IMAGE,
                      a_min=0,
                      a_max=255).astype(np.uint8)
        img_grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Remove areas of the image that don't meet our brightness threshold and then extract
        # contours
        ret, thresh = cv2.threshold(img_grey,
                                    beast.cvar.THRESH_FACTOR * beast.cvar.IMAGE_VARIANCE,
                                    255,
                                    cv2.THRESH_BINARY)
        thresh_contours, contours, hierarchy = cv2.findContours(thresh, 1, 2)

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
                                                       beast.cvar.MAX_FALSE_STARS + 2,
                                                       1)
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
            raise SolverError(f'Solution failed for {img_path}.')

        return dec, ra, ori
