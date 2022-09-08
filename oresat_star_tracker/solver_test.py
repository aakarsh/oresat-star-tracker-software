import unittest
import sys
import os
import io
import cv2
import traceback
import numpy as np
from timeit import default_timer as timer
from olaf import logger
from .solver import Solver, SolverError
from .utils import read_image_file_to_numpy_buffer, read_preprocess_image

logger.add(sys.stdout, level="DEBUG")

class TestSolver(unittest.TestCase):

    def setUp(self):
        logger.debug("entry::setup")
        logger.debug("creating solver")

        test_data_folder = '/home/debian/oresat-star-tracker-software/misc/test-data'

        config_path = f'{test_data_folder}/exp1000/calibration.txt'
        median_path = f'{test_data_folder}/exp1000/median_image.png'

        self._solver = Solver() #config_path=config_path, median_path=median_path)
        logger.debug("startup solver")
        self._solver.startup()
        logger.debug("startup complete")
        logger.debug("exit::setup")


    def test_run(self):
        '''Test solution of images.'''
        test_data_folder = '/home/debian/oresat-star-tracker-software/misc/test-data'
        exposures = ['exp1000', 'exp2500']
        paths = [ f'{test_data_folder}/exp1000/samples' ]
        duration = -1

        x_size = 1280
        y_size = 960

        expected_x_size = 640
        expected_y_size = 480

        solutions = {
            f'{test_data_folder}/exp1000/samples/1.bmp' : [ 339.28, 327.29, -141.00749 ], # dec, ra, ori
            f'{test_data_folder}/exp1000/samples/8.bmp' : [ 35.0021, 232.023, 92.3595  ]  # dec, ra, ori
        }

        # DEC, RA, ORI =
        expected_solutions = [
            [ 74.798045847, 271.257311164, 84.470568   ],
            [ 26.4559966942, 246.783421908, 131.84151  ],
            [ -10.929918586, 226.95598348, 161.47508   ],
            [ -1.89170292451, 176.127248319, -157.29154],
            [ 52.1052996922, 122.156847972, -118.22782 ],
            [ 49.3122891085, 270.202436708, 112.54466  ],
            [ 1.44809534992, 237.50518643, 151.45222   ],
            [ -14.4507215149, 200.764441589, -177.89854]]
        
        for path in paths:
            N = 9 # Number of samples
            for i in range(1, N):
                image_file_name = "%d.bmp" % i
                image_path = path + '/' + image_file_name
                logger.info(f'image_path: {image_path}')
                img_data = read_preprocess_image(image_path, y_size, x_size, expected_y_size, expected_x_size)
 
                #y_size, x_size)  #
                solution = None
                if image_path in solutions:
                    logger.info(f'Found known solutions {solutions[image_path]}')
                    solution = solutions[image_path]
                try:
                    start = timer()
                    # Run the solver
                    if not solution:
                        self.assertRaises(SolverError, self._solver.solve, img_data)
                        logger.info(f'Unsuccessful solution {image_path}')
                    else:
                        dec, ra, ori = self._solver.solve(img_data)
                        if solution:
                            expected_dec, expected_ra, expected_ori = solution
                            self.assertTrue(np.isclose(ra, expected_ra), "ra is not close")
                            self.assertTrue(np.isclose(dec,expected_dec), "dec is not close")
                            self.assertTrue(np.isclose(ori, expected_ori), "ori is not close")
                            logger.info(f'Successful solution {image_path}')

                    stop = timer()
                    duration = stop - start
                    logger.info(f'Solve time[seconds]: {duration}')
                except:
                    traceback.print_exc()
                    logger.info("Failed to solve %s" % image_path)
                self.assertTrue(duration < 10)

if __name__ == """__main__""":
    unittest.main()
