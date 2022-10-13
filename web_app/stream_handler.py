import os
import json
import base64
from pathlib import Path
from datetime import datetime

import numpy as np
import cv2
from loguru import logger
from tornado.websocket import WebSocketHandler
from tornado.ioloop import PeriodicCallback
from tornado.options import options, define
from oresat_star_tracker.solver import Solver
from oresat_star_tracker.camera import Camera


MOCK = os.getenv('MOCK', 'false')

define('mock', default=MOCK.lower() == 'true', help='run in a mocked mode')

capture_only=True

class StreamHandler(WebSocketHandler):

    DATA_DIR = 'data'
    SYSFS_PATH = '/sys/class/pru/prucam'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.stream = PeriodicCallback(self.new_frame, 5000)
        if not capture_only:
            self.solver = Solver(trace_intermediate_images=True)
            self.solver.startup()
        self.img_num = 1

        Path(self.DATA_DIR).mkdir(parents=True, exist_ok=True)

        self._mock = options.mock
        if not self._mock:
            self.camera = Camera()
            self.camera.power_on()

    def open(self, *args, **kwargs):
        '''New connection'''

        self.new_frame()
        self.stream.start()

    def on_message(self, message: str):
        '''New message received'''

        logger.debug(f'new message: {message}')

        # Receive data from WebSocket
        json_dict = json.loads(message)
        json_dict_keys = json_dict.keys()

        try:
            if 'coarse_time' in json_dict_keys:
                coarse_time = json_dict['coarse_time']
                logger.info(f'setting coarse time to {coarse_time}')
                with open(self.SYSFS_PATH + '/context_settings/coarse_time', 'w') as f:
                    f.write(coarse_time)

            if 'coarse_time' in json_dict_keys:
                fine_time = json_dict['fine_time']
                logger.info(f'setting fine time to {fine_time}')
                with open(self.SYSFS_PATH + '/context_settings/fine_time', 'w') as f:
                    f.write(fine_time)

            if 'auto_exposure' in json_dict_keys:
                auto_exposure = json_dict['auto_exposure']
                logger.info(f'setting auto exposure to {auto_exposure}')
                with open(self.SYSFS_PATH + '/auto_exposure_settings/ae_enable', 'w') as f:
                    f.write(int(auto_exposure))
        except Exception as exc:
            logger.error(str(exc))
            self.write_message(json.dumps({'error': str(exc)}))

    def on_close(self):
        '''Connection closed'''

        if self.stream.is_running:
            self.stream.stop()

        logger.debug('connection closed')

    def new_frame(self):
        '''Send new frame to client'''

        dec = 0.0
        ra = 0.0
        ori = 0.0
        error = ''

        correlation_timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

        if self._mock:
            data = cv2.imread(f'misc/test-data/downsample/samples/{self.img_num}.bmp')

            self.img_num += 1
            self.img_num %= 7
            if self.img_num == 0:
                self.img_num += 1
        else:
            data = self.camera.capture()
            correlation_timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

        if not capture_only:
            try:
                dec, ra, ori = self.solver.solve(data, trace_id=correlation_timestamp)
            except Exception as exc:
                error = str(exc)
                logger.error(exc)
        else:
            dec, rec, ori=[-1, -1, -1]

        # datetime.now().strftime('capture-%Y-%m-%d-%H-%M-%S.bmp')
        file_name = 'capture-'+correlation_timestamp+'.bmp'
        file_path = f'{self.DATA_DIR}/{file_name}'
        cv2.imwrite(file_path, data)
        logger.info(f'wrote new capture to: {file_path}')

        _, img = cv2.imencode('.jpg', data)
        frame = base64.b64encode(img).decode('ascii')
        frame_message = 'data:image/jpg;base64, ' + frame


    def inject_properties(img):
        prucam_sysfs_ctx_settings = "/sys/devices/platform/prudev/context_settings/"
        prucam_sysfs_ae_settings = "/sys/devices/platform/prudev/auto_exposure_settings/"

        # possible camera context settings
        ctx_settings = [
            "analog_gain",
            "blue_gain",
            "coarse_time",
            "context",
            "digital_binning",
            "fine_time",
            "frame_len_lines",
            "global_gain",
            "green1_gain",
            "green2_gain",
            "red_gain",
            "x_size",
            "y_odd_inc",
            "y_size",
            ]

        # auto exposure settings
        ae_settings = [
            "ae_ag_en",
            "ae_ag_exposure_hi",
            "ae_ag_exposure_lo",
            "ae_damp_gain",
            "ae_damp_max",
            "ae_damp_offset",
            "ae_dark_cur_thresh",
            "ae_dg_en",
            "ae_enable",
            "ae_luma_target",
            "ae_max_ev_step",
            "ae_max_exposure",
            "ae_min_ana_gain",
            "ae_min_ev_step",
            "ae_min_exposure",
            "ae_roi_x_size",
            "ae_roi_x_start_offset",
            "ae_roi_y_size",
            "ae_roi_y_start_offset",
            ]


        font = cv2.FONT_HERSHEY_SIMPLEX

        img_height, img_width, img_depth = img.shape
        start_position=(10,10)
        previous_position = start_position

        time = datetime.now()
        text_seperation=20

        constant_properties = [('time', time), ('img_shape', img.shape)]
        for (name, val) in constant_properties:
            img = cv2.putText(img, f'{name}: {val}', previous_position, font, .4, (0, 255, 0), 1, cv2.LINE_AA)
            previous_position = (previous_position[0], previous_position[1]+text_seperation)

        for param in ctx_settings:
            param_path = os.path.join(prucam_sysfs_ctx_settings, param)
            with open(param_path, 'r+') as f:
                val = str(f.read())
                img = cv2.putText(img, f'{param}: {val}', previous_position, font, .4, (0, 255, 0), 1, cv2.LINE_AA)
                previous_position = (previous_position[0], previous_position[1]+text_seperation)

        for param in ae_settings:
            param_path = os.path.join(prucam_sysfs_ae_settings, param)
            with open(param_path, 'r+') as f:
                val = str(f.read())
                img = cv2.putText(img, f'{param}: {val}', previous_position, font, .4, (0, 255, 0), 1, cv2.LINE_AA)
                previous_position = (previous_position[0], previous_position[1]+text_seperation)
        return img



        def read_trace_data(trace_name):
            if capture_only:
                    return None
            file_name = f'solver-{trace_name}-{correlation_timestamp}.jpg'
            file_path = f'{self.DATA_DIR}/{file_name}'
            data = cv2.imread(file_path)
            # Resize image
            scale_percent = 80 # Percent of original size.
            width = int(data.shape[1] * scale_percent / 100)
            height = int(data.shape[0] * scale_percent / 100)
            dim = (width, height)
            data_resized = cv2.resize(data, dim, interpolation = cv2.INTER_AREA)
            _, img = cv2.imencode('.jpg', data_resized)
            frame = base64.b64encode(img).decode('ascii')
            frame_message = 'data:image/jpg;base64, ' + frame
            return frame_message


        logger.info(f'wrote new capture to: {file_path}')


        message = {
            'frame': frame_message,
            'grey_frame': read_trace_data('grey'),
            'thresh_frame': read_trace_data('thresh'),
            'contours_frame': read_trace_data('contours'),

            'dec': dec,
            'ra': ra,
            'ori': ori,
            'error': error,
        }

        self.write_message(json.dumps(message))
        logger.debug('new frame was sent')
