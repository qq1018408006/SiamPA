from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from siamban.core.config import cfg
from siamban.tracker.siamban_tracker3 import SiamPATracker

TRACKS = {
          'SiamPATracker': SiamPATracker
         }


def build_tracker(model):
    return TRACKS[cfg.TRACK.TYPE](model)
