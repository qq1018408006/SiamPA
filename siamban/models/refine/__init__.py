from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from siamban.models.refine.refine2_1 import Refinement




REFINE = {
        'refinement':Refinement,
       }


def get_refine(name, **kwargs):
    return REFINE[name](**kwargs)

