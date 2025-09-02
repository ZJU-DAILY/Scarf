from unittest import TestCase

from numpy import arange

from flink.knob import KnobDef, KnobType
from utils.types import KnobUnit, StorageUnit


class TestKnobDef(TestCase):
    knob_def = KnobDef(
        "test.knob",
        KnobType.STORAGE_SIZE,
        StorageUnit.from_str("32 mb"),
        [StorageUnit.from_str("16 mb"), StorageUnit.from_str("64 mb")],
        True,
    )
    for i in arange(0, 1, 0.05):
        print(f"{i:.2f}", knob_def.get_value_from_percentile(i))
