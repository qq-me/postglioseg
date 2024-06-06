"""PostGlioSeg, автор - Никишев Иван Олегович"""
import highdicom as hd
from pydicom.sr.codedict import codes

from pydicom.sr.coding import Code

# ALGO_FAMILY = Code()

# ALGO = hd.AlgorithmIdentificationSequence("Postoperative glioblastoma segmentation", family=1)
NECROSIS = hd.seg.SegmentDescription(
    segment_number=1,
    segment_label='necrosis',
    segmented_property_category=codes.SCT.Glioblastoma, # type:ignore
    segmented_property_type=codes.SCT.Necrosis,# type:ignore
    algorithm_type=hd.seg.SegmentAlgorithmTypeValues.MANUAL, # won't work in automatic and all libraries use manual as well it seems
    # algorithm_identification = 'Necrosis segmentation',
)
EDEMA = hd.seg.SegmentDescription(
    segment_number=2,
    segment_label='edema',
    segmented_property_category=codes.SCT.Glioblastoma,# type:ignore
    segmented_property_type=codes.SCT.Edema,# type:ignore
    algorithm_type=hd.seg.SegmentAlgorithmTypeValues.MANUAL,
)

ENCHANCING_TUMOR = hd.seg.SegmentDescription(
    segment_number=3,
    segment_label='enchancing tumor',
    segmented_property_category=codes.SCT.Glioblastoma,# type:ignore
    segmented_property_type=codes.SCT.Tumor,# type:ignore
    algorithm_type=hd.seg.SegmentAlgorithmTypeValues.MANUAL,
)

SEGMENTS = [NECROSIS, EDEMA, ENCHANCING_TUMOR]