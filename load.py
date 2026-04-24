from fast_alpr import ALPR

alpr = ALPR(
    detector_model="yolo-v9-s-608-license-plate-end2end",
    ocr_model="cct-xs-v2-global-model",
)

