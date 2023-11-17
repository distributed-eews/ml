"""This module defines a BentoML service that uses a Keras model to classify
digits.
"""

from typing import Optional, List

import numpy as np
import bentoml
from pathlib import Path
import json

from bentoml.io import NumpyNdarray
from bentoml.io import JSON
from pydantic import BaseModel

from pipeline import Pipeline

BENTO_MODEL_TAG_P = "p_model:latest"
BENTO_MODEL_TAG_S = "s_model:latest"

p_detector_runner = bentoml.keras.get(BENTO_MODEL_TAG_P).to_runner()
s_detector_runner = bentoml.keras.get(BENTO_MODEL_TAG_S).to_runner()

wave_arrival_detector = bentoml.Service("wave_arrival_detector", runners=[p_detector_runner, s_detector_runner])

# Setting pipeline data
pipeline_p = Pipeline(Path("pipeline/model_p_best.pipeline"), 5)
# pipeline_s = Pipeline(Path("pipeline/model_s_best.pipeline"), 5)

class InputDataPWave(BaseModel):
    x: List[List[float]]
    metadata: dict

class InputDataSWave(BaseModel):
    x: list
    metadata: dict
    reset: Optional[bool]

# @wave_arrival_detector.api(input=JSON(pydantic_model=InputDataSWave), output=NumpyNdarray())
# def predict_s(input_data: json) -> np.ndarray:
#     # Unpack json
#     x = np.array(input_data.x)
#     metadata = input_data.metadata
#
#     if input_data.reset:
#         pipeline_s.reset()
#
#     # Preprocess x
#     preprocessed_x = pipeline_s.process(x)
#
#     # return s_detector_runner.predict.run(preprocessed_x)
#     return preprocessed_x

@wave_arrival_detector.api(input=JSON(pydantic_model=InputDataPWave), output=NumpyNdarray())
def predict_p(input_data: json) -> np.ndarray:
    # Unpack json
    x = np.array(input_data.x)
    metadata = input_data.metadata

    # Preprocess x
    preprocessed_x = pipeline_p.process(x)

    # Extract Insights

    # perform inference if initialization has been initialized
    return p_detector_runner.predict.run(preprocessed_x)
