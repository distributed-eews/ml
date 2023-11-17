from pathlib import Path
from typing import Set, List

import numpy as np
import pipeline_functions
import redis
import pickle


class Pipeline:
    def __init__(self, path: Path, warmup_duration: int, name: str = "untitled"):
        self._states: dict = {}
        self._pipeline: list = []
        self._names: Set[str] = set()
        self._path: Path = path

        self._name = name
        self._redis_client = redis.Redis(host="localhost", port="6379", db=0)

        self.warmup_duration = warmup_duration
        self._redis_client.set(self._name + "~warmup_x", pickle.dumps([]))
        self._redis_client.set(self._name + "~warmup_x_len", 0)

        self._processing_function = self._warmup

    def set_name(self, name: str) -> str:
        i = 0
        while True:
            new_name = f'{self._name}_{name}_{i}'
            if new_name not in self._names:
                self._names.add(new_name)
                break
            i += 1
        return new_name

    def get_redis_client(self) -> redis.Redis:
        return self._redis_client

    def reset(self):
        pass

    def _build_pipeline(self, init_x: np.ndarray):
        self._pipeline = []
        with open(self._path) as f:
            x = init_x
            for line in f:
                pipeline_function: pipeline_functions.PipelineFunction = eval("pipeline_functions." + line)
                pipeline_function.set_parent(self)
                pipeline_function.set_initial_state(x)
                x = pipeline_function.compute(x)
                self._pipeline.append(pipeline_function)

        return x

    def process(self, x: np.ndarray) -> np.ndarray:  # Generator state machine
        return self._processing_function(x)

    def _process(self, x: np.ndarray) -> np.ndarray:
        for pipeline_function in self._pipeline:
            x = pipeline_function.compute(x)
        return x

    def _warmup(self, x: np.ndarray) -> np.ndarray:
        # Get the latest warmup x
        warmup_x: List[bytes] = pickle.loads(self._redis_client.get(self._name + "~warmup_x"))
        warmup_x_len: int = int(self._redis_client.get(self._name + "~warmup_x_len"))

        # Update state
        warmup_x.append(x.dumps())
        warmup_x_len += len(x)

        if warmup_x_len >= self.warmup_duration:
            # Convert all the warmup xs to numpy array
            warmup_x_np: List[np.ndarray] = [pickle.loads(b) for b in warmup_x]
            x = np.concatenate(warmup_x_np, axis=0)
            x = self._build_pipeline(x)
            self._processing_function = self._process

            return x

        # Save state
        print("Warming Up")
        self._redis_client.set(self._name + "~warmup_x", pickle.dumps(warmup_x))
        self._redis_client.set(self._name + "~warmup_x_len", warmup_x_len)

        return np.array([np.nan])