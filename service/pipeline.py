from pathlib import Path
import numpy as np
import pipeline_functions

class Pipeline:
    def __init__(self, path: Path):
        self._states: dict = {}
        self._pipeline: list = []
        self._path: Path = path

        # Build pipeline
        self._build_pipeline()

    def reset(self):
        self._build_pipeline()

    def _build_pipeline(self):
        self._pipeline = []
        with open(self._path) as f:
            for line in f:

                self._pipeline.append(eval("pipeline_functions." + line))

    def process(self, x: np.ndarray, metadata: dict):  # Generator state machine
        for pipeline_function in self._pipeline:
            x = pipeline_function.compute(x)
        return x


if __name__ == '__main__':

    print("Initializing test...")
    # Test reading pipeline success
    p0 = Pipeline(Path("pipeline/model_p_best.pipeline"))
    print("Success building pipeline p...")
    p1 = Pipeline(Path("pipeline/model_s_best.pipeline"))
    print("Success building pipeline s...")

    # Testing pipeline
    p = Pipeline(Path("pipeline/test.pipeline"))
    x = np.array([[1.0, 2.0, 0.0],
                  [1.0, 0.0, 0.0],
                  [1.0, 3.0, 0.0],
                  [1.0, 0.0, 0.0]])
    metadata = {'f': 20.0}
    print(f"Initial x:\n{x}\n")

    # Generate function
    convert_to_velocity = p.convert_to_velocity("1")
    exponential_smoothing = p.exponential_smoothing(0.5, "2")
    multi_exponential_smoothing = p.multi_exponential_smoothing(np.array([0.0, 0.5, 1.0]), "3")
    square = p.square("4")
    add_channels = p.add_channels("5")
    pairwise_ratio = p.pairwise_ratio(np.array([0.0, 0.5, 1.0]), "6")
    poly_decay = p.poly_decay(1, 1, "7")

    # test functions
    print(f"convert_to_velocity:\n{convert_to_velocity(x, metadata)}\n")
    print(f"exponential_smothing:\n{exponential_smoothing(x, metadata)}\n")
    print(f"multi_exponential_smoothing:\n{multi_exponential_smoothing(np.array([[1.0] for i in range(100)]), metadata)}\n")
    print(f"square:\n{square(x, metadata)}\n")
    print(f"add_channels:\n{add_channels(x, metadata)}\n")
    print(f"pairwise_ratio:\n{pairwise_ratio(x, metadata)}\n")
    print(f"poly_decay:\n{poly_decay(np.array([[1.0, 1.0] for i in range(10)]), metadata)}\n")



