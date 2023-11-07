from pathlib import Path
import numpy as np

class Pipeline:
    def __init__(self, path: Path):
        self._states: dict = {}
        self._pipeline: list = []
        self._path: Path = path

        # Build pipeline
        self._build_pipeline()

    def _build_pipeline(self):
        self._pipeline = []
        with open(self._path) as f:
            for line in f:
                self._pipeline.append(eval("self." + line))

    def reset(self):
        self._build_pipeline()

    def convert_to_velocity(self, name:str ='c2v'):
        self._states[name] = 0

        def _convert_to_velocity(x: np.ndarray, metadata: dict):
            velocity = []
            for val in x:
                velocity.append((val - self._states[name]) * metadata['f'])
                self._states[name] = val
            return np.array(velocity)
        return _convert_to_velocity

    def exponential_smoothing(self, alpha: np.float32, name: str ='expsmth'):
        # Cache value
        alpha_ = 1 - alpha
        # Set initial statue
        self._states[name] = 0

        def _exponential_smoothing(x: np.ndarray, metadata: dict):
            smoothed = []
            for val in x:
                old_ema = self._states[name]
                new_ema = old_ema * alpha + val * alpha_
                self._states[name] = new_ema
                smoothed.append(new_ema)
            return np.array(smoothed)
        return _exponential_smoothing

    def multi_exponential_smoothing(self, alphas: np.ndarray, name:str ='mul_expsmth'):  # expect 1 dimensional x
        # Cache values
        alphas = np.array(alphas)
        alphas_ = 1 - alphas
        # Set initial state
        self._states[name] = np.zeros(len(alphas))

        def _multi_exponential_smoothing(x: np.ndarray, metadata: dict):  # Expected dimension = 1
            # TODO : Implement this for multichannel
            emas = []
            for val in x:
                old_ema = self._states[name]
                new_ema = old_ema * alphas + val * alphas_
                self._states[name] = new_ema
                emas.append(new_ema.flatten())
            return np.array(emas)
        return _multi_exponential_smoothing

    def square(self, name:str ='square'):
        def _square(x: np.ndarray, metadata: dict):
            return x*x
        return _square

    def add_channels(self, name:str ='add_channels'):
        def _add_channels(x: np.ndarray, metadata: dict):
            return np.sum(x, axis=-1, keepdims=True)
        return _add_channels

    def pairwise_ratio(self, alpha: np.ndarray, name:str ='pairwise_ratio'):
        alpha = np.array(alpha)
        # Generate Order
        idx_area = [(i, -np.log(1 - a) * (a) / (1 - a)) for i, a in enumerate(alpha)]
        triplets = [(j, i, idx_area[i][1] / idx_area[j][1]) for i in range(len(alpha)) for j in range(len(alpha)) if
                    i > j]
        triplets = sorted(triplets, key=lambda t: t[2])
        order = [(t[0], t[1]) for t in triplets]

        def _pairwise_ratio(x: np.ndarray, metadata: dict):
            new_x = []
            for i, j in order:
                new_x.append((x[:,i]+1)/(x[:,j]+1))
            return np.array(new_x).T
        return _pairwise_ratio

    def poly_decay(self, k: float, p: float, name='poly_decay'):
        self._states[name] = 0  # time

        def _poly_decay(x: np.ndarray,  metadata: dict):
            decayed = []
            for val in x:
                time = self._states[name]
                self._states[name] += 1/metadata['f']
                decayed.append(val * (1 - 1/((time/k)**(2*p)+1)))
            return np.array(decayed)
        return _poly_decay

    def process(self, x: np.ndarray, metadata: dict):  # Generator state machine
        for pipeline_function in self._pipeline:
            x = pipeline_function(x, metadata)
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



