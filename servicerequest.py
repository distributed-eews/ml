import os
from typing import Tuple
import json

import numpy as np
import requests
import pickle
import matplotlib.pyplot as plt
import datetime

SERVICE_URL_P = "http://localhost:3000/predict_p"
SERVICE_URL_S = "http://localhost:3000/predict_s"


def make_request_to_bento_service(service_url: str, input_array: np.ndarray, metadata: dict, reset: bool|None = None) -> str:
    input_data = {'x': input_array.tolist(), 'metadata': metadata}
    if reset is not None:
        input_data['reset'] = True
    serialized_input_data = json.dumps(input_data)
    response = requests.post(
        service_url,
        data=serialized_input_data,
        headers={"content-type": "application/json"}
    )
    return json.loads(response.text)


def main():

    for filename in os.listdir("./sample_data"):
        with open(f"./sample_data/{filename}", "rb") as f:
            earthquake = np.load(f)

            # Misalkan setiap satu detik, service lain ngirim 5000 datapoints (bisa aja 20, tapi ini exagerrated)
            # for i in range(0, len(earthquake), 5000):
            #     prediction = np.array(make_request_to_bento_service(
            #         SERVICE_URL_P,
            #         earthquake[i:i + 5000],
            #         {'f': 20.0, 'time': str(datetime.datetime.now())}))

            prediction_p = np.array(make_request_to_bento_service(
                SERVICE_URL_P,
                earthquake,
                {'f': 20.0, 'time': str(datetime.datetime.now())}
            ))

            prediction_s = np.array(make_request_to_bento_service(
                SERVICE_URL_S,
                earthquake,
                {'f': 20.0, 'time': str(datetime.datetime.now())},
                reset=True
            ))

            print(prediction_s)

            fig, axs = plt.subplots(3)
            axs[0].plot(earthquake); axs[0].set_title(f"Gempa {filename}")
            axs[1].plot(prediction_p); axs[1].set_title("p_prediction")
            axs[2].plot(prediction_s); axs[2].set_title("s_prediction")
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    main()
