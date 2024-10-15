from src.simulate.protocol import ReqResponse
from src.analysis.report import RequestLevelReport
from src.analysis.generate_report import generate_request_level_report
from typing import List, Tuple
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import bisect
import io

def save_plot_as_webp(plt, path: str):
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    img.save(path, 'WEBP')
    buf.close()

def RequestsStatus(req_ress: List[ReqResponse], path: str):
    plt.clf()
    start_timestamp = req_ress[0].start_timestamp
    end_timestamp = max([v.end_timestamp for v in req_ress])
    interval = 1
    x = np.arange(0, end_timestamp - start_timestamp, interval)
    issued_delta = np.zeros(len(x))
    success_delta = np.zeros(len(x))
    fail_delta = np.zeros(len(x))
    for r in req_ress:
        issued_delta[int((r.start_timestamp - start_timestamp) / interval)] += 1
        if r.error_info is not None:
            fail_delta[int((r.end_timestamp - start_timestamp) / interval)] += 1
        else:
            success_delta[int((r.end_timestamp - start_timestamp) / interval)] += 1

    def sumup(l):
        ret = []
        for i in l:
            if len(ret) == 0:
                ret.append(i)
            else:
                ret.append(ret[-1] + i)
        return np.array(ret)

    issued = sumup(issued_delta)
    success = sumup(success_delta)
    fail = sumup(fail_delta)
    reqs = {
        "Fail": fail,
        "Success": success,
        "In progress": issued - success - fail,
    }
    _, ax = plt.subplots()
    ax.stackplot(
        x,
        reqs.values(),
        labels=reqs.keys(),
        alpha=0.95,
        colors=["#fc4f30", "#6d904f", "#008fd5"],
    )
    ax.legend(loc="upper left", reverse=True)
    ax.set_title("Request status over time")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("number of requests")
    #plt.savefig(path)
    save_plot_as_webp(plt, path.replace('.png', '.webp'))

def Throughput(report: RequestLevelReport, path: str, **kwargs):
    plt.clf()
    data: List[Tuple[float, int]] = report.token_timestamp
    start = data[0][0]
    end = data[-1][0]
    time_step = kwargs.get("time_step", 0.05)
    window_size = kwargs.get("window_size", 5)
    x = np.arange(0, end - start, time_step)
    y = np.zeros(len(x))
    for i in range(len(y)):
        ty = start + i * time_step
        y[i] = bisect.bisect_right(
            data, ty + window_size / 2, key=lambda x: x[0]
        ) - bisect.bisect_left(data, ty - window_size / 2, key=lambda x: x[0])
    y = y / window_size
    plt.plot(x, y)
    plt.title("Tokens processed over time")
    plt.xlabel("time (s)")
    plt.ylabel("number of tokens")
    #plt.savefig(path)
    save_plot_as_webp(plt, path.replace('.png', '.webp'))


if __name__ == "__main__":
    import pickle

    name = "ecac3861-8c5e-40d7-8a18-e7c8ef24aa99"
    loaded: List[ReqResponse] = sum(
        [v.responses for v in pickle.load(open(f"tmp/responses_{name}.pkl", "rb"))],
        [],
    )
    RequestsStatus(loaded, f"tmp/rs_{name}.webp")
    Throughput(
        generate_request_level_report(loaded, "meta-llama/Llama-2-7b-chat-hf"),
        f"tmp/tp_{name}.webp",
    )
