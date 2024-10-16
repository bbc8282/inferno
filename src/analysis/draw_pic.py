from src.simulate.protocol import ReqResponse
from src.analysis.report import RequestLevelReport
from src.analysis.generate_report import generate_request_level_report
from typing import List, Tuple
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import bisect
import io

# Set the default figure size to 1280x960 pixels
plt.rcParams.update({'font.size': 16})
plt.rcParams['figure.figsize'] = [12, 9]
plt.rcParams['figure.dpi'] = 120

def save_plot_as_webp(plt, path: str):
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120)  # Increased DPI for better quality
    buf.seek(0)
    img = Image.open(buf)
    img = img.resize((1280, 960), Image.LANCZOS)
    img.save(path, 'WEBP', quality=100)  # High quality WEBP
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
    ax.legend(loc="upper left", reverse=True, fontsize=18)
    ax.set_title("Request status over time", fontsize=26, fontweight='bold')
    ax.set_xlabel("Time (s)", fontsize=20)
    ax.set_ylabel("Number of requests", fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    save_plot_as_webp(plt, path.replace('.png', '.webp'))
    #plt.savefig(path)

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
    plt.plot(x, y, linewidth=2)
    plt.title("Tokens processed over time", fontsize=26, fontweight='bold')
    plt.xlabel("Time (s)", fontsize=20)
    plt.ylabel("Number of tokens", fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    #plt.savefig(path)
    save_plot_as_webp(plt, path.replace('.png', '.webp'))


if __name__ == "__main__":
    import pickle

    name = "test12"
    loaded: List[ReqResponse] = sum(
        [v.responses for v in pickle.load(open(f"tmp/responses_{name}.pkl", "rb"))],
        [],
    )
    RequestsStatus(loaded, f"tmp/rs_{name}.webp")
    Throughput(
        generate_request_level_report(loaded, "mistralai/Mistral-7B-Instruct-v0.2"),
        f"tmp/tp_{name}.webp",
    )
