# LLM-Serving-benchmark
A benchmark framework for LLM serving performance, based on API call

#### 시작하기 전에
이 프로젝트는 Python 3을 기반으로 하므로, Python 3가 설치되어 있어야 합니다.
필요하다면 Python3 가상환경을 생성하세요.
    
**e.g**
```bash
sudo apt update && sudo apt upgrade -y 
sudo apt-get install python3.10 python3.10-venv
virtualenv venv
source venv/bin/activate
```

#### 실행 과정

1. **Python3 패키지 설치**
   ```bash
   python3 -m pip install -r requirements.txt
   ```

2. **tmp 폴더 생성**

   프로젝트의 일시적인 데이터를 저장할 임시 폴더를 생성합니다. (base 폴더 내에 생성)
   ```bash
   mkdir tmp
   ```

3. **API 서버 실행**

   프로젝트의 메인 API 서버를 실행합니다. 이 서버는 Config 등록 및 테스트 관리를 담당합니다.
   ```bash
   python3 -m src.api_server.app
   ```
   이 명령은 `src/api_server/app.py` 파일을 실행합니다.

4. **Worker 실행**

   백그라운드에서 작업을 처리할 worker를 실행합니다.
   ```bash
   python3 -m src.api_server.worker
   ```
   이 명령은 `src/api_server/worker.py` 파일을 실행하여, 등록된 테스트 요청을 처리합니다.

   현재 여러개의 worker를 통한 병렬 작업 처리는 지원하지 않습니다.

5. **테스트 등록**

   curl을 사용하여 API 서버에 테스트를 등록합니다. 아래는 예시로 제공된 JSON 본문을 포함한 POST 요청입니다.
   ```bash
   curl -X POST localhost:8000/register_test \
        -H "Content-Type: application/json" \
        -d '{
               "url": "http://10.0.0.42:8000/v1",
               "model": "mistralai/Mistral-7B-Instruct-v0.1",
               "dataset_name": "synthesizer",
               "endpoint_type": "vllm",
               "dataset_config": {
                   "func": "lambda t: int(t / 0.1 + 1) if t < 60 else None",
                   "random_seed": 1234,
                   "prompt_source": "arena"
               },
               "kwargs": {
                   "temperature": 0.9,
                   "top_p": 1,
                   "max_tokens": 512,
                   "skip_idle_min": 20,
                   "time_step": 0.01,
                   "request_timeout": 3600,
                   "hf_auth_key": "hf_1234567890qwertyuiop"
               }
            }'
   ```
   이 요청은 서버에 새로운 테스트를 등록하며, 테스트 세부 사항은 JSON 본문에서 설정할 수 있습니다.

   dataset_config.func, dataset_config.random_seed를 통해 Dataset내에서도 일부만을 사용해 테스트하도록 설정할 수 있습니다.

   huggingface Access token이 필요한 경우(권한이 필요한 모델을 사용하는 경우), hf_auth_key를 통해 설정할 수 있습니다.
   
   등록에 성공한 경우 등록된 테스트의 ID를 반환합니다.
   ```bash
   "5fe3affd-f6ac-44cf-bbd6-f1d3661447ae"
   ```

6. **테스트 실행**
   등록된 테스트를 시작합니다. `{id}`는 등록된 테스트의 고유 ID입니다.
   ```bash
   curl 127.0.0.1:8000/start_test/{id}
   ```

7. **테스트 결과**
   등록된 테스트 요청을 처리하고 다음과 같은 형식의 결과를 JSON 파일로 저장합니다:

   report_{id}.json 파일 예시
   
   ```json
   {
       "request_num": 591,
       "fail_rate": 0.0,
       "TTFT": {
           "min": 0.06914210319519043,
           "max": 2.051903247833252,
           "avg": 0.19164396905657,
           "std": 0.18363592223892217
       },
       "SLO": 1.0,
       "TPOT": {
           "min": 0.0052838377900175995,
           "max": 0.10848593711853027,
           "avg": 0.025429586140997603,
           "std": 0.008789927535285968
       },
       "Throughput": 2342.2
   }
   ```
   request_num: 처리된 요청의 총 수
   fail_rate: 실패율
   TTFT (Time To First Byte): 첫 번째 바이트까지의 시간에 대한 통계
   SLO (Service Level Objective): 서비스 수준 목표 달성률
   TPOT (Time Per Output Token): 출력 토큰당 시간에 대한 통계
   Throughput: 초당 처리량
   결과 파일은 각 테스트 세션의 성능을 평가하는 데 사용됩니다.