# 03차시: Ollama HTTP API 사용
import json
from urllib.request import Request, urlopen

payload = {"model": "qwen2.5:0.5b", "prompt": "UART가 무엇인지 두 문장으로 설명해줘.",
           "stream": False, "options": {"temperature": 0}}
request = Request("http://127.0.0.1:11434/api/generate",
                  data=json.dumps(payload).encode(),
                  headers={"Content-Type": "application/json"}, method="POST")
with urlopen(request, timeout=120) as response:
    result = json.load(response)
print("model:", result.get("model"))
print("answer:", result.get("response"))
print("tokens:", result.get("eval_count"))