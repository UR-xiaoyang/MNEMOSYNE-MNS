# C++ 部署指南

## 方案 A：Docker 容器化（推荐 ✅）

### 优点
- 环境完全隔离，无依赖冲突
- 跨平台，易于分发
- 不需要重写代码

### 构建步骤

**1. 创建 Dockerfile**

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# 安装依赖
RUN pip install --no-cache-dir torch transformers sklearn numpy joblib fastapi uvicorn

# 复制代码
COPY scripts/ ./scripts/
COPY runs/rank_hybrid_v1/ ./runs/rank_hybrid_v1/

# 预下载 E5 模型到容器
RUN python3 -c "from transformers import AutoModel, AutoTokenizer; \
    AutoModel.from_pretrained('intfloat/multilingual-e5-small'); \
    AutoTokenizer.from_pretrained('intfloat/multilingual-e5-small')"

# API 服务
COPY docker/server.py .
EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
```

**2. 创建 API 服务**

```python
# docker/server.py
from fastapi import FastAPI
from pydantic import BaseModel
import sys
sys.path.insert(0, "/app/scripts")

from memory_pipeline import create_pipeline
from memory_schema import synthesize_native_metadata_for_alpsbench

app = FastAPI()
pipeline = create_pipeline("hybrid")

class RetrievalRequest(BaseModel):
    query: str
    memories: list[dict]  # [{"id": "m1", "content": "..."}]
    top_k: int = 10

@app.post("/retrieve")
def retrieve(req: RetrievalRequest):
    candidates = [
        synthesize_native_metadata_for_alpsbench(m["content"], i, "api")
        for i, m in enumerate(req.memories)
    ]
    
    results = pipeline.retrieve_and_rank(
        query_text=req.query,
        candidate_memories=candidates,
        top_k=req.top_k
    )
    
    return {
        "results": [{
            "memory_id": req.memories[i]["id"],
            "rank": r["rank"],
            "score": r["score"]
        } for i, r in enumerate(results)]
    }

@app.get("/health")
def health():
    return {"status": "ok"}
```

**3. 构建镜像**

```bash
cd /path/to/MNEMOSYNE-MNS
docker build -t mnemosyne-mns:1.0 .
```

**4. 运行容器**

```bash
# 启动服务
docker run -d -p 8000:8000 --name mnemosyne mnemosyne-mns:1.0

# 测试
curl -X POST http://localhost:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{
    "query": "如何配置代理？",
    "memories": [
      {"id": "m1", "content": "在 git config 中设置 http.proxy"},
      {"id": "m2", "content": "使用 SSH 密钥"}
    ],
    "top_k": 5
  }'
```

**5. 客户端调用（任何语言）**

Python:
```python
import requests
response = requests.post("http://localhost:8000/retrieve", json={...})
```

C++:
```cpp
#include <curl/curl.h>
// 使用 libcurl 调用 HTTP API
```

Go:
```go
import "net/http"
// 使用标准库调用
```

---

## 方案 B：C++ + ONNX Runtime（高性能）

### 优点
- 完全独立，无 Python 依赖
- 性能最优
- 适合嵌入式部署

### 实现步骤

**1. 导出模型为 ONNX**

```bash
cd /path/to/MNEMOSYNE-MNS
python3 scripts/export_to_onnx.py
# 生成: runs/onnx_export/e5_encoder.onnx, scaler.json, ranker.json
```

**2. 安装 ONNX Runtime**

```bash
# Ubuntu/Debian
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.0/onnxruntime-linux-x64-1.16.0.tgz
tar -xzf onnxruntime-linux-x64-1.16.0.tgz
sudo cp -r onnxruntime-linux-x64-1.16.0/include/* /usr/local/include/
sudo cp -r onnxruntime-linux-x64-1.16.0/lib/* /usr/local/lib/
sudo ldconfig

# 安装 JSON 库
sudo apt-get install nlohmann-json3-dev
```

**3. 编译 C++ 推理代码**

```bash
cd /path/to/MNEMOSYNE-MNS/cpp
g++ -std=c++17 mnemosyne_inference.cpp \
  -I/usr/local/include \
  -L/usr/local/lib \
  -lonnxruntime \
  -o mnemosyne_inference

# 运行
./mnemosyne_inference
```

**4. 集成到项目**

```cpp
#include "mnemosyne_inference.h"

int main() {
    MnemosyneRanker ranker("./runs/onnx_export");
    
    std::vector<Memory> candidates = load_memories();
    auto results = ranker.rank_all(query, candidates);
    
    for (auto& r : results) {
        std::cout << "Memory " << r.id << ": " << r.score << std::endl;
    }
}
```

### 注意事项

**C++ 方案的限制：**
1. **Tokenizer 复杂**：E5 使用 SentencePiece，需要额外集成
2. **特征工程**：需要手动实现 TF-IDF、Jaccard 等特征
3. **维护成本高**：模型更新需要重新编译

**建议：优先使用 Docker 方案，性能需求极高时才考虑 C++。**

---

## 方案 C：预编译可执行文件（最简单）

使用 PyInstaller 打包成单个可执行文件。

**1. 安装 PyInstaller**

```bash
pip install pyinstaller
```

**2. 创建启动脚本**

```python
# mnemosyne_server.py
import sys
from pathlib import Path

# 添加脚本路径
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from memory_pipeline import create_pipeline
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI()
pipeline = create_pipeline("hybrid")

class RetrievalRequest(BaseModel):
    query: str
    memories: list[dict]
    top_k: int = 10

@app.post("/retrieve")
def retrieve(req: RetrievalRequest):
    from memory_schema import synthesize_native_metadata_for_alpsbench
    
    candidates = [
        synthesize_native_metadata_for_alpsbench(m["content"], i, "api")
        for i, m in enumerate(req.memories)
    ]
    
    results = pipeline.retrieve_and_rank(
        query_text=req.query,
        candidate_memories=candidates,
        top_k=req.top_k
    )
    
    return {"results": results}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**3. 打包**

```bash
pyinstaller --onefile \
  --add-data "scripts:scripts" \
  --add-data "runs/rank_hybrid_v1:runs/rank_hybrid_v1" \
  --hidden-import torch \
  --hidden-import transformers \
  mnemosyne_server.py

# 生成: dist/mnemosyne_server.exe (Windows) 或 dist/mnemosyne_server (Linux)
```

**4. 分发**

```bash
# 将整个 dist/ 目录打包分发
zip -r mnemosyne-mns-standalone.zip dist/
```

**5. 用户使用**

```bash
# 解压
unzip mnemosyne-mns-standalone.zip

# 运行（无需安装 Python）
./dist/mnemosyne_server

# 调用
curl http://localhost:8000/retrieve -d '{...}'
```

---

## 方案对比

| 方案 | 环境依赖 | 性能 | 分发便利性 | 开发成本 |
|---|---|---|---|---|
| **Docker** | Docker | 中 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| C++ + ONNX | libonnxruntime | 高 | ⭐⭐⭐ | ⭐⭐ |
| PyInstaller | 无 | 中 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

---

## 推荐方案

**对于 Hermes Agent：**
- 如果用户已有 Docker → **Docker 容器化**
- 如果需要单文件分发 → **PyInstaller**
- 如果性能是瓶颈 → **C++ + ONNX**

**实施优先级：**
1. ✅ **Docker**（1-2 小时实现）
2. ✅ **PyInstaller**（2-3 小时实现）
3. ⏳ **C++ ONNX**（1-2 天实现，需要更多测试）

是否需要我实现完整的 Docker 或 PyInstaller 方案？
