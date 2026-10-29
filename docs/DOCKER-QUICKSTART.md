# Docker 快速部署指南

## 一键部署（推荐）

### 1. 构建镜像

```bash
cd /path/to/MNEMOSYNE-MNS
docker build -t mnemosyne-mns:latest .
```

构建时间：约 5-10 分钟（首次会下载 E5 模型）

### 2. 运行容器

```bash
docker run -d \
  --name mnemosyne \
  -p 8000:8000 \
  --restart unless-stopped \
  mnemosyne-mns:latest
```

### 3. 测试

```bash
# 健康检查
curl http://localhost:8000/health

# 检索测试
curl -X POST http://localhost:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{
    "query": "如何配置代理？",
    "memories": [
      {"id": "m1", "content": "在 git config 中设置 http.proxy 和 https.proxy"},
      {"id": "m2", "content": "使用 SSH 密钥连接远程服务器"},
      {"id": "m3", "content": "配置环境变量 HTTP_PROXY"}
    ],
    "top_k": 5
  }'
```

### 4. 查看日志

```bash
docker logs -f mnemosyne
```

---

## API 文档

启动后访问：http://localhost:8000/docs

### 端点列表

| 端点 | 方法 | 描述 |
|---|---|---|
| `/health` | GET | 健康检查 |
| `/retrieve` | POST | 检索并排序记忆 |
| `/detect_citations` | POST | 检测引用 |
| `/models` | GET | 列出可用模型 |
| `/docs` | GET | Swagger 文档 |

### 示例调用

**Python:**
```python
import requests

response = requests.post("http://localhost:8000/retrieve", json={
    "query": "如何配置代理？",
    "memories": [
        {"id": "m1", "content": "设置 http.proxy"},
        {"id": "m2", "content": "使用 SSH"}
    ],
    "top_k": 5,
    "scope_level": "global"
})

results = response.json()["results"]
for r in results:
    print(f"Rank {r['rank']}: {r['memory_id']} (score: {r['score']:.4f})")
```

**JavaScript:**
```javascript
const response = await fetch('http://localhost:8000/retrieve', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    query: "如何配置代理？",
    memories: [
      {id: "m1", content: "设置 http.proxy"},
      {id: "m2", content: "使用 SSH"}
    ],
    top_k: 5
  })
});

const data = await response.json();
console.log(data.results);
```

**cURL:**
```bash
curl -X POST http://localhost:8000/retrieve \
  -H "Content-Type: application/json" \
  -d @request.json
```

---

## 高级配置

### 使用 Docker Compose

创建 `docker-compose.yml`：

```yaml
version: '3.8'

services:
  mnemosyne:
    image: mnemosyne-mns:latest
    container_name: mnemosyne
    ports:
      - "8000:8000"
    environment:
      - WORKERS=2
      - LOG_LEVEL=info
    volumes:
      - ./shadow_traces:/app/shadow_traces
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

启动：
```bash
docker-compose up -d
```

### GPU 支持

如果有 GPU：

```dockerfile
# 修改 Dockerfile 基础镜像
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# 安装 Python
RUN apt-get update && apt-get install -y python3.10 python3-pip

# 其余同原 Dockerfile
```

运行：
```bash
docker run --gpus all -p 8000:8000 mnemosyne-mns:latest
```

### 环境变量

| 变量 | 默认值 | 描述 |
|---|---|---|
| `WORKERS` | 1 | Uvicorn worker 数量 |
| `LOG_LEVEL` | info | 日志级别 |
| `MAX_TOKENS` | 4000 | 上下文 token 预算 |

---

## 集成到 Hermes Agent

### 方案 1：直接调用

```python
import requests

class HermesMemoryClient:
    def __init__(self, url="http://localhost:8000"):
        self.url = url
    
    def retrieve(self, query: str, memories: list) -> list:
        response = requests.post(f"{self.url}/retrieve", json={
            "query": query,
            "memories": [{"id": m.id, "content": m.content} for m in memories],
            "top_k": 10
        })
        return response.json()["results"]

# 在 Hermes Agent 中使用
memory_client = HermesMemoryClient()
results = memory_client.retrieve(user_query, memory_pool)
```

### 方案 2：作为 Hermes 插件

```python
# hermes_plugins/mnemosyne_plugin.py
class MnemosynePlugin:
    def __init__(self):
        self.client = requests.Session()
        self.base_url = "http://localhost:8000"
    
    def on_query(self, context):
        """Hermes 查询钩子"""
        memories = context.get_candidate_memories()
        
        results = self.client.post(f"{self.base_url}/retrieve", json={
            "query": context.query,
            "memories": [{"id": m.id, "content": m.content} for m in memories]
        }).json()["results"]
        
        # 注入到上下文
        context.inject_memories(results)
```

---

## 生产部署

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mnemosyne
spec:
  replicas: 3
  selector:
    matchLabels:
      app: mnemosyne
  template:
    metadata:
      labels:
        app: mnemosyne
    spec:
      containers:
      - name: mnemosyne
        image: mnemosyne-mns:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
---
apiVersion: v1
kind: Service
metadata:
  name: mnemosyne
spec:
  selector:
    app: mnemosyne
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

### 监控

添加 Prometheus metrics：

```python
# 在 server.py 中添加
from prometheus_client import Counter, Histogram, make_asgi_app

request_count = Counter('mnemosyne_requests_total', 'Total requests')
request_latency = Histogram('mnemosyne_request_latency_seconds', 'Request latency')

# 挂载 metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)
```

---

## 故障排查

### 容器无法启动

```bash
# 查看日志
docker logs mnemosyne

# 进入容器调试
docker exec -it mnemosyne bash
python3 -c "from memory_pipeline import create_pipeline; create_pipeline('hybrid')"
```

### 性能优化

```bash
# 增加 workers
docker run -e WORKERS=4 -p 8000:8000 mnemosyne-mns:latest

# 使用 GPU
docker run --gpus all -p 8000:8000 mnemosyne-mns:latest
```

### 内存不足

```bash
# 限制内存
docker run -m 4g -p 8000:8000 mnemosyne-mns:latest
```

---

## 更新部署

```bash
# 拉取最新代码
git pull origin main

# 重新构建镜像
docker build -t mnemosyne-mns:latest .

# 重启容器
docker stop mnemosyne
docker rm mnemosyne
docker run -d --name mnemosyne -p 8000:8000 mnemosyne-mns:latest
```

---

## 镜像分发

### 推送到 Docker Hub

```bash
docker tag mnemosyne-mns:latest your-username/mnemosyne-mns:1.0
docker push your-username/mnemosyne-mns:1.0
```

### 用户拉取

```bash
docker pull your-username/mnemosyne-mns:1.0
docker run -d -p 8000:8000 your-username/mnemosyne-mns:1.0
```

---

## 性能基准

| 配置 | 延迟 (P50) | 延迟 (P99) | 吞吐量 |
|---|---:|---:|---:|
| 1 CPU core | 800ms | 1500ms | ~1.2 req/s |
| 2 CPU cores | 600ms | 1000ms | ~2.5 req/s |
| 4 CPU cores | 400ms | 700ms | ~5 req/s |
| GPU (T4) | 250ms | 400ms | ~10 req/s |

测试条件：100 候选记忆，top_k=10

---

**下一步：**
1. 构建 Docker 镜像
2. 测试 API
3. 集成到 Hermes Agent
4. 启用 Shadow Mode 收集 traces
