# smart_assistant_agent

Python Agent 编排服务。

## 能力范围（V1）
- 会话短期记忆（进程内，后续可切 Redis）
- 文本意图识别（规则版）
- 工具调用编排（转发到 HA Bridge）
- 降级回复（桥接不可用时也能返回结果）

## 技术栈
- FastAPI
- Uvicorn
- httpx

## 本地运行
```bash
python -m venv .venv
. .venv/Scripts/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8091 --reload
```

## 环境变量
参考 `.env.example`。

关键变量：
- `HA_BRIDGE_URL`：HA Bridge 服务地址
- `AGENT_MEMORY_MAX_TURNS`：每个会话保留多少条历史
- `AGENT_TOOL_AUTO_EXECUTE`：是否自动执行工具调用

## 主要接口
- `GET /health`
- `POST /v1/agent/respond`
- `GET /v1/agent/session/{session_id}`
- `DELETE /v1/agent/session/{session_id}`

## Docker
```bash
docker build -t smart-assistant-agent .
docker run --rm -p 8091:8091 --env-file .env.example smart-assistant-agent
```

