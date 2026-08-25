# R0 基础镜像：slim 够用，沙箱镜像（python:3.12-slim）由 SANDBOX_IMAGE 单独管理
FROM python:3.12-slim

WORKDIR /app

# 先拷依赖清单再拷代码，利用 Docker 层缓存（依赖不变就不重装）
COPY pyproject.toml ./
COPY scholar_agent ./scholar_agent
# 国内网络加清华镜像，否则直连 PyPI 极慢
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple .

EXPOSE 8000

CMD ["uvicorn", "scholar_agent.main:app", "--host", "0.0.0.0", "--port", "8000"]
