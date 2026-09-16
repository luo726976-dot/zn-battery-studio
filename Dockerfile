# ==============================================================================
# Dockerfile - ZnBattery Stacking & Bayesian Studio 生产级轻量化镜像
# 兼容: Hugging Face Spaces (UID 1000) / Streamlit Cloud / 本地私有化容器部署
# ==============================================================================

# 1. 采用 Python 3.10 官方轻量镜像 (Debian Bookworm)
FROM python:3.10-slim

# 2. 设置全局非交互与 Python 运行时环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# 3. 安装 RDKit、PyTorch 与科学绘图所需的系统级 C/C++ 共享库
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libxrender1 \
    libxext6 \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 4. 创建符合 Hugging Face Spaces 规范的非 root 用户 (UID 1000)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# 5. 设定工作目录并授权
WORKDIR $HOME/app

# 6. 利用 Docker 缓存层安装 Python 依赖
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# 7. 拷贝完整应用代码、基准数据库与 Streamlit 配置文件
COPY --chown=user . $HOME/app

# 8. 暴露端口 (Hugging Face Spaces 默认 7860，标准 Streamlit 默认 8501)
EXPOSE 7860 8501

# 9. 容器健康检查 (Healthcheck)
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:7860/_stcore/health || curl -f http://localhost:8501/_stcore/health || exit 1

# 10. 启动指令 (默认绑定 7860 端口适配 HF Spaces，亦可通过参数覆盖)
ENTRYPOINT ["streamlit", "run", "app.py"]
CMD ["--server.port=7860", "--server.address=0.0.0.0"]
