#!/bin/bash
# ==============================================================================
# deploy_to_hf.sh - Automated CI/CD Deployment Script for Hugging Face Spaces
# 生产级自动化部署脚本：防错机制、动态时间戳 Commit、彩色控制台输出
# ==============================================================================
set -e
set -o pipefail

# ANSI 终端转义色彩定义
COLOR_RESET="\033[0m"
COLOR_INFO="\033[1;34m"     # 粗体高亮蓝 (INFO)
COLOR_SUCCESS="\033[1;32m"  # 粗体高亮绿 (SUCCESS)
COLOR_WARN="\033[1;33m"     # 粗体高亮黄 (WARN)
COLOR_ERROR="\033[1;31m"    # 粗体高亮红 (ERROR)
COLOR_CYAN="\033[1;36m"     # 粗体高亮青 (PROMPT)

log_info() {
    echo -e "${COLOR_INFO}[INFO]${COLOR_RESET} $1"
}

log_success() {
    echo -e "${COLOR_SUCCESS}[SUCCESS]${COLOR_RESET} $1"
}

log_warn() {
    echo -e "${COLOR_WARN}[WARN]${COLOR_RESET} $1"
}

log_error() {
    echo -e "${COLOR_ERROR}[ERROR]${COLOR_RESET} $1" >&2
}

# 注册异常中断捕获器 (Fail-Fast 审计追踪)
trap 'log_error "自动化部署流水线中断！上一条指令执行异常退出，退出码: $?"' ERR

echo -e "${COLOR_CYAN}======================================================${COLOR_RESET}"
echo -e "${COLOR_CYAN}  🔬 ZnBattery Studio - Hugging Face 一键部署流水线    ${COLOR_RESET}"
echo -e "${COLOR_CYAN}======================================================${COLOR_RESET}"

# 1. 检查本地 Git 仓库状态，缺失则自动自愈初始化
if [ ! -d ".git" ]; then
    log_warn "未检测到本地 .git 目录，系统自愈启动: 执行 git init -b main..."
    git init -b main
    log_success "本地 Git 仓库初始化就绪 (主分支: main)。"
else
    log_info "本地 Git 仓库自检通过。"
fi

# 1.1 自动配置 Git 提交者身份（避免因未设置 user.name/email 导致 commit 中断）
if [ -z "$(git config user.name 2>/dev/null || true)" ]; then
    git config user.name "ZnBattery Researcher"
    log_info "已自动配置本地 Git 提交者: ZnBattery Researcher"
fi
if [ -z "$(git config user.email 2>/dev/null || true)" ]; then
    git config user.email "researcher@users.noreply.huggingface.co"
    log_info "已自动配置本地 Git 邮箱: researcher@users.noreply.huggingface.co"
fi

# 2. 检查远程仓库 'space' 是否已配置
if ! git remote | grep -q "^space$"; then
    log_error "未检测到名为 'space' 的远程 Hugging Face 仓库源！"
    echo -e "${COLOR_CYAN}请在终端运行以下指令完成一次性远程源绑定:${COLOR_RESET}"
    echo -e "  git remote add space https://<用户名>:<HF_Token>@huggingface.co/spaces/<用户名>/<Space名>"
    echo ""
    exit 1
fi

# 3. 动态时间戳日志生成 (支持接收命令行传参)
if [ -n "$*" ]; then
    COMMIT_MSG="$*"
    log_info "检测到用户自定义 Commit 信息: \"${COMMIT_MSG}\""
else
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    COMMIT_MSG="Auto-deployed via script on ${TIMESTAMP}"
    log_info "未传入提交说明，自动生成动态时间戳: \"${COMMIT_MSG}\""
fi

# 4. 执行暂存 (依据 .gitignore 过滤非必要大文件与敏感信息)
log_info "正在暂存核心生产环境文件 (git add .)..."
git add .
log_success "文件暂存完成。"

# 5. 安全提交 (检查暂存区变动，防止空提交报错)
if git diff --staged --quiet; then
    log_warn "工作区与暂存区无新增变动，无需执行本地 commit。"
else
    log_info "正在提交版本快照..."
    git commit -m "${COMMIT_MSG}"
    log_success "本地提交完成: ${COMMIT_MSG}"
fi

# 6. 推送至 Hugging Face Space 远程仓库 (强制覆盖以确保云端镜像与本地完全一致)
log_info "正在同步推送至 Hugging Face Space (分支: main)..."
git push -f space main
log_success "远端仓库同步成功！"

# 7. 部署终态反馈
echo ""
echo -e "${COLOR_SUCCESS}🚀 部署完成！请前往 Hugging Face Space 查看实时构建状态。${COLOR_RESET}"
echo ""
