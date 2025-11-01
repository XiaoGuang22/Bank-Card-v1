#!/bin/bash

# 在tmux会话中后台运行训练脚本，并将输出保存到日志文件

# 设置工作目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TRAIN_SCRIPT="$PROJECT_ROOT/crnn/train.py"

# 检查训练脚本是否存在
if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "❌ 错误：找不到训练脚本 $TRAIN_SCRIPT"
    exit 1
fi

# 创建日志目录
LOG_DIR="$PROJECT_ROOT/crnn/logs"
mkdir -p "$LOG_DIR"

# 生成日志文件名（带时间戳）
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/train_${TIMESTAMP}.log"

echo "🚀 准备在tmux会话中启动训练..."
echo "📝 日志文件: $LOG_FILE"

# conda环境名称
CONDA_ENV="yolo"

# tmux会话名称
SESSION_NAME="credit_card_ocr_train"

# 检查是否已存在同名会话
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "⚠️  tmux会话 '$SESSION_NAME' 已存在，自动终止..."
    tmux kill-session -t "$SESSION_NAME"
fi

# 创建新的tmux会话并在后台运行
echo "创建tmux会话: $SESSION_NAME"
echo "使用conda环境: $CONDA_ENV"
tmux new-session -d -s "$SESSION_NAME" -c "$PROJECT_ROOT/crnn" "bash -c 'source /home/liangpeng/anaconda3/etc/profile.d/conda.sh && conda activate $CONDA_ENV && python3 -u train.py 2>&1 | tee -a \"$LOG_FILE\"'"

# 等待一下确保会话启动
sleep 2

# 检查会话是否还在运行
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "✅ 训练已启动！"
    echo ""
    echo "📊 会话信息："
    echo "  - 会话名称: $SESSION_NAME"
    echo "  - 日志文件: $LOG_FILE"
    echo ""
    echo "💡 常用命令："
    echo "  - 查看会话: tmux list-sessions"
    echo "  - 附加会话: tmux attach -t $SESSION_NAME"
    echo "  - 分离会话: 在tmux会话中按 Ctrl+B, 然后按 D"
    echo "  - 终止会话: tmux kill-session -t $SESSION_NAME"
    echo "  - 实时查看日志: tail -f $LOG_FILE"
    echo ""
    echo "🔍 当前会话状态:"
    tmux list-sessions | grep "$SESSION_NAME"
else
    echo "❌ 训练启动失败"
    exit 1
fi

