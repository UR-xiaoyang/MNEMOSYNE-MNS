#!/bin/bash
# 远程训练进度监控脚本

REMOTE_HOST="jx@172.29.2.51"
LOG_FILE="~/logs/rank_hard_negatives.log"
CHECK_INTERVAL=30  # 每30秒检查一次

echo "=== MNEMOSYNE RANK 训练监控 ==="
echo "远程服务器: $REMOTE_HOST"
echo "日志文件: $LOG_FILE"
echo "检查间隔: ${CHECK_INTERVAL}秒"
echo ""

# 获取训练进程状态
check_process() {
    ssh $REMOTE_HOST "ps aux | grep train_rank_enhanced_v2.py | grep -v grep"
}

# 解析日志获取进度
parse_progress() {
    LOG_CONTENT=$(ssh $REMOTE_HOST "tail -100 $LOG_FILE 2>/dev/null")
    
    # 检测各阶段
    if echo "$LOG_CONTENT" | grep -q "Loading training data"; then
        echo "📥 数据加载阶段"
    fi
    
    if echo "$LOG_CONTENT" | grep -q "Loading E5 model"; then
        echo "🔧 E5 模型加载"
    fi
    
    if echo "$LOG_CONTENT" | grep -q "Encoding training queries"; then
        echo "🔄 正在编码训练查询..."
        # 尝试提取进度条
        PROGRESS=$(echo "$LOG_CONTENT" | grep -oP '\d+%' | tail -1)
        if [ -n "$PROGRESS" ]; then
            echo "   进度: $PROGRESS"
        fi
    fi
    
    if echo "$LOG_CONTENT" | grep -q "Encoding validation queries"; then
        echo "🔄 正在编码验证查询..."
    fi
    
    if echo "$LOG_CONTENT" | grep -q "Building features"; then
        echo "⚙️  特征工程阶段"
    fi
    
    if echo "$LOG_CONTENT" | grep -q "Training LightGBM"; then
        echo "🎯 LightGBM 训练中..."
        # 提取训练轮次
        ROUNDS=$(echo "$LOG_CONTENT" | grep -oP '\[\d+\]' | tail -1)
        if [ -n "$ROUNDS" ]; then
            echo "   当前轮次: $ROUNDS"
        fi
    fi
    
    if echo "$LOG_CONTENT" | grep -q "Offline Gate"; then
        echo "✅ Offline Gate 验证"
    fi
    
    if echo "$LOG_CONTENT" | grep -q "Training completed"; then
        echo "🎉 训练完成！"
        return 0
    fi
    
    # 检查错误
    if echo "$LOG_CONTENT" | grep -qiE "error|exception|failed"; then
        echo "❌ 检测到错误，查看日志:"
        echo "$LOG_CONTENT" | grep -iE "error|exception|failed" | tail -5
        return 1
    fi
    
    return 2
}

# 主监控循环
echo "开始监控..."
echo ""

while true; do
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$TIMESTAMP]"
    
    # 检查进程
    PROCESS_INFO=$(check_process)
    if [ -z "$PROCESS_INFO" ]; then
        echo "⚠️  训练进程未运行"
        echo ""
        echo "最后50行日志:"
        ssh $REMOTE_HOST "tail -50 $LOG_FILE 2>/dev/null"
        break
    else
        CPU_USAGE=$(echo "$PROCESS_INFO" | awk '{print $3}')
        MEM_USAGE=$(echo "$PROCESS_INFO" | awk '{print $4}')
        echo "✓ 进程运行中 (CPU: ${CPU_USAGE}%, MEM: ${MEM_USAGE}%)"
    fi
    
    # 解析进度
    parse_progress
    STATUS=$?
    
    if [ $STATUS -eq 0 ]; then
        echo ""
        echo "训练已完成！"
        break
    elif [ $STATUS -eq 1 ]; then
        echo ""
        echo "检测到错误，停止监控"
        break
    fi
    
    echo "---"
    echo ""
    
    sleep $CHECK_INTERVAL
done

echo ""
echo "监控结束"
