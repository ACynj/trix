#!/bin/bash

# 脚本：运行 command.md 中的所有命令
# 确保每个命令都正确执行完毕

set -o pipefail  # 管道命令中任何一个失败都会导致整个管道失败

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 日志文件
LOG_FILE="run_commands_$(date +%Y%m%d_%H%M%S).log"
FAILED_COMMANDS="failed_commands_$(date +%Y%m%d_%H%M%S).txt"

# 统计变量
TOTAL=0
SUCCESS=0
FAILED=0

# 打印带颜色的消息
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

# 检查虚拟环境
if ! command -v conda &> /dev/null; then
    print_error "conda 未找到，请确保已安装 conda"
    exit 1
fi

# 激活虚拟环境
print_info "正在激活虚拟环境: trix"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate trix || {
    print_error "无法激活虚拟环境 trix"
    exit 1
}
print_info "虚拟环境 trix 已激活"

# 切换到项目目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || {
    print_error "无法切换到项目目录: $SCRIPT_DIR"
    exit 1
}
print_info "当前工作目录: $(pwd)"

# 读取命令文件
COMMAND_FILE="command_rel.md"
if [ ! -f "$COMMAND_FILE" ]; then
    print_error "找不到命令文件: $COMMAND_FILE"
    exit 1
fi

print_info "开始读取命令文件: $COMMAND_FILE"
print_info "日志将保存到: $LOG_FILE"
echo ""

# 执行命令函数
execute_command() {
    local cmd="$1"
    local cmd_num=$2
    
    if [ -z "$cmd" ] || [[ "$cmd" =~ ^[[:space:]]*$ ]] || [[ "$cmd" =~ ^=+$ ]]; then
        return 0  # 跳过空行和分隔符
    fi
    
    TOTAL=$((TOTAL + 1))
    
    print_info "=========================================="
    print_info "执行命令 $cmd_num: $cmd"
    print_info "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
    
    # 执行命令并捕获输出
    if eval "$cmd" >> "$LOG_FILE" 2>&1; then
        SUCCESS=$((SUCCESS + 1))
        print_info "✓ 命令 $cmd_num 执行成功"
        print_info "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
        echo ""
        return 0
    else
        FAILED=$((FAILED + 1))
        local exit_code=$?
        print_error "✗ 命令 $cmd_num 执行失败 (退出码: $exit_code)"
        print_error "失败的命令: $cmd"
        echo "$cmd" >> "$FAILED_COMMANDS"
        print_info "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
        echo ""
        
        # 询问是否继续
        read -p "是否继续执行下一个命令? (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_warning "用户选择停止执行"
            return 1
        fi
        return 0
    fi
}

# 读取并执行命令
cmd_num=0
while IFS= read -r line || [ -n "$line" ]; do
    # 跳过空行和分隔符行
    if [[ -z "$line" ]] || [[ "$line" =~ ^[[:space:]]*$ ]] || [[ "$line" =~ ^=+$ ]]; then
        continue
    fi
    
    # 只处理以 python 开头的命令
    if [[ "$line" =~ ^python[[:space:]] ]]; then
        cmd_num=$((cmd_num + 1))
        if ! execute_command "$line" "$cmd_num"; then
            print_error "脚本因用户选择而停止"
            break
        fi
    fi
done < "$COMMAND_FILE"

# 打印统计信息
print_info "=========================================="
print_info "执行完成！"
print_info "总命令数: $TOTAL"
print_info "成功: $SUCCESS"
print_info "失败: $FAILED"
print_info "日志文件: $LOG_FILE"

if [ $FAILED -gt 0 ]; then
    print_error "失败的命令列表已保存到: $FAILED_COMMANDS"
    exit 1
else
    print_info "所有命令执行成功！"
    exit 0
fi

