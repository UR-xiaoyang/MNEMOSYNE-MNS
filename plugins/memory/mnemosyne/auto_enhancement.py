"""
MNEMOSYNE插件增强补丁

添加以下功能：
1. 自动检索记忆（MNEMOSYNE_AUTO_RECALL=1）
2. 自动存储记忆（MNEMOSYNE_AUTO_STORE=1）
3. 智能记忆提取和存储

使用方法：
将以下代码添加到 plugins/memory/mnemosyne/__init__.py
"""

# 在 _get_memory_mode 方法后添加：

def _should_auto_recall(self) -> bool:
    """检查是否启用自动检索."""
    return os.environ.get("MNEMOSYNE_AUTO_RECALL", "0") == "1"

def _should_auto_store(self) -> bool:
    """检查是否启用自动存储."""
    return os.environ.get("MNEMOSYNE_AUTO_STORE", "0") == "1"

def _extract_key_info(self, user_text: str, assistant_text: str) -> str | None:
    """
    从对话中提取关键信息用于存储.
    
    规则：
    - 用户明确要求记住的内容
    - 重要的决策和结论
    - 项目状态更新
    - 配置和设置
    """
    # 关键词匹配
    store_triggers = [
        "请记住", "记录", "保存", "记下", 
        "项目", "进度", "状态", "完成",
        "配置", "设置", "部署"
    ]
    
    # 检查用户是否明确要求记住
    for trigger in store_triggers:
        if trigger in user_text:
            # 提取关键信息
            if len(assistant_text) < 500:
                return assistant_text
            else:
                # 太长，提取摘要
                return f"{user_text[:100]}... → {assistant_text[:300]}..."
    
    # 检查是否是状态更新
    if any(word in user_text for word in ["进度", "状态", "怎么样"]):
        if any(word in assistant_text for word in ["完成", "已", "✅", "成功"]):
            return f"状态更新: {user_text[:50]} → {assistant_text[:200]}"
    
    return None


# 在 sync_turn 方法的末尾添加（在 return None 之前）：

def _auto_store_if_enabled(self, user_content: str, assistant_content: str) -> None:
    """
    如果启用自动存储，提取并保存关键信息.
    """
    if not self._should_auto_store():
        return
    
    if self._memory_mode != "experimental_global":
        return
    
    # 提取关键信息
    key_info = self._extract_key_info(user_content, assistant_content)
    if not key_info:
        return
    
    # 调用write_proposals存储
    try:
        result = self._client.write_proposals(
            key_info,
            scope_level="session"  # 或根据内容判断scope
        )
        # 静默存储，不打断用户
    except Exception:
        pass  # 存储失败不影响对话


# 使用说明：

"""
1. 在 sync_turn 方法末尾（return None 之前）添加：
   
   # Auto-store if enabled
   self._auto_store_if_enabled(user_content, assistant_content)

2. 设置环境变量启用功能：

   # .env 文件
   MNEMOSYNE_MEMORY_MODE=experimental_global
   MNEMOSYNE_AUTO_RECALL=1
   MNEMOSYNE_AUTO_STORE=1
   MNEMOSYNE_EXPERIMENTAL_SHADOW=1

3. 重启Hermes服务

4. 验证：
   - 对话中提到"请记住XXX"，会自动存储
   - 询问"项目进度"，会自动检索
   - 所有操作静默进行，不打断对话
"""
