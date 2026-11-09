#!/usr/bin/env python
"""检查测试后的日志"""

import os
import sqlite3
from pathlib import Path

print("=" * 60)
print("日志检查结果")
print("=" * 60)
print()

# 1. Hermes sync_all日志
hermes_log = Path.home() / "hermes_sync_debug.log"
print("1. Hermes sync_all日志:")
if hermes_log.exists():
    with open(hermes_log, 'r', encoding='utf-8') as f:
        content = f.read()
    print(f"   ✅ 存在 ({len(content)} 字符)")
    print("   内容:")
    for line in content.strip().split('\n'):
        print(f"      {line}")
else:
    print("   ❌ 不存在")

print()

# 2. 插件sync_turn日志
plugin_log = Path.home() / "plugin_sync_debug.log"
print("2. 插件sync_turn日志:")
if plugin_log.exists():
    with open(plugin_log, 'r', encoding='utf-8') as f:
        content = f.read()
    print(f"   ✅ 存在 ({len(content)} 字符)")
    print("   内容:")
    for line in content.strip().split('\n'):
        print(f"      {line}")
else:
    print("   ❌ 不存在 → sync_turn没被调用!")

print()

# 3. 记忆数
print("3. 记忆数变化:")
conn = sqlite3.connect('D:/project/MNEMOSYNE-MNS/data/memory.db')
cursor = conn.cursor()
cursor.execute('SELECT COUNT(*) FROM memories')
count = cursor.fetchone()[0]
conn.close()

print(f"   当前: {count}")
print(f"   之前: 358")
if count > 358:
    print(f"   ✅ 增加了 {count - 358} 条!")
else:
    print("   ❌ 未增加")

print()
print("=" * 60)
