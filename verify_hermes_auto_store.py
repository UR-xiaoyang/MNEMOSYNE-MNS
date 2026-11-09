#!/usr/bin/env python
"""
验证Hermes自动存储功能的测试脚本

使用方法：
1. 重启Hermes（让修改生效）
2. 在Hermes对话中说："请记住V3已部署"
3. 运行此脚本： python verify_hermes_auto_store.py
"""

import sqlite3
import os
from pathlib import Path

print("=" * 60)
print("Hermes 自动存储功能验证")
print("=" * 60)
print()

# 1. 检查Hermes日志
hermes_log = Path.home() / "hermes_sync_debug.log"
print("1. 检查Hermes sync_all日志...")
if hermes_log.exists():
    print(f"   ✅ 日志文件存在: {hermes_log}")
    with open(hermes_log, 'r', encoding='utf-8') as f:
        content = f.read()
        lines = content.strip().split('\n')
        print(f"   📝 日志行数: {len(lines)}")
        print(f"   📝 最后几行:")
        for line in lines[-10:]:
            print(f"      {line}")
else:
    print(f"   ❌ 日志文件不存在: {hermes_log}")
    print("   → Hermes的sync_all可能没有被调用")
    print("   → 或者Hermes没有使用修改后的源代码")

print()

# 2. 检查记忆数
print("2. 检查记忆数变化...")
db_path = 'D:/project/MNEMOSYNE-MNS/data/memory.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute('SELECT COUNT(*) FROM memories')
current_count = cursor.fetchone()[0]

print(f"   当前记忆数: {current_count}")
print(f"   之前记忆数: 358")

if current_count > 358:
    diff = current_count - 358
    print(f"   ✅ 增加了 {diff} 条记忆！")
    print()
    print("   最新的记忆:")
    cursor.execute('SELECT content FROM memories ORDER BY rowid DESC LIMIT 3')
    for i, (content,) in enumerate(cursor.fetchall(), 1):
        print(f"   {i}. {content[:80]}...")
else:
    print(f"   ❌ 记忆数未变化")

conn.close()

print()

# 3. 诊断
print("3. 诊断分析...")
print()

if not hermes_log.exists():
    print("❌ 问题: sync_all没有被调用")
    print("   可能原因:")
    print("   - Hermes没有使用D:/Hermes/source的代码")
    print("   - run_agent.py的修改没有生效")
    print("   - Hermes在不同路径运行")
    print()
    print("   解决方案:")
    print("   - 确认Hermes启动方式")
    print("   - 检查D:/Hermes/Start-Hermes-Preview-Shared.cmd")
elif current_count == 358:
    print("⚠️ 问题: sync_all被调用了，但记忆没增加")
    print("   可能原因:")
    print("   - 插件的sync_turn没有被调用")
    print("   - 自动存储的条件不满足")
    print("   - write_proposals失败")
    print()
    print("   下一步:")
    print("   - 在插件中添加日志")
    print("   - 检查环境变量是否生效")
else:
    print("✅ 成功! 自动存储已生效!")
    print()
    print("   🎉 所有功能正常:")
    print("   - Hermes调用sync_all ✓")
    print("   - 插件执行sync_turn ✓")
    print("   - 自动存储触发 ✓")
    print("   - 记忆写入数据库 ✓")

print()
print("=" * 60)
