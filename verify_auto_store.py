#!/usr/bin/env python
"""验证MNEMOSYNE自动存储功能"""
import sqlite3
import sys

print('=' * 50)
print('MNEMOSYNE 自动存储功能验证')
print('=' * 50)
print()

# 检查记忆数
conn = sqlite3.connect('D:/project/MNEMOSYNE-MNS/data/memory.db')
cursor = conn.cursor()

cursor.execute('SELECT COUNT(*) FROM memories')
current_count = cursor.fetchone()[0]

print(f'之前记忆数: 358')
print(f'当前记忆数: {current_count}')
print()

if current_count > 358:
    diff = current_count - 358
    print(f'✅ 成功！记忆增加了 {diff} 条')
    print('✅ 自动存储功能已生效！')
    print()
    
    # 显示最新的记忆
    cursor.execute('SELECT content FROM memories ORDER BY rowid DESC LIMIT 3')
    print('最新的记忆:')
    for i, (content,) in enumerate(cursor.fetchall(), 1):
        print(f'  {i}. {content[:80]}...')
    print()
    print('🎉 插件增强完全成功！')
    sys.exit(0)
else:
    print('❌ 记忆数未增加')
    print()
    print('可能原因:')
    print('  1. Hermes未完全重启（还有旧进程）')
    print('  2. 环境变量未生效')
    print('  3. sync_turn未被调用')
    print()
    print('调试步骤:')
    print('  1. 确认所有Hermes.exe进程已关闭')
    print('  2. 检查Hermes启动日志中的[DEBUG]输出')
    print('  3. 确认使用的是shared模式')
    sys.exit(1)

conn.close()
