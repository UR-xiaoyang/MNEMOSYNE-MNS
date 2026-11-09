@echo off
echo ============================================
echo Hermes 完全重启和验证脚本
echo ============================================
echo.

echo 1. 关闭所有Hermes进程...
taskkill /F /IM Hermes.exe 2>nul
if %ERRORLEVEL% == 0 (
    echo    ✓ 已关闭Hermes进程
) else (
    echo    - 没有Hermes进程在运行
)

echo.
echo 2. 等待3秒...
timeout /t 3 /nobreak >nul

echo.
echo 3. 检查当前记忆数...
python -c "import sqlite3; c=sqlite3.connect('D:/project/MNEMOSYNE-MNS/data/memory.db'); count=c.execute('SELECT COUNT(*) FROM memories').fetchone()[0]; print(f'   当前记忆数: {count}'); c.close()"

echo.
echo 4. 启动Hermes...
echo    请手动运行: D:\Hermes\Start-Hermes-Preview-Shared.cmd
echo.
echo 5. 启动后在Hermes中测试：
echo    说："请记住MNEMOSYNE V3已部署"
echo.
echo 6. 然后运行验证：
echo    python D:/project/MNEMOSYNE-MNS/verify_auto_store.py
echo.
pause
