#!/usr/bin/env python3
"""
从V2的Shadow数据中提取贝叶斯统计，用于V3模型。

V2已经收集了：
- 1,266 次检索事件
- 273 次引用事件  
- 153 个记忆有统计数据
- 覆盖率: 45.8%

这些数据可以直接转换为V3的贝叶斯特征。
"""
import sqlite3
import json
from pathlib import Path
from collections import defaultdict
import numpy as np


def extract_bayesian_stats_from_shadow(shadow_db_path: str) -> dict:
    """
    从Shadow数据库提取每个记忆的贝叶斯统计。
    
    Returns:
        dict: {memory_id: {'n_retrievals': int, 'n_citations': int}}
    """
    conn = sqlite3.connect(shadow_db_path)
    cursor = conn.cursor()
    
    stats = defaultdict(lambda: {'n_retrievals': 0, 'n_citations': 0})
    
    print("正在从Shadow数据库提取统计...")
    
    # 1. 统计每个记忆被检索的次数
    cursor.execute("""
        SELECT json_extract(value, '$.memory_id') as memory_id
        FROM retrieval_events, json_each(retrieval_events.candidates_json)
    """)
    
    for (memory_id,) in cursor.fetchall():
        if memory_id:
            stats[memory_id]['n_retrievals'] += 1
    
    # 2. 统计每个记忆被引用的次数
    cursor.execute("""
        SELECT memory_id, COUNT(*) as n_citations
        FROM citation_events
        GROUP BY memory_id
    """)
    
    for memory_id, n_cit in cursor.fetchall():
        stats[memory_id]['n_citations'] = n_cit
    
    conn.close()
    
    # 转换为普通dict
    stats = {k: dict(v) for k, v in stats.items()}
    
    print(f"✓ 提取了 {len(stats)} 个记忆的统计")
    print(f"✓ 有引用记录: {sum(1 for s in stats.values() if s['n_citations'] > 0)} 个")
    
    return stats


def compute_bayesian_features(n_retrievals: int, n_citations: int, 
                               alpha_prior: float = 1.0, 
                               beta_prior: float = 1.0) -> dict:
    """
    从检索/引用次数计算贝叶斯特征。
    
    使用Beta分布：Beta(α, β)
    - α = alpha_prior + n_citations
    - β = beta_prior + (n_retrievals - n_citations)
    
    Returns:
        dict: {
            'bayes_mean': float,
            'bayes_std': float,
            'bayes_ci_width': float,
            'bayes_coverage': float,
            'bayes_consistency': float,
            'bayes_confidence': float
        }
    """
    if n_retrievals == 0:
        # 无数据，返回先验
        return {
            'bayes_mean': 0.0,
            'bayes_std': 0.0,
            'bayes_ci_width': 0.0,
            'bayes_coverage': 0.0,
            'bayes_consistency': 0.0,
            'bayes_confidence': 0.0
        }
    
    # Beta后验参数
    alpha = alpha_prior + n_citations
    beta = beta_prior + (n_retrievals - n_citations)
    
    # 后验均值
    mean = alpha / (alpha + beta)
    
    # 后验标准差
    variance = (alpha * beta) / ((alpha + beta)**2 * (alpha + beta + 1))
    std = np.sqrt(variance)
    
    # 95%置信区间宽度（近似为4倍标准差）
    ci_width = 4 * std
    
    # 覆盖率：是否有足够数据（阈值：>=5次检索）
    coverage = 1.0 if n_retrievals >= 5 else 0.0
    
    # 一致性：引用率
    consistency = n_citations / n_retrievals
    
    # 置信度：不确定性的倒数
    confidence = 1.0 / (std + 1e-6)
    
    return {
        'bayes_mean': float(mean),
        'bayes_std': float(std),
        'bayes_ci_width': float(ci_width),
        'bayes_coverage': float(coverage),
        'bayes_consistency': float(consistency),
        'bayes_confidence': float(confidence)
    }


def create_bayesian_stats_table(memory_db_path: str):
    """在memory.db中创建贝叶斯统计表。"""
    conn = sqlite3.connect(memory_db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS bayesian_stats (
            memory_id TEXT PRIMARY KEY,
            n_retrievals INTEGER NOT NULL,
            n_citations INTEGER NOT NULL,
            bayes_mean REAL,
            bayes_std REAL,
            bayes_ci_width REAL,
            bayes_coverage REAL,
            bayes_consistency REAL,
            bayes_confidence REAL,
            last_updated REAL NOT NULL
        )
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_bayesian_memory 
        ON bayesian_stats(memory_id)
    """)
    
    conn.commit()
    conn.close()
    
    print("✓ 贝叶斯统计表已创建")


def save_bayesian_stats(memory_db_path: str, stats: dict):
    """将贝叶斯统计保存到memory.db。"""
    import time
    
    conn = sqlite3.connect(memory_db_path)
    cursor = conn.cursor()
    
    timestamp = time.time()
    saved_count = 0
    
    for memory_id, stat in stats.items():
        # 计算贝叶斯特征
        features = compute_bayesian_features(
            stat['n_retrievals'], 
            stat['n_citations']
        )
        
        cursor.execute("""
            INSERT OR REPLACE INTO bayesian_stats
            (memory_id, n_retrievals, n_citations, 
             bayes_mean, bayes_std, bayes_ci_width,
             bayes_coverage, bayes_consistency, bayes_confidence,
             last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            memory_id,
            stat['n_retrievals'],
            stat['n_citations'],
            features['bayes_mean'],
            features['bayes_std'],
            features['bayes_ci_width'],
            features['bayes_coverage'],
            features['bayes_consistency'],
            features['bayes_confidence'],
            timestamp
        ))
        saved_count += 1
    
    conn.commit()
    conn.close()
    
    print(f"✓ 保存了 {saved_count} 个记忆的贝叶斯统计")


def analyze_stats(stats: dict):
    """分析统计数据质量。"""
    print()
    print("=== 统计数据分析 ===")
    
    n_total = len(stats)
    n_with_citations = sum(1 for s in stats.values() if s['n_citations'] > 0)
    n_sufficient = sum(1 for s in stats.values() if s['n_retrievals'] >= 5)
    
    total_retrievals = sum(s['n_retrievals'] for s in stats.values())
    total_citations = sum(s['n_citations'] for s in stats.values())
    
    print(f"记忆总数: {n_total}")
    print(f"有引用记录: {n_with_citations} ({n_with_citations/n_total*100:.1f}%)")
    print(f"数据充足(≥5次): {n_sufficient} ({n_sufficient/n_total*100:.1f}%)")
    print()
    print(f"总检索次数: {total_retrievals:,}")
    print(f"总引用次数: {total_citations:,}")
    print(f"整体引用率: {total_citations/total_retrievals*100:.1f}%")
    print()
    
    # 引用率分布
    citation_rates = [
        s['n_citations'] / s['n_retrievals'] 
        for s in stats.values() 
        if s['n_retrievals'] > 0
    ]
    
    if citation_rates:
        print("引用率分布:")
        print(f"  平均: {np.mean(citation_rates)*100:.1f}%")
        print(f"  中位数: {np.median(citation_rates)*100:.1f}%")
        print(f"  标准差: {np.std(citation_rates)*100:.1f}%")
        print(f"  范围: {np.min(citation_rates)*100:.1f}% - {np.max(citation_rates)*100:.1f}%")


def main():
    """主函数：提取V2统计并保存到memory.db。"""
    shadow_db = "data/shadow.db"
    memory_db = "data/memory.db"
    
    print("=" * 60)
    print("从V2 Shadow数据提取贝叶斯统计")
    print("=" * 60)
    print()
    
    # 1. 提取统计
    stats = extract_bayesian_stats_from_shadow(shadow_db)
    
    # 2. 分析数据质量
    analyze_stats(stats)
    
    # 3. 创建表
    print()
    create_bayesian_stats_table(memory_db)
    
    # 4. 保存统计
    save_bayesian_stats(memory_db, stats)
    
    print()
    print("=" * 60)
    print("✅ 完成！V2的贝叶斯统计已继承到V3")
    print("=" * 60)
    print()
    print("下一步:")
    print("1. V3推理时会自动从bayesian_stats表读取特征")
    print("2. 启动服务后继续收集更多数据")
    print("3. 定期运行此脚本更新统计")


if __name__ == "__main__":
    main()
