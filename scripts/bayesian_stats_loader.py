#!/usr/bin/env python3
"""
从memory.db加载贝叶斯统计的工具函数。

V3推理时使用这些函数来填充贝叶斯特征。
"""
import sqlite3
from pathlib import Path
from typing import Dict, List


class BayesianStatsLoader:
    """加载和缓存贝叶斯统计。"""
    
    def __init__(self, db_path: str = "data/memory.db"):
        self.db_path = Path(db_path)
        self._cache = None
        self._load_all()
    
    def _load_all(self):
        """加载所有贝叶斯统计到内存。"""
        self._cache = {}  # 初始化为空字典
        
        if not self.db_path.exists():
            return
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # 检查表是否存在
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='bayesian_stats'
        """)
        
        if not cursor.fetchone():
            conn.close()
            return
        
        # 加载所有统计
        cursor.execute("""
            SELECT memory_id, n_retrievals, n_citations,
                   bayes_mean, bayes_std, bayes_ci_width,
                   bayes_coverage, bayes_consistency, bayes_confidence
            FROM bayesian_stats
        """)
        
        for row in cursor.fetchall():
            memory_id = row[0]
            self._cache[memory_id] = {
                'n_retrievals': row[1],
                'n_citations': row[2],
                'bayes_mean': row[3],
                'bayes_std': row[4],
                'bayes_ci_width': row[5],
                'bayes_coverage': row[6],
                'bayes_consistency': row[7],
                'bayes_confidence': row[8]
            }
        
        conn.close()
    
    def get_stats(self, memory_id: str) -> Dict[str, float]:
        """
        获取指定记忆的贝叶斯统计。
        
        Returns:
            dict with keys: bayes_mean, bayes_std, bayes_ci_width,
                           bayes_coverage, bayes_consistency, bayes_confidence
        """
        if memory_id in self._cache:
            stats = self._cache[memory_id]
            return {
                'bayes_mean': stats['bayes_mean'],
                'bayes_std': stats['bayes_std'],
                'bayes_ci_width': stats['bayes_ci_width'],
                'bayes_coverage': stats['bayes_coverage'],
                'bayes_consistency': stats['bayes_consistency'],
                'bayes_confidence': stats['bayes_confidence']
            }
        else:
            # 返回零值（先验）
            return {
                'bayes_mean': 0.0,
                'bayes_std': 0.0,
                'bayes_ci_width': 0.0,
                'bayes_coverage': 0.0,
                'bayes_consistency': 0.0,
                'bayes_confidence': 0.0
            }
    
    def get_batch_stats(self, memory_ids: List[str]) -> List[Dict[str, float]]:
        """批量获取贝叶斯统计。"""
        return [self.get_stats(mid) for mid in memory_ids]
    
    def reload(self):
        """重新加载统计（用于更新后刷新缓存）。"""
        self._load_all()
    
    @property
    def coverage(self) -> float:
        """返回统计覆盖率。"""
        if not self._cache:
            return 0.0
        return len(self._cache)
    
    def __len__(self):
        return len(self._cache)


# 全局单例
_global_loader = None


def get_bayesian_stats_loader(db_path: str = "data/memory.db") -> BayesianStatsLoader:
    """获取全局贝叶斯统计加载器（单例模式）。"""
    global _global_loader
    if _global_loader is None:
        _global_loader = BayesianStatsLoader(db_path)
    return _global_loader


def reload_bayesian_stats():
    """重新加载贝叶斯统计。"""
    global _global_loader
    if _global_loader is not None:
        _global_loader.reload()
