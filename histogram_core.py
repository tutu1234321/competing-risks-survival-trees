import numpy as np
from numba import njit

# 【修改点 1】: 给 weightings 加上默认值 1，防止漏传报错
@njit(cache=True)
def lr_hist_statistic(time_hist_1, time_hist_2, cens_hist_1, cens_hist_2,
                      weightings=1, obs_weights=None):
    N_1_j = np.cumsum(time_hist_1[::-1])[::-1]
    N_2_j = np.cumsum(time_hist_2[::-1])[::-1]
    ind = np.where((cens_hist_1 + cens_hist_2 != 0) & (N_1_j * N_2_j != 0))[0]
    if ind.shape[0] == 0:
        return 0.0

    N_1_j = N_1_j[ind]  # + 1
    N_2_j = N_2_j[ind]  # + 1
    O_1_j = cens_hist_1[ind]
    O_2_j = cens_hist_2[ind]

    N_j = N_1_j + N_2_j
    O_j = O_1_j + O_2_j
    E_1_j = N_1_j * O_j / N_j
    res = np.zeros((N_j.shape[0], 3), dtype=np.float32)
    res[:, 1] = O_1_j - E_1_j
    res[:, 2] = E_1_j * (N_j - O_j) * N_2_j / (N_j * np.maximum(N_j - 1, 1))
    res[:, 0] = 1.0
    if weightings == 2:
        res[:, 0] = N_j
    elif weightings == 3:
        res[:, 0] = np.sqrt(N_j)
    var = (res[:, 0] * res[:, 0] * res[:, 2]).sum()
    num = res[:, 0] * res[:, 1]
    stat_val = np.power(num.sum(), 2) / var if var > 0 else 0.0 #  # ((res[:, 0] * res[:, 0] * res[:, 2]).sum())
    return stat_val


@njit(cache=True)
def vector_logrank_hist(l_time, r_time,
                        l_cens_all, r_cens_all):
    """
    对所有事件类型分别计算统计量
    """
    k = l_cens_all.shape[0]
    U = np.zeros(k, dtype=np.float32)

    for i in range(k):
        # 【修改点 2】: 在这里显式传入 weightings=1，满足 Numba 严格的参数校验
        U[i] = lr_hist_statistic(
            l_time, r_time,
            l_cens_all[i], r_cens_all[i],
            1 
        )
    return U


@njit(cache=True)
def aggregate_histograms(X_bin_idxs, time_bin_idxs, events, 
                         n_feature_bins, n_time_bins, k_events):
    """
    构建 3D 直方图：(Feature_Bin, Event_Type, Time_Bin)
    复杂度 O(N)，一次遍历完成数据聚合
    """
    time_hist = np.zeros((n_feature_bins, n_time_bins), dtype=np.float32)
    cens_hist = np.zeros((n_feature_bins, k_events, n_time_bins), dtype=np.float32)
    
    n = len(events)
    for i in range(n):
        fb = X_bin_idxs[i]
        tb = time_bin_idxs[i]
        e = events[i]
        
        time_hist[fb, tb] += 1.0
        if e > 0:
            # 只有 e > 0 才视为发生了某种竞争风险事件
            cens_hist[fb, e - 1, tb] += 1.0
            
    return time_hist, cens_hist


@njit(cache=True)
def find_best_split_by_bins(time_hist, cens_hist, split_values, min_samples_leaf, n_total, mode):
    """
    在特征直方图上滑动遍历寻找最佳切分点
    
    Parameters
    ----------
    mode : int
        0 -> "max" (选取各事件统计量的最大值作为得分)
        1 -> "mean" (选取平均值)
    """
    n_feature_bins = time_hist.shape[0]
    n_time_bins = time_hist.shape[1]
    k_events = cens_hist.shape[1]
    
    # 初始右侧（全量数据）
    r_time = np.sum(time_hist, axis=0)
    r_cens = np.sum(cens_hist, axis=0)
    
    l_time = np.zeros((n_time_bins,), dtype=np.float32)
    l_cens = np.zeros((k_events, n_time_bins), dtype=np.float32)
    
    best_score = -np.inf
    best_thr = np.nan
    best_U = np.zeros(k_events, dtype=np.float32)
    
    n_left = 0
    n_right = n_total
    
    # 滑动切分点：从左往右移动 bin
    for i in range(n_feature_bins - 1):
        cur_time = time_hist[i]
        cur_cens = cens_hist[i]
        
        l_time += cur_time
        r_time -= cur_time
        l_cens += cur_cens
        r_cens -= cur_cens
        
        count_in_bin = np.sum(cur_time)
        n_left += count_in_bin
        n_right -= count_in_bin
        
        # 叶子节点最小样本量限制
        if n_left < min_samples_leaf or n_right < min_samples_leaf:
            continue
            
        # 计算该切分点下的向量统计量
        U = vector_logrank_hist(l_time, r_time, l_cens, r_cens)
        
        # 根据聚合模式计算最终得分
        if mode == 0:     # "max"
            score = np.max(U)
        else:             # "mean"
            score = np.mean(U)
        
        if score > best_score:
            best_score = score
            best_thr = split_values[i]
            # 必须使用 .copy()，否则 U 的变动会影响结果
            best_U = U.copy()
            
    return best_score, best_thr, best_U


def hist_best_attr_split_competing(arr,
                                   min_samples_leaf=10, 
                                   signif=1.0, 
                                   signif_stat=0.0,
                                   n_bins=50, 
                                   thres_cont_bin_max=100, 
                                   comp_select="max",
                                   *args, **kwargs):
    """
    高层接口：基于直方图寻找特征的最佳切分点。
    支持竞争风险数据，返回包含切分向量的结果。
    """
    X = np.asarray(arr[0], dtype=np.float32).ravel()
    event = arr[1].astype(np.int32)
    time = arr[2].astype(np.float32)
    
    n = X.shape[0]
    default_thr = {"stat_val": signif_stat, "p_value": signif, "sign_split": 0, "values": [], "pos_nan": [1, 0]}
    
    # 前置条件检查
    if n < 2 * min_samples_leaf:
        return default_thr

    k = int(event.max())
    if k <= 0: # 必须有事件发生
        return default_thr

    t_min, t_max = time.min(), time.max()
    if t_min == t_max:
        return default_thr
        
    # 1. 时间分桶 (Time Binning)
    time_bins_edges = np.linspace(t_min, t_max, n_bins + 1)
    t_idx = np.searchsorted(time_bins_edges, time, side="right") - 1
    t_idx = np.clip(t_idx, 0, n_bins - 1).astype(np.int32)

    # 2. 特征离散化/分桶 (Feature Binning)
    unique_x = np.unique(X)
    if len(unique_x) <= 1:
        return default_thr
        
    if len(unique_x) > thres_cont_bin_max:
        # 连续特征使用分位数分桶
        percentiles = np.linspace(0, 100, thres_cont_bin_max + 2)[1:-1]
        split_points = np.unique(np.percentile(X, percentiles))
    else:
        # 离散值较少时取中点作为候选点
        split_points = (unique_x[:-1] + unique_x[1:]) / 2.0
    
    X_bin_idxs = np.searchsorted(split_points, X, side='right').astype(np.int32)
    n_feature_bins = len(split_points) + 1
    
    # 3. 聚合直方图
    time_hist, cens_hist = aggregate_histograms(
        X_bin_idxs, t_idx, event, 
        n_feature_bins, n_bins, k
    )
    
    # 4. 寻找最佳切分点
    mode = 0 if comp_select == "max" else 1
    best_score, best_thr, best_U = find_best_split_by_bins(
        time_hist, cens_hist, split_points, min_samples_leaf, n, mode
    )
    
    if np.isnan(best_thr) or best_score <= signif_stat:
        return default_thr

    return {
        "stat_val": float(best_score),
        "threshold": float(best_thr),
        "values": [f"<= {best_thr}", f"> {best_thr}"],
        "sign_split": 1,
        "pos_nan": [1, 0],
        "stat_vec": best_U  # 返回每种事件类型的具体统计向量
    }