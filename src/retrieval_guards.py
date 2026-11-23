"""检索输出检查：只拒绝非法结果，不补邻居、不重检索、不改变概率或算法。"""
import torch

def validate_neighbors(distances, ids, datastore_size):
 if not isinstance(datastore_size,int) or datastore_size<=0:
  raise ValueError('datastore_size必须是正整数')
 if not isinstance(distances,torch.Tensor) or not isinstance(ids,torch.Tensor):
  raise TypeError('FAISS torch集成预期返回Tensor')
 if distances.ndim!=2 or ids.shape!=distances.shape or ids.numel()==0:
  raise ValueError('邻居必须为非空且同形状的[frames,k]张量')
 if ids.dtype not in (torch.int32,torch.int64):raise TypeError('邻居ID必须是整数Tensor')
 if not torch.isfinite(distances).all().item():raise ValueError('FAISS距离包含NaN/Inf，禁止计算概率')
 if ((ids<0)|(ids>=datastore_size)).any().item():
  raise ValueError('FAISS返回非法邻居ID（可能含-1/超出库容量）；禁止负索引读取标签，不自动补邻居或改变probe')
 return distances,ids
