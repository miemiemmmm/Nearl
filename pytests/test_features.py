import numpy as np 
import nearl.all_actions as all_actions 

# from nearl import features

agg_map = {
  1: "mean",
  2: "std",
  3: "median",
  4: "variance",
  5: "max",
  6: "min",
  7: "sum",
  8: "slope"
}

def get_mean(points): 
  return np.mean(points)

def get_std(points): 
  return np.std(points)

def get_median(points): 
  points = np.sort(points)
  n = len(points)
  if n % 2 == 0: 
    return (points[n//2] + points[n//2 - 1]) / 2
  else: 
    return points[n//2]

def get_variance(points): 
  return np.var(points)

def get_max(points): 
  return np.max(points)

def get_min(points):
  return np.min(points)

def get_slope(points):
  n = len(points)
  if n < 2:
    return None  # Need at least 2 points to calculate slope
  
  sum_x = sum_y = sum_xx = sum_xy = 0
  
  for x, y in enumerate(points):
    sum_x += x
    sum_y += y
    sum_xx += x * x
    sum_xy += x * y

  # Calculate numerator and denominator for slope formula
  numerator = n * sum_xy - sum_x * sum_y
  denominator = n * sum_xx - sum_x ** 2
  
  if denominator == 0:
    return 0  # Avoid division by zero (vertical line or single point)
  else: 
    return numerator / denominator

def get_info_entropy(Arr):
  # Split into 16 bins
  INFORMATION_ENTROPY_BINS = 16
  N = len(Arr)
  if N <= 1:
    return 0.0
  # Normalization
  min_val = np.min(Arr)
  max_val = np.max(Arr)
  if min_val == max_val:
      return 0.0
  # Compute histogram with bins between min_val and max_val
  hist, _ = np.histogram(Arr, bins=INFORMATION_ENTROPY_BINS, range=(min_val, max_val))
  
  # Calculate entropy
  entropy_val = 0.0
  for count in hist:
    if count > 0:
      prob = count / N
      entropy_val -= prob * np.log2(prob)
  
  return entropy_val


def test_aggregation_functions(): 
  np.random.seed(0) 
  x = np.random.rand(50, 32*32*32)

  for agg_mode in range(1, 9): 
  # for agg_mode in [1, 3, 5, 6]:
    c = 0
    ret = all_actions.aggregate(x, agg_mode)
    _agg_cpu = []
    for i in range(32*32*32): 
      if agg_mode == 1:
        agg_cpu = get_mean(x[:, i])
      elif agg_mode == 2:
        agg_cpu = get_std(x[:, i])
      elif agg_mode == 3:
        agg_cpu = get_median(x[:, i])
      elif agg_mode == 4:
        agg_cpu = get_variance(x[:, i])
      elif agg_mode == 5:
        agg_cpu = get_max(x[:, i])
      elif agg_mode == 6:
        agg_cpu = get_min(x[:, i])
      elif agg_mode == 7:
        agg_cpu = get_info_entropy(x[:, i])
      elif agg_mode == 8:
        agg_cpu = get_slope(x[:, i])
      else: 
        continue
      _agg_cpu.append(agg_cpu)

    agg_cpu = np.array(_agg_cpu, dtype=np.float32)

    print(f"Examine the aggregation: {agg_map[agg_mode]:8}: ")
    print(f"CPU mean: {np.mean(agg_cpu):6.4f}; GPU mean: {np.mean(ret):6.4f}; ")
    print(f"Number of mismatch {np.count_nonzero(~np.isclose(agg_cpu, ret, atol=1e-3))}")

    assert np.isclose(np.sum(agg_cpu), np.sum(ret), rtol=1e-3, atol=1e-3)
    assert np.isclose(np.mean(agg_cpu), np.mean(ret), rtol=1e-3, atol=1e-3)
    assert np.isclose(np.min(agg_cpu), np.min(ret), rtol=1e-3, atol=1e-3)
    assert np.isclose(np.max(agg_cpu), np.max(ret), rtol=1e-3, atol=1e-3)
    assert np.count_nonzero(~np.isclose(agg_cpu, ret, atol=1e-3)) < 10
