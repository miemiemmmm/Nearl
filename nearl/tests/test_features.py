import numpy as np 
from nearl import features
import all_actions 

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


def test_aggregation_functions(): 
  np.random.seed(0) 
  x = np.random.rand(50, 32*32*32)

  for agg_mode in range(1, 9): 
    c = 0
    ret = all_actions.aggregate(x, agg_mode)
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
        continue
      elif agg_mode == 8:
        agg_cpu = get_slope(x[:, i])

      if not np.isclose(agg_cpu, ret[i]): 
        print(c, i, np.isclose(agg_cpu, ret[i]), np.round(agg_cpu, 8), np.round(ret[i], 8))
        c += 1
    print(f"{agg_map} Found {c} mismatches")

