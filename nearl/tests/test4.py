grid = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10), np.linspace(0, 1, 10))
grid_coord = np.vstack([grid[0].flatten(), grid[1].flatten(), grid[2].flatten()]).T