from board_cell_classes import DLGrid

grid =  DLGrid((18, 18))
grid.run_n_episodes(n=100, vis=False)