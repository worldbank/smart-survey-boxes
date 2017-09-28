import multiprocessing

import prep as prep

if __name__ == "__main__":
    multiprocessing.set_start_method('forkserver', force=True)
    # ----------SET UP-----------------------------------------
    config = prep.Configurations(platform='mac')
    config.debug_mode = False