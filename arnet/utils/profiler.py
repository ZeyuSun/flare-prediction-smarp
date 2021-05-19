import cProfile
import pstats

class Profiler(cProfile.Profile):
    def __exit__(self, *exc_info):
        self.disable()
        self.ps = pstats.Stats(self)
        self.ps.sort_stats('cumtime').print_stats(30)
