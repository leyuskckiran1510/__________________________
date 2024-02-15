import pstats


def read_profile_results(filename):
    stats = pstats.Stats(filename)
    stats.strip_dirs().sort_stats(-1).print_stats()


if __name__ == "__main__":
    read_profile_results("profile.txt")
