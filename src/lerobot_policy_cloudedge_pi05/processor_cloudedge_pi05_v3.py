from .processor_cloudedge_pi05 import make_cloudedge_pi05_pre_post_processors


def make_cloudedge_pi05_v3_pre_post_processors(config, dataset_stats=None):
    return make_cloudedge_pi05_pre_post_processors(
        config, dataset_stats=dataset_stats
    )
