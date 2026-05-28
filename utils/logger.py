import logging
import warnings


def setup_logger(name="hydroverse", level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s · %(levelname)s · %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger(name)
    for noisy in ["urllib3", "matplotlib", "fiona", "folium", "rasterio", "PIL", "fsspec", "botocore", "s3transfer"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)
    warnings.filterwarnings("ignore")
    return log


get_logger = setup_logger
log = setup_logger()
