import time,logging,sys

logging.basicConfig(format = '%(module)s-%(levelname)s- %(message)s')
logger = logging.getLogger('featuretoolsOnSpark')
logger.setLevel(20)

if sys.platform == 'win32':
    process_time = time.clock
else:
    process_time = time.time