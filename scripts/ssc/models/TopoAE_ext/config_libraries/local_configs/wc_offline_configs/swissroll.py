import itertools

from scripts.ssc.wc_offline.config_libraries.global_register_definitions import \
    (
    PATH_GR_SWISSROLL_LOCAL, PATH_GR_SWISSROLL_LOCAL_SERVER)

SWISSROLL_NONOISE36 = [
    dict(uid='SwissRoll-bs64-seed36-f5c07948', path_global_register=PATH_GR_SWISSROLL_LOCAL),
    dict(uid='SwissRoll-bs128-seed36-0fee517e', path_global_register=PATH_GR_SWISSROLL_LOCAL),
    dict(uid='SwissRoll-bs256-seed36-b3aa917f', path_global_register=PATH_GR_SWISSROLL_LOCAL),
    dict(uid='SwissRoll-bs512-seed36-50563bfc', path_global_register=PATH_GR_SWISSROLL_LOCAL)]


SWISSROLL_NONOISE36_LOCALSERVER = [
    dict(uid='SwissRoll-bs64-seed36-f5c07948', path_global_register= PATH_GR_SWISSROLL_LOCAL_SERVER),
    dict(uid='SwissRoll-bs128-seed36-0fee517e', path_global_register= PATH_GR_SWISSROLL_LOCAL_SERVER),
    dict(uid='SwissRoll-bs256-seed36-b3aa917f', path_global_register= PATH_GR_SWISSROLL_LOCAL_SERVER),
    dict(uid='SwissRoll-bs512-seed36-50563bfc', path_global_register= PATH_GR_SWISSROLL_LOCAL_SERVER)]



