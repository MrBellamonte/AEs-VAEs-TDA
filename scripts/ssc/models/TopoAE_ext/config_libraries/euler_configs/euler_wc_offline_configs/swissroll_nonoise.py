import itertools

from scripts.ssc.wc_offline.config_libraries.global_register_definitions import \
    PATH_GR_SWISSROLL_EULER

SWISSROLL_NONOISE36 = [
    dict(uid='SwissRoll-bs64-seed36-219e7a83', path_global_register=PATH_GR_SWISSROLL_EULER),
    dict(uid='SwissRoll-bs128-seed36-e11a0aee', path_global_register=PATH_GR_SWISSROLL_EULER),
    dict(uid='SwissRoll-bs256-seed36-8b60871f', path_global_register=PATH_GR_SWISSROLL_EULER),
    dict(uid='SwissRoll-bs512-seed36-d3233ff9', path_global_register=PATH_GR_SWISSROLL_EULER)]
SWISSROLL_NONOISE3851 = [
    dict(uid='SwissRoll-bs64-seed3851-53c22261', path_global_register=PATH_GR_SWISSROLL_EULER),
    dict(uid='SwissRoll-bs128-seed3851-1064a442', path_global_register=PATH_GR_SWISSROLL_EULER),
    dict(uid='SwissRoll-bs256-seed3851-1363f36b', path_global_register=PATH_GR_SWISSROLL_EULER),
    dict(uid='SwissRoll-bs512-seed3851-32952589', path_global_register=PATH_GR_SWISSROLL_EULER)]
SWISSROLL_NONOISE2570 = [
    dict(uid='SwissRoll-bs64-seed2570-155693e8', path_global_register=PATH_GR_SWISSROLL_EULER),
    dict(uid='SwissRoll-bs128-seed2570-f7f04474', path_global_register=PATH_GR_SWISSROLL_EULER),
    dict(uid='SwissRoll-bs256-seed2570-139af65a', path_global_register=PATH_GR_SWISSROLL_EULER),
    dict(uid='SwissRoll-bs512-seed2570-0fc65b2e', path_global_register=PATH_GR_SWISSROLL_EULER)]
SWISSROLL_NONOISE4304 = [
    dict(uid='SwissRoll-bs64-seed4304-0cbd8ebf', path_global_register=PATH_GR_SWISSROLL_EULER),
    dict(uid='SwissRoll-bs128-seed4304-73c572a9', path_global_register=PATH_GR_SWISSROLL_EULER),
    dict(uid='SwissRoll-bs256-seed4304-7cb0629d', path_global_register=PATH_GR_SWISSROLL_EULER),
    dict(uid='SwissRoll-bs512-seed4304-4f0dab79', path_global_register=PATH_GR_SWISSROLL_EULER)]
SWISSROLL_NONOISE1935 = [
    dict(uid='SwissRoll-bs64-seed1935-dac50137', path_global_register=PATH_GR_SWISSROLL_EULER),
    dict(uid='SwissRoll-bs128-seed1935-8373809c', path_global_register=PATH_GR_SWISSROLL_EULER),
    dict(uid='SwissRoll-bs256-seed1935-674e2b46', path_global_register=PATH_GR_SWISSROLL_EULER),
    dict(uid='SwissRoll-bs512-seed1935-af8e4387', path_global_register=PATH_GR_SWISSROLL_EULER)]
SWISSROLL_NONOISE7954 = [
    dict(uid='SwissRoll-bs64-seed7954-f5764f94', path_global_register=PATH_GR_SWISSROLL_EULER),
    dict(uid='SwissRoll-bs128-seed7954-ec899e0c', path_global_register=PATH_GR_SWISSROLL_EULER),
    dict(uid='SwissRoll-bs256-seed7954-4fed45a9', path_global_register=PATH_GR_SWISSROLL_EULER),
    dict(uid='SwissRoll-bs512-seed7954-62f47c2c', path_global_register=PATH_GR_SWISSROLL_EULER)]
SWISSROLL_NONOISE5095 = [
    dict(uid='SwissRoll-bs64-seed5095-51767276', path_global_register=PATH_GR_SWISSROLL_EULER),
    dict(uid='SwissRoll-bs128-seed5095-df6346bb', path_global_register=PATH_GR_SWISSROLL_EULER),
    dict(uid='SwissRoll-bs256-seed5095-cedd2abc', path_global_register=PATH_GR_SWISSROLL_EULER),
    dict(uid='SwissRoll-bs512-seed5095-5e9429f9', path_global_register=PATH_GR_SWISSROLL_EULER)]
SWISSROLL_NONOISE5310 = [
    dict(uid='SwissRoll-bs64-seed5310-1da8fdb0', path_global_register=PATH_GR_SWISSROLL_EULER),
    dict(uid='SwissRoll-bs128-seed5310-d39df50c', path_global_register=PATH_GR_SWISSROLL_EULER),
    dict(uid='SwissRoll-bs256-seed5310-b344784f', path_global_register=PATH_GR_SWISSROLL_EULER),
    dict(uid='SwissRoll-bs512-seed5310-0fb98ff8', path_global_register=PATH_GR_SWISSROLL_EULER)]
SWISSROLL_NONOISE1577 = [
    dict(uid='SwissRoll-bs64-seed1577-af3626e9', path_global_register=PATH_GR_SWISSROLL_EULER),
    dict(uid='SwissRoll-bs128-seed1577-0415c381', path_global_register=PATH_GR_SWISSROLL_EULER),
    dict(uid='SwissRoll-bs256-seed1577-22fb713b', path_global_register=PATH_GR_SWISSROLL_EULER),
    dict(uid='SwissRoll-bs512-seed1577-2f71df00', path_global_register=PATH_GR_SWISSROLL_EULER)]
SWISSROLL_NONOISE3288 = [
    dict(uid='SwissRoll-bs64-seed3288-b430a7a3',  path_global_register=PATH_GR_SWISSROLL_EULER),
    dict(uid='SwissRoll-bs128-seed3288-8bf65659', path_global_register=PATH_GR_SWISSROLL_EULER),
    dict(uid='SwissRoll-bs256-seed3288-b350a4a4', path_global_register=PATH_GR_SWISSROLL_EULER),
    dict(uid='SwissRoll-bs512-seed3288-5a828a69', path_global_register=PATH_GR_SWISSROLL_EULER)]

SWISSROLL_NONOISE_all = itertools.chain(SWISSROLL_NONOISE36,
                         SWISSROLL_NONOISE3851,
                         SWISSROLL_NONOISE2570,
                         SWISSROLL_NONOISE4304,
                         SWISSROLL_NONOISE1935,
                         SWISSROLL_NONOISE7954,
                         SWISSROLL_NONOISE5095,
                         SWISSROLL_NONOISE5310,
                         SWISSROLL_NONOISE1577,
                         SWISSROLL_NONOISE3288)

SWISSROLL_NONOISE_h1 = itertools.chain(SWISSROLL_NONOISE36,
                         SWISSROLL_NONOISE3851,
                         SWISSROLL_NONOISE2570,
                         SWISSROLL_NONOISE4304,
                         SWISSROLL_NONOISE1935)

SWISSROLL_NONOISE_h2 = itertools.chain(
                         SWISSROLL_NONOISE7954,
                         SWISSROLL_NONOISE5095,
                         SWISSROLL_NONOISE5310,
                         SWISSROLL_NONOISE1577,
                         SWISSROLL_NONOISE3288)
