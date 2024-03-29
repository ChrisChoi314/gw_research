import os
import numpy as np
import scipy.linalg as sl
import matplotlib.pyplot as plt

# matplotlib inline

import glob, pickle, json

import hasasia.sensitivity as hsen
import hasasia.sim as hsim
import hasasia.skymap as hsky

import matplotlib as mpl

mpl.rcParams["figure.dpi"] = 300
mpl.rcParams["figure.figsize"] = [5, 3]
mpl.rcParams["text.usetex"] = True

from enterprise.pulsar import Pulsar as ePulsar

noise_dir = "../NANOGrav_15yr_v1.0.1/narrowband/noise/"
noise_files = sorted(glob.glob(noise_dir + "*wb.pars.txt"))

psr_list = [
    "B1855+09",
    "B1937+21",
    "B1937+21ao",
    "B1937+21gbt",
    "B1953+29",
    "J0023+0923",
    "J0030+0451",
    "J0340+4130",
    "J0406+3039",
    "J0437-4715",
    "J0509+0856",
    "J0557+1551",
    "J0605+3757",
    "J0610-2100",
    "J0613-0200",
    "J0614-3329",
    "J0636+5128",
    "J0645+5158",
    "J0709+0458",
    "J0740+6620",
    "J0931-1902",
    "J1012+5307",
    "J1012-4235",
    "J1022+1001",
    "J1024-0719",
    "J1125+7819",
    "J1312+0051",
    "J1453+1902",
    "J1455-3330",
    "J1600-3053",
    "J1600-3053gbt",
    "J1614-2230",
    "J1630+3734",
    "J1640+2224",
    "J1643-1224",
    "J1643-1224gbt",
    "J1705-1903",
    "J1713+0747",
    "J1713+0747ao",
    "J1713+0747gbt",
    "J1719-1438",
    "J1730-2304",
    "J1738+0333",
    "J1741+1351",
    "J1744-1134",
    "J1745+1017",
    "J1747-4036",
    "J1751-2857",
    "J1802-2124",
    "J1811-2405",
    "J1832-0836",
    "J1843-1113",
    "J1853+1303",
    "J1903+0327",
    "J1903+0327ao",
    "J1909-3744",
    "J1909-3744gbt",
    "J1910+1256",
    "J1911+1347",
    "J1918-0642",
    "J1923+2515",
    "J1944+0907",
    "J1946+3417",
    "J2010-1323",
    "J2017+0603",
    "J2033+1734",
    "J2043+1711",
    "J2124-3358",
    "J2145-0750",
    "J2214+3000",
    "J2229+2643",
    "J2234+0611",
    "J2234+0944",
    "J2302+4442",
    "J2317+1439",
    "J2322+2057",
]


def get_psrname(file, name_sep="_"):
    return file.split("/")[-1].split(name_sep)[0]


noise_files = [f for f in noise_files if get_psrname(f, ".") in psr_list]

dir = "../NANOGrav_15yr_v1.0.1/narrowband/noise/"

NOISE_SAVE_FILE = "emir/emir_hasasia/noise_narrowband.json"

if os.path.exists(NOISE_SAVE_FILE):
    with open(NOISE_SAVE_FILE, "r") as f:
        noise = json.load(f)
else:
    noise = {}
    for psr in psr_list:
        keys = []
        with open(dir + psr + ".nb.pars.txt", "r") as f:
            for x in f:
                x = x.strip("\r\n")
                if x:
                    keys.append(x)

        all_elems = []
        with open(dir + psr + ".nb.chain_1.txt", "r") as f:
            for line in f:
                elems = line.strip("\r\n").split("\t")
                elems = [float(x) for x in elems]
                all_elems.append(elems)

        all_elems = np.array(all_elems)
        all_elems = all_elems.mean(axis=0)

        for k, v in zip(keys, all_elems):
            noise[k] = v

    with open(NOISE_SAVE_FILE, "w") as f:
        json.dump(noise, f)

print(noise)