import numpy as np
import csv
import subprocess
import re
import os
import sys
from scipy.stats import chi, norm
import math
import platform


def create_intervals(dims, no_bins, latent_range):    
    partition_density = 1.0 / no_bins
    density = 0
    parts = list()
    for i in range(no_bins + 1):
        if density > 1:
            density = 1

        rv = norm.ppf(density)

        if rv == -np.inf:
            rv = latent_range[0]
        elif rv > latent_range[1]:
            rv = latent_range[1]
            
        parts.append(rv)
        density += partition_density
    
    return np.array(parts)


def measure_coverage(feature_array, acts, ways=3, timeout=10, suffix="temp"):
    # write feature array to CSV
    csv_header = ",".join([f"p{i+1}" for i in range(feature_array.shape[1])])
    csv_file = f"CA_{suffix}.csv"
    np.savetxt(csv_file, feature_array, delimiter=",", header=csv_header, fmt='%d')

    # classpath separator
    cp_sep = ";" if platform.system() == "Windows" else ":"

    # compiled Java classes directory
    bin_dir = os.path.join("tools", "ccm", "CCMCL", "bin")

    # jars in res/
    res_dir = os.path.join("tools", "ccm", "CCMCL", "res")
    jars = [
        "jfreechart-1.0.19.jar",
        "jcommon-1.0.23.jar",
        "choco-solver-2.1.5.jar",
        "JavaPlot.jar"
    ]
    classpath = cp_sep.join([bin_dir] + [os.path.join(res_dir, j) for j in jars])

    # build Java command
    cmd = [
        "timeout", str(timeout),
        "java", "-cp", classpath,
        "com.nist.ccmcl.Main",
        "-A", str(acts),
        "-I", str(csv_file),
        "-T", str(ways)
    ]

    p1 = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p1.communicate()

    stdout_str = out.decode(errors="ignore")
    stderr_str = err.decode(errors="ignore")

    if stderr_str.strip():
        print("⚠️ Java STDERR:\n", stderr_str)

    rex = re.compile(r"Total\s+\d+-way coverage:\s+([0-9.]+)")
    match = rex.search(stdout_str)

    if not match:
        raise RuntimeError(
            f"Could not parse coverage from Java output.\nSTDOUT:\n{stdout_str}\nSTDERR:\n{stderr_str}"
        )

    return float(match.group(1))


def create_acts(k, v):
    acts = f"Config/{k}params_{v}bins.txt"
    subprocess.call(['./create_acts.sh', "IDC", str(k), '1', str(v), str(acts)])
    assert os.path.exists(acts), "acts not generated"
    return acts


def generate_array(latent, density, no_bins=10):
    radin, radout = chi.interval(density, latent.shape[1])
    latent_range = (-radout, radout)
    intervals = create_intervals(latent.shape[1], no_bins, latent_range)

    x_squares = np.square(latent)
    radius_vector = np.sqrt(np.sum(x_squares, axis=1)).reshape(-1, 1)
    latent = latent[(radius_vector >= radin).reshape(-1)]
    x_squares = np.square(latent)
    radius_vector = np.sqrt(np.sum(x_squares, axis=1)).reshape(-1, 1)
    latent = latent[(radius_vector <= radout).reshape(-1)]

    cov_array = np.digitize(latent[:, 0], intervals).reshape(-1, 1)
    for i in range(1, latent.shape[1]):
        cov_vector = np.digitize(latent[:, i], intervals).reshape(-1, 1)
        cov_array = np.concatenate((cov_array, cov_vector), axis=1)

    return cov_array, latent.shape[0], ([], [])
