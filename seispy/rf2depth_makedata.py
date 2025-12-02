"""
class RF2depth : process cal RF2depth
class sta_part, sta_full, _RFInd

"""

from os.path import join, exists
import argparse
import sys
import glob
from collections import namedtuple
from logging import Logger
from multiprocessing import Pool, cpu_count

import numpy as np

from seispy.core.depmodel import _load_mod, DepModel
from seispy.rfcorrect import (
    RFStation,
    psrf2depth,
    psrf_1D_raytracing,
    psrf_3D_migration,
    time2depth,
    psrf_3D_raytracing,
)
from seispy.core.pertmod import Mod3DPerturbation
import numpy as np
from seispy.ccppara import ccppara, CCPPara
from seispy.setuplog import SetupLog
from seispy.geo import latlon_from, rad2deg

# sta_part is not used
sta_part = namedtuple("sta_part", ["station", "stla", "stlo"])
sta_full = namedtuple("sta_full", ["station", "stla", "stlo", "stel"])


class Station(object):
    def __init__(self, sta_lst: str):
        """
        Read station list

        :param sta_lst: Path to station list
        :type sta_lst: string
        """
        dtype = {
            "names": ("station", "stla", "stlo", "stel"),
            "formats": ("U20", "f4", "f4", "f2"),
        }
        try:
            self.station, self.stla, self.stlo, self.stel = np.loadtxt(
                sta_lst, dtype=dtype, unpack=True, ndmin=1
            )
        except:
            dtype = {
                "names": ("station", "stla", "stlo"),
                "formats": ("U20", "f4", "f4"),
            }
            self.station, self.stla, self.stlo = np.loadtxt(
                sta_lst, dtype=dtype, unpack=True, ndmin=1
            )
            self.stel = np.zeros(self.stla.size)
        self.sta_num = self.stla.shape[0]

    def __getitem__(self, index):
        """
        allow for sta[index]
        """
        if hasattr(self, "stel"):
            return sta_full(
                self.station[index],
                self.stla[index],
                self.stlo[index],
                self.stel[index],
            )
        else:
            return sta_full(self.station[index], self.stla[index], self.stlo[index], 0)

    def __iter__(self):
        """
        allow for sta in stalist
        """
        for index in range(self.stla.shape[0]):
            if hasattr(self, "stel"):
                yield sta_full(
                    self.station[index],
                    self.stla[index],
                    self.stlo[index],
                    self.stel[index],
                )
            else:
                yield sta_full(
                    self.station[index], self.stla[index], self.stlo[index], 0
                )

    def __len__(self):
        return self.station.shape[0]


class RFDepth:
    """Convert receiver function to depth axis"""

    def __init__(
        self,
        cpara: CCPPara,
        log: Logger = SetupLog().RF2depthlog,
        raytracing3d=False,
        velmod3d=None,
        modfolder1d=None,
    ) -> None:
        """
        :param cpara: CCPPara object
        :type cpara: CCPPara
        :param log: Log object
        :type log: logging.Logger
        :param raytracing3d: If True, use 3D ray tracing to calculate the travel time
        :type raytracing3d: bool
        :param velmod3d: Path to 3D velocity model in npz file
        :type velmod3d: str
        :param modfolder1d: Folder path to 1D velocity model files with staname.vel as the file name
        :type modfolder1d: str
        """
        self.ismod1d = False
        self.cpara = cpara
        self.modfolder1d = modfolder1d
        self.log = log
        self.raytracing3d = raytracing3d
        self.velmod3d_path = None  # Store path instead of object
        if velmod3d is not None:
            if isinstance(velmod3d, str):
                self.velmod3d_path = velmod3d  # Store path for pickling
                self.mod3d = Mod3DPerturbation(
                    velmod3d, cpara.depth_axis, velmod=cpara.velmod
                )
            else:
                log.error("Path to 3d velocity model should be in str")
                sys.exit(1)
        elif modfolder1d is not None:
            if isinstance(modfolder1d, str):
                if exists(modfolder1d):
                    self.ismod1d = True
                else:
                    log.error("No such folder of {}".format(modfolder1d))
                    sys.exit(1)
            else:
                log.error("Folder to 1d velocity model files should be in str")
                sys.exit(1)
        else:
            self.ismod1d = True
        # Store path to rayp_lib instead of loading it (for picklability)
        if cpara.rayp_lib is not None:
            self.rayp_lib_path = cpara.rayp_lib
        else:
            self.rayp_lib_path = None

        self.sta_info = Station(cpara.stalist)
        self.rfdepth = []
        self._test_comp()

    def _test_comp(self):
        rfpath = join(self.cpara.rfpath, self.sta_info.station[0])
        self.prime_comp = ""
        for comp in ["R", "Q", "L", "Z"]:
            if glob.glob(join(rfpath, "*{}.sac".format(comp))):
                self.prime_comp = comp
                break
        if not self.prime_comp:
            raise FileNotFoundError(
                "No such any RF files in 'R'," "'Q', 'L', and 'Z' components"
            )

    def makedata(self, psphase=1, num_workers=None):
        """Convert receiver function to depth axis using parallel processing

        :param psphase: 1 for Ps, 2 for PpPs, 3 for PpSs
        :type psphase: int
        :param num_workers: Number of parallel workers (default: cpu_count()//2)
        :type num_workers: int
        """
        if num_workers is None:
            num_workers = max(1, cpu_count() // 2)

        # Prepare arguments for parallel processing
        # Pass all necessary data as picklable arguments
        args_list = []
        for _i, _sta in enumerate(self.sta_info):
            args_list.append(
                (
                    _i,
                    _sta,
                    psphase,
                    self.cpara.rfpath,
                    self.prime_comp,
                    self.cpara.depth_axis,
                    self.ismod1d,
                    self.modfolder1d,
                    self.cpara.velmod,
                    self.rayp_lib_path,  # Pass path instead of loaded object
                    self.raytracing3d,
                    self.velmod3d_path,  # Pass path instead of object
                    len(self.sta_info),
                )
            )

        self.log.info(
            f"Starting parallel processing with {num_workers} workers for {len(args_list)} stations"
        )

        # Process stations in parallel
        with Pool(processes=num_workers) as pool:
            results = pool.map(_process_single_station_worker, args_list)

        # Filter out None results (failed stations) and collect valid results
        self.rfdepth = [r for r in results if r is not None]

        self.log.info(
            f"Successfully processed {len(self.rfdepth)}/{len(args_list)} stations"
        )
        np.save(self.cpara.depthdat, self.rfdepth)


# Global worker function (outside class for picklability)
def _process_single_station_worker(args):
    """Process a single station (worker function for parallel processing)"""
    (
        _i,
        _sta,
        psphase,
        rfpath_base,
        prime_comp,
        depth_axis,
        ismod1d,
        modfolder1d,
        velmod_default,
        rayp_lib_path,
        raytracing3d,
        velmod3d_path,
        total_stations,
    ) = args

    # Load srayp in worker if needed (avoid pickling file handles)
    if rayp_lib_path is not None:
        srayp = np.load(rayp_lib_path)
    else:
        srayp = None

    rfpath = join(rfpath_base, _sta.station)

    try:
        stadatar = RFStation(rfpath, only_r=True, prime_comp=prime_comp)
        stadatar.stel = _sta.stel
        stadatar.stla = _sta.stla
        stadatar.stlo = _sta.stlo
        if stadatar.prime_phase == "P":
            sphere = True
        else:
            sphere = False
        print(
            f"Processing {_i + 1}/{total_stations}: {_sta.station} with {stadatar.ev_num} events"
        )
    except Exception as e:
        print(f"Error reading RF data for station {_sta.station}: {e}")
        return None

    #### 1d model for each station
    if ismod1d:
        if modfolder1d is not None:
            velmod = _load_mod(modfolder1d, _sta.station)
        else:
            velmod = velmod_default

        ps_rfdepth, end_index, x_s, _ = psrf2depth(
            stadatar,
            depth_axis,
            velmod=velmod,
            srayp=srayp,
            sphere=sphere,
            phase=psphase,
        )

        piercelat, piercelon = np.zeros_like(x_s, dtype=np.float64), np.zeros_like(
            x_s, dtype=np.float64
        )

        for j in range(stadatar.ev_num):
            piercelat[j], piercelon[j] = latlon_from(
                _sta.stla, _sta.stlo, stadatar.bazi[j], rad2deg(x_s[j])
            )
    else:
        ### 3d model interp - recreate the 3D model object from path
        if velmod3d_path is not None:
            mod3d = Mod3DPerturbation(velmod3d_path, depth_axis, velmod=velmod_default)
        else:
            mod3d = None

        if raytracing3d:
            pplat_s, pplon_s, pplat_p, pplon_p, newtpds = psrf_3D_raytracing(
                stadatar,
                depth_axis,
                mod3d,
                srayp=srayp,
                sphere=sphere,
                elevation=stadatar.stel,
            )
        else:
            pplat_s, pplon_s, pplat_p, pplon_p, raylength_s, raylength_p, tps = (
                psrf_1D_raytracing(
                    stadatar,
                    depth_axis,
                    velmod=velmod_default,
                    srayp=srayp,
                    sphere=sphere,
                    phase=psphase,
                )
            )
            newtpds = psrf_3D_migration(
                pplat_s,
                pplon_s,
                pplat_p,
                pplon_p,
                raylength_s,
                raylength_p,
                tps,
                depth_axis,
                mod3d,
            )
        if stadatar.prime_phase == "P":
            piercelat, piercelon = pplat_s, pplon_s
        else:
            piercelat, piercelon = pplat_p, pplon_p
        ps_rfdepth, end_index = time2depth(stadatar, depth_axis, newtpds)

    # Create rfdep dictionary
    rfdep = {}
    rfdep["station"] = stadatar.staname
    rfdep["stalat"] = stadatar.stla
    rfdep["stalon"] = stadatar.stlo
    rfdep["depthrange"] = depth_axis
    rfdep["bazi"] = stadatar.bazi
    rfdep["rayp"] = stadatar.rayp
    rfdep["moveout_correct"] = ps_rfdepth
    rfdep["piercelat"] = piercelat
    rfdep["piercelon"] = piercelon
    rfdep["stopindex"] = end_index

    return rfdep


def rf2depth():
    """
    CLI for Convert receiver function to depth axis
    There's  4 branch provided to do RF 2 depth conversion

    1. only -d :do moveout correction
    2. only -r : do raytracing but no moveout correction
    3. -d and -r : do moveout correction and raytracing
    4. -m : use {staname}.vel file for RF2depth conversion

    """
    parser = argparse.ArgumentParser(description="Convert Ps RF to depth axis")
    parser.add_argument(
        "-d",
        help="Path to 3d vel model in npz file for moveout correcting",
        metavar="3d_velmodel_path",
        type=str,
        default="",
    )
    parser.add_argument(
        "-m",
        help="Folder path to 1d vel model files with staname.vel as the file name",
        metavar="1d_velmodel_folder",
        type=str,
        default="",
    )
    parser.add_argument(
        "-r",
        help="Path to 3d vel model in npz file for 3D ray tracing",
        metavar="3d_velmodel_path",
        type=str,
        default="",
    )
    parser.add_argument(
        "cfg_file", help="Path to configure file", metavar="ccp.cfg", type=str
    )
    arg = parser.parse_args()
    cpara = ccppara(arg.cfg_file)
    if arg.d != "" and arg.r != "":
        #### print help issue
        raise ValueError("Specify only 1 argument in '-d' and '-r'")
    elif arg.d != "" and arg.r == "" and arg.m == "":
        #### do 3d moveout correction but 1d rf2depth
        raytracing3d = False
        velmod3d = arg.d
        modfolder1d = None
    elif arg.d == "" and arg.r != "" and arg.m == "":
        #### do 3d raytraying
        raytracing3d = True
        velmod3d = arg.d
        modfolder1d = None
    elif arg.d == "" and arg.r == "" and arg.m != "":
        #### use multiple 1d velmod for time2depth convertion
        raytracing3d = False
        velmod3d = None
        modfolder1d = arg.m
    else:
        ### all last, use default 1D vel model do time2depth conversion
        raytracing3d = False
        velmod3d = None
        modfolder1d = None
    rfd = RFDepth(
        cpara,
        raytracing3d=raytracing3d,
        velmod3d=velmod3d,
        modfolder1d=modfolder1d,
    )
    rfd.makedata()


if __name__ == "__main__":
    pass
