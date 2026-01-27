import math
import numpy as np


class distaz:
    """
    Subroutine to calculate the Great Circle Arc distance
        between two sets of geographic coordinates

    Equations take from Bullen, pages 154, 155

    T. Owens, September 19, 1991
              Sept. 25 -- fixed az and baz calculations
    P. Crotwell, Setember 27, 1995
    Converted to c to fix annoying problem of fortran giving wrong
    answers if the input doesn't contain a decimal point.

    H. P. Crotwell, September 18, 1997
    Java version for direct use in java programs.
    *
    * C. Groves, May 4, 2004
    * Added enough convenience constructors to choke a horse and made public double
    * values use accessors so we can use this class as an immutable

    H.P. Crotwell, May 31, 2006
    Port to python, thus adding to the great list of languages to which
    distaz has been ported from the origin fortran: C, Tcl, Java and now python
    and I vaguely remember a perl port. Long live distaz!

    Mijian Xu, Jan 01, 2016
    Add np.ndarray to available input.

    Shihchi Shao, Jan 2026
    Fix compatibility with NumPy 2.0+ where np.where on 0d arrays raises error.
    """

    def __init__(self, lat1, lon1, lat2, lon2):

        self.stalat = lat1
        self.stalon = lon1
        self.evtlat = lat2
        self.evtlon = lon2

        rad = 2.0 * math.pi / 360.0
        sph = 1.0 / 298.257

        scolat = math.pi / 2.0 - np.arctan(
            (1.0 - sph) * (1.0 - sph) * np.tan(lat1 * rad)
        )
        ecolat = math.pi / 2.0 - np.arctan(
            (1.0 - sph) * (1.0 - sph) * np.tan(lat2 * rad)
        )
        slon = lon1 * rad
        elon = lon2 * rad

        a = np.sin(scolat) * np.cos(slon)
        b = np.sin(scolat) * np.sin(slon)
        c = np.cos(scolat)
        d = np.sin(slon)
        e = -np.cos(slon)
        g = -c * e
        h = c * d
        k = -np.sin(scolat)

        aa = np.sin(ecolat) * np.cos(elon)
        bb = np.sin(ecolat) * np.sin(elon)
        cc = np.cos(ecolat)
        dd = np.sin(elon)
        ee = -np.cos(elon)
        gg = -cc * ee
        hh = cc * dd
        kk = -np.sin(ecolat)

        delrad = np.arccos(a * aa + b * bb + c * cc)
        self.delta = delrad / rad

        rhs1 = (aa - d) * (aa - d) + (bb - e) * (bb - e) + cc * cc - 2.0
        rhs2 = (aa - g) * (aa - g) + (bb - h) * (bb - h) + (cc - k) * (cc - k) - 2.0
        dbaz = np.arctan2(rhs1, rhs2)

        # Use np.where with 3 arguments (ternary form) to avoid nonzero() on 0d arrays
        dbaz = np.where(dbaz < 0.0, dbaz + 2 * math.pi, dbaz)

        self.baz = dbaz / rad

        rhs1 = (a - dd) * (a - dd) + (b - ee) * (b - ee) + c * c - 2.0
        rhs2 = (a - gg) * (a - gg) + (b - hh) * (b - hh) + (c - kk) * (c - kk) - 2.0
        daz = np.arctan2(rhs1, rhs2)

        # Use np.where with 3 arguments (ternary form) to avoid nonzero() on 0d arrays
        daz = np.where(daz < 0.0, daz + 2 * math.pi, daz)

        self.az = daz / rad

        # Make sure 0.0 is always 0.0, not 360.
        # Use np.where with 3 arguments (ternary form) for NumPy 2.0+ compatibility
        self.baz = np.where(np.abs(self.baz - 360.0) < 0.00001, 0.0, self.baz)
        self.baz = np.where(np.abs(self.baz) < 0.00001, 0.0, self.baz)
        self.az = np.where(np.abs(self.az - 360.0) < 0.00001, 0.0, self.az)
        self.az = np.where(np.abs(self.az) < 0.00001, 0.0, self.az)

        # Handle case where lat1 == lat2 and lon1 == lon2
        same_location = (lat1 == lat2) & (lon1 == lon2)
        self.delta = np.where(same_location, 0.0, self.delta)
        self.az = np.where(same_location, 0.0, self.az)
        self.baz = np.where(same_location, 0.0, self.baz)

    def getDelta(self):
        return self.delta

    def getAz(self):
        return self.az

    def getBaz(self):
        return self.baz

    def degreesToKilometers(self):
        return self.delta * 111.19


if __name__ == "__main__":
    ela = 1
    elo = 1
    sla = 2
    slo = 1
    da = distaz(ela, elo, sla, slo)
    print(da.baz)
