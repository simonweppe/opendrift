# https://github.com/MireyaMMO/cLCS/blob/main/cLCS/make_cLCS.py

import numpy as np
from scipy.integrate import solve_ivp
from calendar import monthrange
import pickle
import glob
import logging
import logging.config

logging.basicConfig(level=logging.INFO)


class compute_cLCS_squeezelines(object):
    """
    Loads averaged C-G, and compute squeezelines

    Parameters:
    -----------
    dirr: str
      directory where the climatological files are located, same dirr as used in mean_C
    month_or_id: str
      month to be analysed or id
    """

    def __init__(
        self,
        # dirr,
        # month_or_id,
        ds_lcs ,
        arclength=150,
        nxb=25,
        nyb=25,
        climatology=True,
        files=None,
    ):
        # self.dirr = dirr
        # self.month_or_id = month_or_id
        self.ds_lcs = ds_lcs
        self.arclength = arclength
        self.nxb = nxb
        self.nyb = nyb
        self.climatology= climatology
        self.files = files
        self.logger = logging

    def squeezeline(self, C11, C12, C22, xi, yi, ArcLength):
        """
        SQUEEZELINE Line field intergrator.
        Computes squeezelines from the Cauchy-Green tensor
        Parameters:
        ----------
        C11, C12 and C22 which are a function of (X,Y)

        Returns:
        --------
        [XS, YS]: np.array
            Each column of XS, YS is a squeezeline
        """
        detC = (C11 * C22) - (C12**2)
        trC = C11 + C22
        lda1 = np.real(0.5 * trC - np.sqrt(0.25 * trC**2 - detC))
        lda2 = np.real(0.5 * trC + np.sqrt(0.25 * trC**2 - detC))
        xi2x = np.real(-C12 / np.sqrt((C11 - lda2) ** 2 + (C12**2)))
        xi2y = np.real((C11 - lda2) / np.sqrt((C11 - lda2) ** 2 + (C12**2)))
        xi1x = -xi2y
        xi1y = +xi2x
        v = (lda1 - lda2) ** 2 / (lda1 + lda2) ** 2
        vx = xi1x * v
        vy = xi1y * v
        self.eta_1 = np.copy(vx)
        self.eta_2 = np.copy(vy)
        Nxb = self.nxb  # % boxes in x
        Nyb = self.nyb  # % boxes in y
        X0, Y0 = np.meshgrid(xi, yi)
        X0 = X0.ravel()
        Y0 = Y0.ravel()
        xblim = np.linspace(xi.min(), xi.max(), Nxb + 1)  # ; % box limits
        yblim = np.linspace(yi.min(), yi.max(), Nyb + 1)  # ); % box limits
        xs0 = []
        ys0 = []
        for ixb in range(Nxb):
            for iyb in range(Nyb):
                lda2b = np.copy(lda2)
                Ixb = np.where((xi < xblim[ixb]) | (xi > xblim[ixb + 1]))
                Iyb = np.where((yi < yblim[iyb]) | (yi > yblim[iyb + 1]))
                lda2b[Iyb, :] = np.nan
                lda2b[:, Ixb] = np.nan
                lda2bIb = np.nanmax(lda2b.ravel())
                if ~np.isnan(lda2bIb):
                    Ib = np.where(lda2b.ravel() == lda2bIb)
                    xs0 = np.append(xs0, X0[Ib])
                    ys0 = np.append(ys0, Y0[Ib])
        self.x0 = np.copy(xs0)
        self.y0 = np.copy(ys0)
        if xi.shape != self.eta_1.shape:
            [self.xi, self.yi] = np.meshgrid(xi, yi)

        self.x0 = self.x0.ravel()
        self.y0 = self.y0.ravel()
        [self.m, self.n] = self.xi.shape
        self.Np = len(self.x0)
        self.nn = 0
        x = self.xi[0, :]
        self.dx = np.abs(x[1] - x[0])
        y = self.yi[:, 0]
        self.dy = np.abs(y[1] - y[0])

        self.Xmin = np.min(x)
        self.Ymin = np.min(y)
        self.Xmax = np.max(x)
        self.Ymax = np.max(y)

        self.eta_1[np.where(np.imag(self.eta_1) != 0)] = np.nan
        self.eta_2[np.where(np.imag(self.eta_2) != 0)] = np.nan

        [self.xi1_1b, self.xi1_2b] = self.SmoothVectorField(
            self.x0, self.y0, self.Xmin, self.Ymin, self.dx, self.dy
        )

        self.xi1_1b[np.where(np.isnan(self.xi1_1b))] = 0
        self.xi1_2b[np.where(np.isnan(self.xi1_2b))] = 0
        # make sure that all tensor lines will launch in the same direction
        nonull = np.where((self.xi1_1b != 0) & (self.xi1_2b != 0))[0][0]
        sgn_0 = np.sign(
            self.xi1_1b[nonull] * self.xi1_1b + self.xi1_2b[nonull] * self.xi1_2b
        )
        self.xi1_1b = sgn_0 * self.xi1_1b
        self.xi1_2b = sgn_0 * self.xi1_2b
        t_eval = np.arange(ArcLength[0], ArcLength[1], 1)

        r = solve_ivp(
            self.fun,
            [ArcLength[0], ArcLength[1]],
            np.hstack((self.x0, self.y0)),
            method="RK45",
            dense_output=True,
            t_eval=t_eval,
            rtol=1e-6,
            atol=1e-6,
        )
        pxt = r.y[0 : self.Np, :]
        pyt = r.y[self.Np : :, :]
        return pxt, pyt

    def fun(self, t, y):
        self.nn = self.nn + 1
        [xi1_1, xi1_2] = self.SmoothVectorField(
            y[0 : self.Np], y[self.Np : :], self.Xmin, self.Ymin, self.dx, self.dy
        )
        xi1_1[np.where(np.isnan(xi1_1))] = 0
        xi1_2[np.where(np.isnan(xi1_2))] = 0
        sgn_2 = np.sign(xi1_1 * self.xi1_1b + xi1_2 * self.xi1_2b)
        xi1_1 = sgn_2 * xi1_1
        xi1_2 = sgn_2 * xi1_2
        DY = np.zeros(self.Np * 2)  # a column vector
        DY[0 : self.Np] = xi1_1
        DY[self.Np : :] = xi1_2

        DY[np.isnan(DY)] = 0

        self.xi1_1b = np.copy(xi1_1)
        self.xi1_2b = np.copy(xi1_2)
        return DY

    def SmoothVectorField(self, x0, y0, Xmin, Ymin, dx, dy):
        id1_UL = np.floor((y0 - Ymin) / dy)
        id2_UL = np.floor((x0 - Xmin) / dx)
        i_UL, j_UL = self.safe_sub2ind([self.m, self.n], id1_UL, id2_UL)

        id1_UR = np.copy(id1_UL)
        id2_UR = id2_UL + 1
        i_UR, j_UR = self.safe_sub2ind([self.m, self.n], id1_UR, id2_UR)

        id1_DL = id1_UL + 1
        id2_DL = np.copy(id2_UL)
        i_DL, j_DL = self.safe_sub2ind([self.m, self.n], id1_DL, id2_DL)

        id1_DR = id1_UL + 1
        id2_DR = id2_UL + 1
        i_DR, j_DR = self.safe_sub2ind([self.m, self.n], id1_DR, id2_DR)

        v1_UL = self.eta_1[i_UL, j_DR]
        v1_UR = self.eta_1[i_UR, j_UR]
        v1_DL = self.eta_1[i_DL, j_DL]
        v1_DR = self.eta_1[i_DR, j_DR]

        v2_UL = self.eta_2[i_UL, j_DR]
        v2_UR = self.eta_2[i_UR, j_UR]
        v2_DL = self.eta_2[i_DL, j_DL]
        v2_DR = self.eta_2[i_DR, j_DR]

        sgn_1 = np.sign(v1_UL * v1_UR + v2_UL * v2_UR)
        v1_UR = sgn_1 * v1_UR
        v2_UR = sgn_1 * v2_UR

        sgn_1 = np.sign(v1_UL * v1_DL + v2_UL * v2_DL)
        v1_DL = sgn_1 * v1_DL
        v2_DL = sgn_1 * v2_DL

        sgn_1 = np.sign(v1_UL * v1_DR + v2_UL * v2_DR)
        v1_DR = sgn_1 * v1_DR
        v2_DR = sgn_1 * v2_DR
        # -- Bilinear interpolation
        # Bilinear interpolation for v1
        c1 = (self.xi[i_UR, j_UR] - x0) / dx
        c2 = (x0 - self.xi[i_UL, j_UL]) / dx
        c3 = (self.yi[i_DL, j_DL] - y0) / dy
        c4 = (y0 - self.yi[i_UL, j_UL]) / dy

        v1_0 = c3 * (c1 * v1_UL + c2 * v1_UR) + c4 * (c1 * v1_DL + c2 * v1_DR)
        # Bilinear interpolation for v2
        c1 = (self.xi[i_UR, j_UR] - x0) / dx
        c2 = (x0 - self.xi[i_UL, j_UL]) / dx
        c3 = (self.yi[i_DL, j_DL] - y0) / dy
        c4 = (y0 - self.yi[i_UL, j_UL]) / dy

        v2_0 = c3 * (c1 * v2_UL + c2 * v2_UR) + c4 * (c1 * v2_DL + c2 * v2_DR)

        # %-- Normalizing v
        norm_v = np.sqrt(v1_0**2 + v2_0**2)
        norm_v[np.where(norm_v == 0)] = 1
        v1_0 = v1_0 / norm_v
        v2_0 = v2_0 / norm_v
        return v1_0, v2_0

    def safe_sub2ind(self, sz, rr, cc):
        rr[np.where(rr < 1)] = 0
        rr[np.where(rr > sz[0] - 1)] = sz[0] - 1
        rr = rr.astype("int")
        cc[np.where(cc < 1)] = 0
        cc[np.where(cc > sz[1] - 1)] = sz[1] - 1
        cc = cc.astype("int")
        return [rr, cc]

    def run(self):
       
        # Note in Mireya's code
        # C11total, C12total, C22total are accumulated values for the 
        # particle releases deployed for the climatolgical LCS calculation
        # see https://github.com/MireyaMMO/cLCS/blob/main/cLCS/mean_C.py#L82
        # 
        # so when used in "climatological" mode she's just doing a time-average of these

        # N = len(self.ds_lcs.time) # how many time step
        C11 = self.ds_lcs.A_C11.mean(dim='time') 
        C22 = self.ds_lcs.A_C22.mean(dim='time') 
        C12 = self.ds_lcs.A_C12.mean(dim='time') 
        ArcLength = self.arclength # in meters
        # xspan,yspan are defined in Mireya's code :  https://github.com/MireyaMMO/cLCS/blob/main/cLCS/mean_C.py#L288
        # x = np.arange(0, xmax*1e-3, self.dx0)
        # y = np.arange(0, ymax*1e-3, self.dy0)
        # self.xspan, self.yspan = np.meshgrid(x,y)
        # >> where self.dx0,self.dy0 are in kilometers, see https://github.com/MireyaMMO/cLCS/blob/main/examples/01_cLCS_ROMS.ipynb 

        xspan = self.ds_lcs.X[0,:]-self.ds_lcs.X[0,0] * 1e-3 # X-grid, relative to 0, in kilometers in Mireya's code
        yspan = self.ds_lcs.Y[:,0]-self.ds_lcs.Y[0,0] * 1e-3 # Y-grid, relative to 0, in kilometers in Mireya's code
        # now compute squeezelines
        self.pxt, self.pyt = self.squeezeline(C11, C12, C22, xspan, yspan, [0, ArcLength])
        zeros = np.where((np.diff(self.pxt,1)[:,0]==0) | (np.diff(self.pyt,1)[:,0]==0))[0]
        self.pxt = np.delete(self.pxt, zeros, 0)
        self.pyt = np.delete(self.pyt, zeros, 0)        

        # original code
        if False:
            if isinstance(self.month_or_id, int):
                self.month_or_id = "%02d" % self.month_or_id
        
            month_dirr = self.dirr + self.month_or_id + "/"
            if self.climatology:
                (
                        _,
                        _,
                        _,
                        _,
                        _,
                        _,
                        C11total,
                        C22total,
                        C12total,
                        xspan,
                        yspan,
                        count,
                ) = pickle.load(open(f"{month_dirr}TOT-{self.month_or_id}.p", "rb"))
                N = count
                C11 = C11total / N
                C22 = C22total / N
                C12 = C12total / N
                ArcLength = self.arclength
                self.pxt, self.pyt = self.squeezeline(
                        C11, C12, C22, xspan[-1, :], yspan[:, -1], [0, ArcLength]
                )
                zeros = np.where((np.diff(self.pxt,1)[:,0]==0) | (np.diff(self.pyt,1)[:,0]==0))[0]
                self.pxt = np.delete(self.pxt, zeros, 0)
                self.pyt = np.delete(self.pyt, zeros, 0)
                pickle.dump([self.pxt, self.pyt], open(f"{month_dirr}/cLCS_{self.month_or_id}.p", "wb"))

            else: 
                if not self.files:
                    self.files = sorted(glob.glob(f"{month_dirr}/LCS_{self.month_or_id}*-CG.p"))
                for file in self.files:
                    outfile = file.split('-')[0]
                    (
                                _,
                                _,
                                _,
                                _,
                                _,
                                _,
                                C11,
                                C22,
                                C12,
                                xspan,
                                yspan,
                        ) = pickle.load(open(file, "rb"))
                    ArcLength = self.arclength
                    self.pxt, self.pyt = self.squeezeline(
                            C11, C12, C22, xspan[-1, :], yspan[:, -1], [0, ArcLength]
                    )
                    zeros = np.where(np.diff(self.pxt,1)[:,0]==0)[0]
                    self.pxt = np.delete(self.pxt, zeros, 0)
                    self.pyt = np.delete(self.pyt, zeros, 0)
                    pickle.dump([self.pxt, self.pyt], open(f"{outfile}.p", "wb"))