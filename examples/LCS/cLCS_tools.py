# Tools to compute Compute Cauchy–Green strain tensorlines aka "squeezelines"
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
    Compute Cauchy-Green strain tensorline aka squeezelines()

    Parameters:
    -----------
    ds_lcs : xarray.Dataset
        xarray dataset output from `o.calculate_green_cauchy_tensor()`.
    arclength : int, optional
        Length of the arc along the streamline for LCS computation (default is 150).
    nxb : int, optional
        Number of grid points in the x-direction for squeezeline derivation (default is 25).
    nyb : int, optional
        Number of grid points in the y-direction for squeezeline derivation (default is 25).
    
    (nxb,nyb) will govern the squeezeline density, and arclength their length


    Adapted from cLCS code :
    https://github.com/MireyaMMO/cLCS/blob/main/cLCS/make_cLCS.py
    
    https://github.com/MireyaMMO/cLCS/tree/main
 
    """

    def __init__(
        self,
        ds_lcs , # xarray dataset output from o.calculate_green_cauchy_tensor()
        arclength=150,
        nxb=25,
        nyb=25,
    ):
        self.ds_lcs = ds_lcs
        self.arclength = arclength
        self.nxb = nxb
        self.nyb = nyb
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
        # particle releases deployed for the climatolgical LCS calculation.
        # She then divides them by N, bn of time steps
        # see https://github.com/MireyaMMO/cLCS/blob/main/cLCS/mean_C.py#L82
        # 
        # Here we have an xarray dataset so we can just average below
        
        # prepare inputs for squeezeline computations
        C11 = self.ds_lcs.A_C11.mean(dim='time') 
        C22 = self.ds_lcs.A_C22.mean(dim='time') 
        C12 = self.ds_lcs.A_C12.mean(dim='time') 
        # xspan,yspan are defined in Mireya's code :  https://github.com/MireyaMMO/cLCS/blob/main/cLCS/mean_C.py#L288
        # x = np.arange(0, xmax*1e-3, self.dx0)
        # y = np.arange(0, ymax*1e-3, self.dy0)
        # self.xspan, self.yspan = np.meshgrid(x,y)
        # >> where self.dx0,self.dy0 are in kilometers, see https://github.com/MireyaMMO/cLCS/blob/main/examples/01_cLCS_ROMS.ipynb 
        # xspan = (self.ds_lcs.X[0,:]-self.ds_lcs.X[0,0]) #* 1e-3 # X-grid, relative to 0, in kilometers in Mireya's code
        # yspan = (self.ds_lcs.Y[:,0]-self.ds_lcs.Y[0,0]) #* 1e-3 # Y-grid, relative to 0, in kilometers in Mireya's code
        
        # here we just keep the original X,Y. This fits well with the LCS maps.
        # Note we take the first time if ds_lcs has multiple times
        # 
        xspan = (self.ds_lcs.isel(time=0).X[0,:]) #* 1e-3 # X-grid, in meters
        yspan = (self.ds_lcs.isel(time=0).Y[:,0]) #* 1e-3 # Y-grid, in meters 
        ArcLength = self.arclength # in meters. i.e. the number of segments of each line

        # now compute Cauchy–Green strain tensorlines aka "squeezelines"
        self.pxt, self.pyt = self.squeezeline(C11, C12, C22, xspan, yspan, [0, ArcLength])
        # clean squeezelines
        zeros = np.where((np.diff(self.pxt,1)[:,0]==0) | (np.diff(self.pyt,1)[:,0]==0))[0]
        self.pxt = np.delete(self.pxt, zeros, 0)
        self.pyt = np.delete(self.pyt, zeros, 0)        

        # add squeezeline color-coding data
        # 
        # in cLCS, squeezeline are color coded using np.log(sqrtlda2total / N) which is equivalent np.log(mean(sqrtlda2total))
        # https://github.com/MireyaMMO/cLCS/blob/main/cLCS/plotting_cLCS.py#L81C9-L81C38
        lda2 = self.ds_lcs.A_lda2.mean(dim='time')
        logsqrtlda2 = np.log( np.sqrt(self.ds_lcs.A_lda2).mean(dim='time') ) # in Mireya's code: z=np.log(sqrtlda2total / N)
        from scipy.interpolate import griddata
        self.pzt = griddata(np.array([ self.ds_lcs.isel(time=0).X.values.ravel(), self.ds_lcs.isel(time=0).Y.values.ravel()]).T, 
                                    logsqrtlda2.values.ravel(), 
                                    (self.pxt, self.pyt), 
                                    method='linear')

def get_colourmap(name):
    from matplotlib.colors import ListedColormap, LinearSegmentedColormap
    if name == "Zissou":
        colors = [
            (0.98, 0.98, 0.95),
            (0.23, 0.60, 0.69),
            (0.47, 0.71, 0.77),
            (0.92, 0.8, 0.16),
            (0.88, 0.68, 0),
            (0.95, 0.10, 0),
            (0.79, 0.08, 0),
        ]  # R -> G -> B
        cmap = LinearSegmentedColormap.from_list(name, colors, N=200)
    elif name == "BlueOrange":
        top = cm.get_cmap("Oranges", 128)  # r means reversed version
        bottom = cm.get_cmap("Blues_r", 128)  # combine it all
        colors = np.vstack(
            (bottom(np.linspace(0, 1, 128)), top(np.linspace(0, 1, 128)))
        )  # create a new colormaps with a name of OrangeBlue
        cmap = ListedColormap(colors, name)
    elif name == "Color_blind_1":
        colors = [
            (0.67, 0.34, 0.11),
            (0.89, 0.61, 0.34),
            (1, 0.87, 0.67),
            (0.67, 0.72, 0.86),
            (0.30, 0.45, 0.71),
        ]  # R -> G -> B
        cmap = LinearSegmentedColormap.from_list(name, colors, N=200)
    elif name == "Duran_cLCS":
        colors = np.array(
            [
                [1.0000, 1.0000, 0.9987, 1],
                [0.9971, 1.0000, 0.9970, 1],
                [0.9896, 1.0000, 0.9931, 1],
                [0.9771, 1.0000, 0.9871, 1],
                [0.9593, 0.9900, 0.9789, 1],
                [0.9364, 0.9708, 0.9686, 1],
                [0.9084, 0.9484, 0.9564, 1],
                [0.8759, 0.9243, 0.9422, 1],
                [0.8395, 0.8994, 0.9264, 1],
                [0.8000, 0.8749, 0.9092, 1],
                [0.7585, 0.8516, 0.8906, 1],
                [0.7160, 0.8301, 0.8710, 1],
                [0.6738, 0.8110, 0.8506, 1],
                [0.6330, 0.7948, 0.8296, 1],
                [0.5949, 0.7817, 0.8081, 1],
                [0.5606, 0.7719, 0.7865, 1],
                [0.5310, 0.7657, 0.7649, 1],
                [0.5073, 0.7628, 0.7435, 1],
                [0.4900, 0.7633, 0.7225, 1],
                [0.4798, 0.7671, 0.7019, 1],
                [0.4771, 0.7737, 0.6821, 1],
                [0.4819, 0.7831, 0.6629, 1],
                [0.4943, 0.7946, 0.6446, 1],
                [0.5138, 0.8081, 0.6271, 1],
                [0.5399, 0.8230, 0.6105, 1],
                [0.5720, 0.8387, 0.5948, 1],
                [0.6090, 0.8548, 0.5800, 1],
                [0.6500, 0.8708, 0.5659, 1],
                [0.6938, 0.8860, 0.5525, 1],
                [0.7391, 0.9000, 0.5398, 1],
                [0.7847, 0.9120, 0.5275, 1],
                [0.8292, 0.9217, 0.5155, 1],
                [0.8716, 0.9284, 0.5037, 1],
                [0.9108, 0.9317, 0.4918, 1],
                [0.9457, 0.9310, 0.4797, 1],
                [0.9756, 0.9260, 0.4672, 1],
                [1.0000, 0.9162, 0.4541, 1],
                [1.0000, 0.9013, 0.4401, 1],
                [1.0000, 0.8810, 0.4251, 1],
                [1.0000, 0.8551, 0.4089, 1],
                [1.0000, 0.8235, 0.3912, 1],
                [1.0000, 0.7862, 0.3720, 1],
                [1.0000, 0.7432, 0.3511, 1],
                [1.0000, 0.6947, 0.3284, 1],
                [1.0000, 0.6408, 0.3039, 1],
                [1.0000, 0.5821, 0.2775, 1],
                [0.9900, 0.5190, 0.2494, 1],
                [0.9819, 0.4521, 0.2195, 1],
                [0.9765, 0.3822, 0.1882, 1],
                [0.9744, 0.3102, 0.1556, 1],
                [0.9756, 0.2372, 0.1222, 1],
                [0.9799, 0.1643, 0.0884, 1],
                [0.9864, 0.0931, 0.0547, 1],
                [0.9938, 0.0251, 0.0219, 1],
                [1.0000, 0, 0, 1],
                [1.0000, 0, 0, 1],
                [0.9989, 0, 0, 1],
                [0.9858, 0, 0, 1],
                [0.9601, 0, 0, 1],
                [0.9194, 0, 0, 1],
                [0.8618, 0, 0, 1],
                [0.7874, 0, 0, 1],
                [0.6982, 0, 0, 1],
                [0.6000, 0.0069, 0.0013, 1],
            ]
        )
        cmap = LinearSegmentedColormap.from_list(name, colors)
    elif name == "RedYellowBlue":
        colors = [
            (0.843, 0.188, 0.153),
            (0.988, 0.553, 0.349),
            (0.996, 0.878, 0.565),
            (0.569, 0.749, 0.859),
            (0.271, 0.459, 0.706),
        ]  # R -> G -> B
        cmap = LinearSegmentedColormap.from_list(name, colors, N=200)
    elif name == "BlueYellowRed":
        colors = [
            (0.843, 0.188, 0.153),
            (0.988, 0.553, 0.349),
            (0.996, 0.878, 0.565),
            (0.569, 0.749, 0.859),
            (0.271, 0.459, 0.706),
        ]  # R -> G -> B
        cmap = LinearSegmentedColormap.from_list(name, colors[::-1], N=200)
    elif name == "AlgaeSalmon":
        colors = [
            (0.557, 0.792, 0.902),
            (0.165, 0.616, 0.561),
            (0.914, 0.769, 0.416),
            (0.957, 0.635, 0.38),
            (0.906, 0.435, 0.318),
        ]  # R -> G -> B
        cmap = LinearSegmentedColormap.from_list(name, colors, N=200)
    elif name == "OceanSun":
        colors = [
            (0.0, 0.188, 0.286),
            (0.839, 0.157, 0.157),
            (0.969, 0.498, 0),
            (0.988, 0.749, 0.286),
            (0.918, 0.886, 0.718),
        ]
        cmap = LinearSegmentedColormap.from_list(name, colors, N=200)
    elif name == "SunOcean":
        colors = [
            (0.0, 0.188, 0.286),
            (0.839, 0.157, 0.157),
            (0.969, 0.498, 0),
            (0.988, 0.749, 0.286),
            (0.918, 0.886, 0.718),
        ]
        cmap = LinearSegmentedColormap.from_list(name, colors[::-1], N=200)
    elif name == "RedBlue":
        colors = [
            (0.792, 0.0, 0.125),
            (0.957, 0.647, 0.511),
            (0.969, 0.969, 0.969),
            (0.573, 0.773, 0.871),
            (0.024, 0.439, 0.690),
        ]  # R -> G -> B
        cmap = LinearSegmentedColormap.from_list(name, colors, N=200)
    elif name == "BlueRed":
        colors = [
            (0.792, 0.0, 0.125),
            (0.957, 0.647, 0.511),
            (0.969, 0.969, 0.969),
            (0.573, 0.773, 0.871),
            (0.024, 0.439, 0.690),
        ]  # R -> G -> B
        cmap = LinearSegmentedColormap.from_list(name, colors[::-1], N=200)
    elif name == "PurpleOrange":
        colors = [
            (0.369, 0.235, 0.60),
            (0.698, 0.671, 0.824),
            (0.969, 0.969, 0.969),
            (0.992, 0.722, 0.388),
            (0.902, 0.380, 0.004),
        ]  # R -> G -> B
        cmap = LinearSegmentedColormap.from_list(name, colors, N=200)
    elif name == "SeaLand":
        colors = [
            (0.004, 0.522, 0.443),
            (0.502, 0.804, 0.757),
            (0.969, 0.969, 0.969),
            (0.875, 0.761, 0.490),
            (0.651, 0.380, 0.102),
        ]  # R -> G -> B
        cmap = LinearSegmentedColormap.from_list(name, colors, N=200)
    elif name == "Reds":
        colors = [
            (0.996, 0.941, 0.851),
            (0.992, 0.800, 0.541),
            (0.988, 0.553, 0.349),
            (0.843, 0.188, 0.122),
        ]  # R -> G -> B
        cmap = LinearSegmentedColormap.from_list(name, colors, N=200)
    else:
        cmap = plt.get_cmap(name)
    return cmap


def plot_colourline(x, y, c, cmap, ax=None, lw=0.8, transform=None, climatology=None):
    # Plots LCSs using coloured lines to define strength of attraction
    if climatology:
        c = cmap((c - 0.4) / (1.4 - 0.4))
    else:
        c = cmap((c - 0.4) / (3.0 - 0.4))
    if ax == None:
        ax = plt.gca()
    for i in np.arange(len(x) - 1):
        ax.plot([x[i], x[i + 1]], [y[i], y[i + 1]], c=c[i], lw=lw, transform=transform)
    return