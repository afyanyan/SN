import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class SN:
    def __init__(self, lc, bkg):
        self.lc = lc
        self.bkg = bkg

        # set after reading
        self.epoch = None
        self.time = None         # days since epoch
        self.mag = None
        self.ave_BKG = None

        # detection metadata (MJD)
        self.last_nd_mjd = None
        self.first_det_mjd = None

        # set after shock_breakout
        self.t_sbo = None
        self.t_sbo_l = None
        self.t_sbo_r = None

    # ---------- robust reading helpers ----------
    def _read_table_try_formats(self, path):
        """Try CSV then whitespace; clean column names."""
        try:
            df = pd.read_csv(path, sep=",", comment="#", header=0)
        except Exception:
            df = pd.read_csv(path, delim_whitespace=True, comment="#", header=0)

        df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
        return df

    def _find_time_col(self, df):
        candidates = ["MJD", "mjd", "time", "Time", "jd", "JD", "date", "Date"]
        for c in candidates:
            if c in df.columns:
                return c
        return df.columns[0]

    def _find_mag_col(self, df):
        candidates = [
            "mag", "Mag", "MAG", "magnitude", "Magnitude",
            "limit", "Limit", "lim", "maglim", "MagLim", "UL", "ul"
        ]
        for c in candidates:
            if c in df.columns:
                return c
        if len(df.columns) >= 2:
            return df.columns[1]
        return df.columns[0]

    # ---------- main I/O ----------
    def read_data(self):
        """Read LC and background/UL file; set epoch and ave_BKG."""
        df_bkg = self._read_table_try_formats(self.bkg)
        df_lc = self._read_table_try_formats(self.lc)

        tcol_lc = self._find_time_col(df_lc)
        mcol_lc = self._find_mag_col(df_lc)

        t = np.array(df_lc[tcol_lc], dtype=float)
        m = np.array(df_lc[mcol_lc], dtype=float)

        self.epoch = float(np.nanmin(t))
        self.time = np.array(t - self.epoch, dtype=float)
        self.mag = np.array(m, dtype=float)

        # background/UL average magnitude (horizontal reference line)
        mcol_bkg = self._find_mag_col(df_bkg)
        bkg_vals = np.array(df_bkg[mcol_bkg], dtype=float)
        self.ave_BKG = float(np.nanmean(bkg_vals))

        return self.time, self.mag

    # ---------- last ND + first detection ----------
    def last_non_detection_and_first_detection(self):
        """Compute and store last non-detection (from UL file) and first detection (from LC)."""
        df_lc = self._read_table_try_formats(self.lc)
        tcol_lc = self._find_time_col(df_lc)
        t_lc = np.array(df_lc[tcol_lc], dtype=float)

        self.first_det_mjd = float(np.nanmin(t_lc))

        self.last_nd_mjd = None
        try:
            df_ul = self._read_table_try_formats(self.bkg)
            tcol_ul = self._find_time_col(df_ul)
            t_ul = np.array(df_ul[tcol_ul], dtype=float)

            before = t_ul[t_ul < self.first_det_mjd]
            if len(before) > 0:
                self.last_nd_mjd = float(np.nanmax(before))
            else:
                # fallback: latest UL time available
                self.last_nd_mjd = float(np.nanmax(t_ul))
        except Exception as e:
            print(f"[WARN] Could not parse UL/background file for last ND: {e}")
            self.last_nd_mjd = None

        print(f"First detection (MJD): {self.first_det_mjd}")
        print(f"Last non-detection (MJD): {self.last_nd_mjd}")
        return self.last_nd_mjd, self.first_det_mjd

    # ---------- shock breakout ----------
    def shock_breakout(self, Xp, yp, ystd, search_pad_days=2.0):
        """
        Define t_SBO as where GP mean crosses ave_BKG before first detection.
        Restricts the search region up to first detection (plus a small pad) to avoid nonsense.
        """
        if self.first_det_mjd is None or self.epoch is None:
            half = len(Xp) // 2
            idx = np.argmin(np.abs((yp - self.ave_BKG)[:half]))
            idx_l = np.argmin(np.abs((yp - ystd - self.ave_BKG)[:half]))
            idx_r = np.argmin(np.abs((yp + ystd - self.ave_BKG)[:half]))
        else:
            t_first_rel = (self.first_det_mjd - self.epoch) + search_pad_days
            mask = (Xp <= t_first_rel)
            if not np.any(mask):
                mask = np.ones_like(Xp, dtype=bool)

            Xp_s = Xp[mask]
            yp_s = yp[mask]
            ystd_s = ystd[mask]

            idx = np.argmin(np.abs(yp_s - self.ave_BKG))
            idx_l = np.argmin(np.abs((yp_s - ystd_s) - self.ave_BKG))
            idx_r = np.argmin(np.abs((yp_s + ystd_s) - self.ave_BKG))

            full_idx = np.where(mask)[0]
            idx, idx_l, idx_r = full_idx[idx], full_idx[idx_l], full_idx[idx_r]

        self.t_sbo = float(Xp[idx])
        self.t_sbo_l = float(Xp[idx_l])
        self.t_sbo_r = float(Xp[idx_r])

        print(
            "Shock breakout between",
            self.t_sbo_l + self.epoch,
            "and",
            self.t_sbo_r + self.epoch,
            "MJD"
        )
        return

    # ---------- plotting ----------
    def plot_fit(
        self,
        Xp,
        yp,
        ystd,
        xpad_left=2.0,
        xpad_right=2.0,
        max_extrap_days=1.0,
        outdir="outputs",
    ):
        """
        Colleague-style plot, with guaranteed visibility of the BKG line + SBO point.

        Key fixes vs v3:
        - y-limits include BOTH the observed magnitudes and the BKG level.
          (Your previous plot didn't show the black BKG line because BKG~19 was outside the y-range ~14–16.)
        """
        os.makedirs(outdir, exist_ok=True)

        # observed times in MJD
        t_obs_mjd = self.time + self.epoch
        tmin = float(np.nanmin(t_obs_mjd))
        tmax = float(np.nanmax(t_obs_mjd))

        xmin = tmin - float(xpad_left)
        xmax = tmax + float(xpad_right)

        # hard-clip GP display to avoid edge blow-up
        clip_max = tmax + float(max_extrap_days)

        t_gp_mjd_full = Xp + self.epoch
        m = (t_gp_mjd_full >= xmin) & (t_gp_mjd_full <= min(xmax, clip_max))
        if np.any(m):
            t_gp_mjd = t_gp_mjd_full[m]
            yp_p = yp[m]
            ystd_p = ystd[m]
        else:
            t_gp_mjd = t_gp_mjd_full
            yp_p = yp
            ystd_p = ystd

        plt.figure(figsize=(10, 7))
        plt.plot(t_obs_mjd, self.mag, "r.", label="Observed data", zorder=3)

        # GP curve + band
        plt.plot(t_gp_mjd, yp_p, color="blue", alpha=0.9, linewidth=2.5, label="GP fit", zorder=2)
        plt.fill_between(t_gp_mjd, yp_p - ystd_p, yp_p + ystd_p, alpha=0.2, color="blue", zorder=1)

        # BKG line: thick + high zorder
        plt.axhline(self.ave_BKG, color="black", linewidth=3.0, label="BKG", zorder=4)

        # last non-detection marker on BKG line
        if self.last_nd_mjd is not None:
            plt.scatter(
                self.last_nd_mjd,
                self.ave_BKG,
                s=90,
                color="black",
                label="Last Null Detection",
                zorder=6,
            )

        # t_SBO marker on BKG line
        if self.t_sbo is not None:
            plt.scatter(
                self.t_sbo + self.epoch,
                self.ave_BKG,
                s=120,
                color="blue",
                label=f"('SBO', {self.t_sbo + self.epoch})",
                zorder=7,
            )

        plt.xlabel("MJD")
        plt.ylabel("mag")
        plt.gca().invert_yaxis()

        # ✅ y-range MUST include BKG + observed magnitudes
        y_vals = [np.nanmin(self.mag), np.nanmax(self.mag), self.ave_BKG]
        y_lo = float(np.nanmin(y_vals)) - 0.8
        y_hi = float(np.nanmax(y_vals)) + 0.8
        plt.ylim(y_hi, y_lo)

        # x-range
        plt.xlim(xmin, xmax)

        if self.t_sbo_l is not None and self.t_sbo_r is not None:
            plt.title(
                "Shock break out time from %.2f to %.2f MJD"
                % (self.t_sbo_l + self.epoch, self.t_sbo_r + self.epoch),
                fontsize=22,
            )

        plt.grid(True, alpha=0.35)
        plt.legend()

        stem = os.path.basename(self.lc).split(".")[0]
        outname = os.path.join(outdir, f"{stem}-fit.png")
        plt.savefig(outname, dpi=250, bbox_inches="tight")
        plt.close()
        return outname
