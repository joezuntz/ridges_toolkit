import h5py
import numpy as np
import healpy as hp
from astropy.table import Table
import gc

class Catalog:
    """
    A catalog in HDF5 format. Always has RA and Dec.
    May have "z", "g1", "g2", and "weight" fields.

    RA and Dec are always in degrees in the file. We convert them when needed.
    """

    columns = []
    optional_columns = []

    def __init__(self, filename):
        self.filename = filename
        self.data = {}
        self.metadata = {}
        self.loaded = False

    def dec_ra_in_radians(self):
        dec = np.radians(self.dec)
        ra = np.radians(self.ra)
        return np.column_stack((dec, ra))

    def set_column(self, column, values):
        self.data[column] = values

    def get_column(self, column):
        if column not in self.data:
            raise ValueError(f"Column {column} not loaded")
        return self.data[column]

    # use init_subclass to define properties for all columns specified in
    # the subclass
    def __init_subclass__(cls):
        for col in cls.columns:
            setattr(cls, col, property(lambda self, c=col: self.get_column(c)))

    def save(self):
        # If we are splitting over ranks, we want every rank to write its own chunk of the data.
        # First determine the size of those chunks and the range
        with h5py.File(self.filename, "w") as f:
            for col in self.columns:
                f.create_dataset(col, data=self.data[col])
            for key, value in self.metadata.items():
                f.attrs[key] = value

    def load(self, comm=None, split_over_ranks=True, reload=False):
        if self.loaded and not reload:
            return
        # If we are splitting over ranks, we want every rank to read its own chunk of the data.
        # First determine the size of those chunks and the range
        if comm is not None and split_over_ranks:
            rank = comm.rank
            size = comm.size
            with h5py.File(self.filename, "r") as f:
                length = f[self.columns[0]].shape[0]
            rows = length // size
            if rank == size - 1:
                my_rows = length - rows * (size - 1)
            else:
                my_rows = rows

            start = rank * rows
            slc = slice(start, start + my_rows)
            if rank < 4:
                print("Rank", rank, "loading rows", start, "to", start + my_rows)
            if rank == 4:
                print("Rank", rank, "loading rows", start, "to", start + my_rows, " ... rest of ranks not printed out")
        else:
            slc = slice(None)

        # We should actually load the data if any of:
        # - we have no communicator (single process)
        # - we are rank 0 (the one that will read all the data and send to others)
        # - we are splitting over ranks, in which case every rank should read its own chunk
        if (comm is None) or (comm.rank == 0) or split_over_ranks:
            with h5py.File(self.filename, "r") as f:
                for col in self.columns:
                    if col not in f and col not in self.optional_columns:
                        raise ValueError(f"Column {col} not found in file {self.filename}")
                    elif col in f:
                        self.data[col] = f[col][slc]
                        if f[col].dtype == np.float32:
                            self.data[col] = self.data[col].astype(np.float64)

        # In this case every process should get all the catalog
        if (comm is not None) and (not split_over_ranks):
            comm.barrier()
            self.data = comm.bcast(self.data, root=0)

        # everyone reads the metadata
        with h5py.File(self.filename, "r") as f:
            self.metadata = dict(f.attrs)

        self.loaded = True

    def unload(self):
        self.data = {}
        self.loaded = False
        gc.collect()
    
    def cut(self, mask):
        for col, data in list(self.data.items()):
            self.data[col] = data[mask]


    def cut_to_redshift_range(self, zmin, zmax):
        if "z" not in self.data:
            raise ValueError("No redshift information loaded")
        mask = (self.data["z"] >= zmin) & (self.data["z"] < zmax)
        for col, data in list(self.data.items()):
            self.data[col] = data[mask]


class LensCatalog(Catalog):
    columns = ["ra", "dec", "z", "weight"]
    optional_columns = ["z", "weight"]


class SourceCatalog(Catalog):
    columns = ["ra", "dec", "z", "g1", "g2", "weight"]
    optional_columns = ["z"]


class RidgePointCatalog(Catalog):
    columns = ["ra", "dec", "density"]

    def apply_density_cut(self, percentile):
        density_threshold = np.percentile(self.density, percentile)
        mask = self.density >= density_threshold
        for col, data in list(self.data.items()):
            self.data[col] = data[mask]

    # TODO: Adapt function to trim around edges below


class RidgeSegmentCatalog(Catalog):
    columns = ["ra", "dec", "ridge_id"]

    def iterate_ridges(self, radians=False):
        self.load()
        # Ridges are (now) stored in order of ridge id,
        # so we can just find wherever the ridge ID increases and take
        # those chunks as the ridges.
        change_points = np.where(np.diff(self.ridge_id) != 0)[0] + 1
        boundaries = np.concatenate(([0], change_points, [len(self.ridge_id)]))
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            label = self.ridge_id[start]
            ra_chunk = self.ra[start:end]
            dec_chunk = self.dec[start:end]
            if radians:
                ra_chunk = np.radians(ra_chunk) % (2 * np.pi)
                dec_chunk = np.radians(dec_chunk)
            yield label, ra_chunk, dec_chunk



class ShearMeasurement:
    columns = ["sep_bin_center", "weighted_sep", "g_plus", "g_cross", "counts", "weight"]

    def __init__(self, filename):
        self.filename = filename
        self.data = Table(names=self.columns, dtype=[float, float, float, float, int, float])
        self.loaded = False

    def load(self):
        if self.loaded:
            return

        self.data = Table.read(self.filename, format="ascii.commented_header")
        self.loaded = True

    def save(self):
        self.data.write(self.filename, format="ascii.commented_header", overwrite=True)


def ridge_edge_filter_disk(ridge_ra, ridge_dec, mask, nside, radius_arcmin, min_coverage=1.0):
    """Return boolean array of ridge points with coverage ≥ min_coverage."""
    radius = np.radians(radius_arcmin / 60.0)
    theta_ridges = (np.pi / 2.0) - ridge_dec  # NEW: ridge_dec is radians
    phi_ridges = ridge_ra  # NEW: ridge_ra is radians
    # theta_ridges = np.radians(90.0 - ridge_dec)
    # phi_ridges = np.radians(ridge_ra)
    vec_ridges = hp.ang2vec(theta_ridges, phi_ridges)
    keep_idx = np.zeros(len(ridge_ra), dtype=bool)

    for i, v in enumerate(vec_ridges):
        disk_pix = hp.query_disc(nside, v, radius, inclusive=True)
        if len(disk_pix) == 0:
            frac = 0.0
        else:
            frac = mask[disk_pix].sum() / len(disk_pix)
        if frac >= min_coverage:
            keep_idx[i] = True
    return keep_idx
