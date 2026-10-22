

import h5py

filename = "/ceph/cephfs/jzuntz/ridges/des-data/DESY3_maglim_redmagic_v0.5.1.h5"
outfile = "./DESY3_maglim_redmagic_v0.5.1.h5"

index_file = "/ceph/cephfs/jzuntz/ridges/des-data/DESY3_indexcat.h5"

with h5py.File(index_file) as findex:
    index = findex['/index/maglim/select'][:]

with h5py.File(filename) as fin:
    gin = fin['catalog/maglim']
    w = gin['weight'][:][index]
    sel = w > 0
    w = w[sel]
    
    with h5py.File(outfile, 'w') as fout:
        fout.create_dataset('weight', data=w)
        fout.create_dataset('ra', data=gin['ra'][:][index][sel])
        fout.create_dataset('dec', data=gin['dec'][:][index][sel])
