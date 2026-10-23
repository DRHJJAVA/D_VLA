#!/usr/bin/env python3
"""Precompute PointNet++ sampling, radius groups and 3-NN interpolation graphs."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree


def fps(xyz,count):
    selected=np.empty(count,dtype=np.int32)
    distance=np.full(len(xyz),np.inf,dtype=np.float32)
    current=int(np.argmax(np.sum((xyz-xyz.mean(axis=0))**2,axis=1)))
    for step in range(count):
        selected[step]=current
        diff=xyz-xyz[current]
        distance=np.minimum(distance,np.einsum('ij,ij->i',diff,diff))
        current=int(np.argmax(distance))
    return selected


def neighbors(source,centers,radius,k=32):
    tree=cKDTree(source)
    distance,index=tree.query(centers,k=k,distance_upper_bound=radius,workers=-1)
    if index.ndim==1:index=index[:,None]
    bad=index>=len(source)
    index[bad]=np.broadcast_to(index[:,0,None],index.shape)[bad]
    return index.astype(np.int32)


def interpolate(source,target):
    distance,index=cKDTree(source).query(target,k=3,workers=-1)
    weight=1/np.maximum(distance,1e-8)
    weight=weight/weight.sum(axis=1,keepdims=True)
    return index.astype(np.int32),weight.astype(np.float32)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--sample-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--min-samples',type=int,default=20)
    p.add_argument('--wait-seconds',type=int,default=3600)
    a=p.parse_args()
    start=time.monotonic()
    while len(list(a.sample_dir.glob('*.npz')))<a.min_samples:
        if time.monotonic()-start>a.wait_seconds:raise TimeoutError('Point samples not ready')
        time.sleep(10)
    a.output_dir.mkdir(parents=True,exist_ok=True)
    for path in sorted(a.sample_dir.glob('*.npz')):
        target=a.output_dir/path.name
        if target.exists():continue
        with np.load(path) as sample:
            xyz=sample['xyz'].astype(np.float32)
        if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) < 1024: raise ValueError(f'Unexpected sample shape: {path}: {xyz.shape}')
        idx1=fps(xyz,1024);xyz1=xyz[idx1]
        idx2=fps(xyz1,256);xyz2=xyz1[idx2]
        idx3=fps(xyz2,64);xyz3=xyz2[idx3]
        nb1=neighbors(xyz,xyz1,0.04);nb2=neighbors(xyz1,xyz2,0.08)
        nb3=neighbors(xyz2,xyz3,0.16)
        fp3_idx,fp3_weight=interpolate(xyz3,xyz2)
        fp2_idx,fp2_weight=interpolate(xyz2,xyz1)
        fp1_idx,fp1_weight=interpolate(xyz1,xyz)
        np.savez_compressed(target,idx1=idx1,idx2=idx2,idx3=idx3,
            nb1=nb1,nb2=nb2,nb3=nb3,
            fp3_idx=fp3_idx,fp3_weight=fp3_weight,
            fp2_idx=fp2_idx,fp2_weight=fp2_weight,
            fp1_idx=fp1_idx,fp1_weight=fp1_weight)
        print(json.dumps({'graph':str(target)}),flush=True)
    print(json.dumps({'completed':len(list(a.output_dir.glob('*.npz')))}),flush=True)

if __name__=='__main__':main()
