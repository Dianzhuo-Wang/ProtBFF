# -*- coding: utf-8 -*-
"""
Generate mutant PDB structures from Bloom dataset CSV file (parallelized version).
"""
import sys
import os
import subprocess
import pandas as pd
import shutil
from multiprocessing import Pool, cpu_count
import time

# Constants
CHAIN_ID = 'A'  # 7C01 RBD chain (SARS-CoV-2 numbering)

# Global configuration (will be set by main())
CONFIG = {
    'foldx_bin': None,
    'pdb_in': None,
    'pdb_dir': None,
    'workdir': None,
    'residue_map': None,
}

def get_residue_map_from_pdb(pdb_file, chain_id):
    """Get mapping of residue numbers to amino acids from PDB ATOM records."""
    aa3_to_1 = {
        'ALA':'A', 'ARG':'R', 'ASN':'N', 'ASP':'D', 'CYS':'C', 'GLN':'Q', 'GLU':'E', 'GLY':'G',
        'HIS':'H', 'ILE':'I', 'LEU':'L', 'LYS':'K', 'MET':'M', 'PHE':'F', 'PRO':'P', 'SER':'S',
        'THR':'T', 'TRP':'W', 'TYR':'Y', 'VAL':'V'
    }
    
    residue_map = {}
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith('ATOM') and line[21:22].strip() == chain_id:
                resnum = int(line[22:26].strip())
                resname = line[17:20].strip()
                if resnum not in residue_map:
                    residue_map[resnum] = aa3_to_1.get(resname, 'X')
    
    return residue_map

def process_mutation(args):
    """Process a single mutation. Designed to be called in parallel."""
    i, row = args
    mutation = row['mutation']
    site_str = row['site']
    
    # Unpack global config
    foldx_bin = CONFIG['foldx_bin']
    pdb_in = CONFIG['pdb_in']
    data_root = CONFIG['pdb_dir']
    workdir = CONFIG['workdir']
    residue_map = CONFIG['residue_map']
    
    print(f"\n{'='*80}")
    print(f"Processing {i+1}: site={site_str}, mutation={mutation}")
    
    # Parse site
    if not site_str or str(site_str).strip() == '' or str(site_str).strip().lower() in ['nan', 'none', '']:
        print(f"? No site specified, using row index as site identifier")
        site_int = i
    else:
        try:
            site_int = int(float(str(site_str).strip()))
        except (ValueError, TypeError):
            print(f"? Invalid site '{site_str}', using row index {i}")
            site_int = i
    
    # Check if residue exists in PDB
    if site_int not in residue_map:
        print(f"? Residue {site_int} not found in PDB chain {CHAIN_ID}")
        print(f"  ? Skipped (residue not in PDB structure)")
        return False
    
    # Get wildtype amino acid from PDB
    wildtype_aa = residue_map[site_int]
    
    print(f"  Wildtype: {wildtype_aa}, Mutation: {wildtype_aa} -> {mutation}")
    
    # Create mutation string for FoldX
    mut_str = f"{wildtype_aa}{CHAIN_ID}{site_int}{mutation};"
    
    wildtype_output = os.path.join(workdir, "wildtype", f"{i}_wildtype.pdb")
    mutant_output = os.path.join(workdir, "optimized", f"{i}_optimized.pdb")
    
    try:
        # Check if both already exist
        if os.path.exists(mutant_output) and os.path.exists(wildtype_output):
            print(f"  ? Already exists")
            return True
        
        # Create unique worker directory
        worker_dir = os.path.join(workdir, f"worker_{i}")
        os.makedirs(worker_dir, exist_ok=True)
        
        cwd = os.getcwd()
        os.chdir(worker_dir)
        
        # Copy PDB file
        pdb_filename = os.path.basename(pdb_in)
        shutil.copy(pdb_in, pdb_filename)
        
        # Copy rotabase
        rotabase_src = os.path.join(data_root, "rotabase.txt")
        if os.path.exists(rotabase_src):
            shutil.copy(rotabase_src, "rotabase.txt")
        
        # STEP 1: Build wildtype (identity mutation)
        print(f"  Building wildtype...")
        wildtype_mut = f"{wildtype_aa}{CHAIN_ID}{site_int}{wildtype_aa};"
        with open("individual_list.txt", 'w') as f:
            f.write(wildtype_mut)
        
        build_cmd = [
            foldx_bin,
            "--command=BuildModel",
            f"--pdb={pdb_filename}",
            "--mutant-file=individual_list.txt",
            "--numberOfRuns=1"
        ]
        
        subprocess.run(build_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Find wildtype output
        pdb_base = pdb_filename.replace('.pdb', '')
        wt_file = f"{pdb_base}_1.pdb"
        
        if not os.path.exists(wt_file):
            print(f"  ? Wildtype build failed")
            os.chdir(cwd)
            # Cleanup worker directory
            try:
                shutil.rmtree(worker_dir)
            except:
                pass
            # Remove any existing files
            if os.path.exists(wildtype_output):
                try:
                    os.remove(wildtype_output)
                except:
                    pass
            if os.path.exists(mutant_output):
                try:
                    os.remove(mutant_output)
                except:
                    pass
            return False
        
        # Save wildtype temporarily (will be moved to final location only if mutant succeeds)
        temp_wildtype = os.path.join(worker_dir, "temp_wildtype.pdb")
        shutil.copy(wt_file, temp_wildtype)
        print(f"  ? Built wildtype")
                
        # STEP 2: Build mutant
        print(f"  Building mutant...")
        with open("individual_list.txt", 'w') as f:
            f.write(mut_str)
        
        build_cmd = [
            foldx_bin,
            "--command=BuildModel",
            f"--pdb={wt_file}",
            "--mutant-file=individual_list.txt",
            "--numberOfRuns=1"
        ]
        
        subprocess.run(build_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Find mutant output
        wt_base = wt_file.replace('.pdb', '')
        mutant_file = f"{wt_base}_1.pdb"
        
        if not os.path.exists(mutant_file):
            print(f"  ? Mutant build failed - removing all files")
            os.chdir(cwd)
            # Cleanup worker directory (including temp wildtype)
            try:
                shutil.rmtree(worker_dir)
            except:
                pass
            # Remove wildtype and mutant if they exist
            if os.path.exists(wildtype_output):
                try:
                    os.remove(wildtype_output)
                    print(f"  ? Removed wildtype")
                except:
                    pass
            if os.path.exists(mutant_output):
                try:
                    os.remove(mutant_output)
                    print(f"  ? Removed mutant")
                except:
                    pass
            return False
        
        # STEP 3: Optimize
        print(f"  Optimizing...")
        opt_cmd = [foldx_bin, "--command=Optimize", f"--pdb={mutant_file}"]
        subprocess.run(opt_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Find optimized output
        optimized_file = f"Optimized_{mutant_file}"
        if os.path.exists(optimized_file):
            final_file = optimized_file
        else:
            # If optimization failed, still remove everything
            print(f"  ? Optimization failed - removing all files")
            os.chdir(cwd)
            try:
                shutil.rmtree(worker_dir)
            except:
                pass
            if os.path.exists(wildtype_output):
                try:
                    os.remove(wildtype_output)
                    print(f"  ? Removed wildtype")
                except:
                    pass
            if os.path.exists(mutant_output):
                try:
                    os.remove(mutant_output)
                    print(f"  ? Removed mutant")
                except:
                    pass
            return False
        
        # STEP 4: Copy both wildtype and mutant to output (only if everything succeeded)
        os.makedirs(os.path.dirname(wildtype_output), exist_ok=True)
        os.makedirs(os.path.dirname(mutant_output), exist_ok=True)
        
        shutil.copy(temp_wildtype, wildtype_output)
        shutil.copy(final_file, mutant_output)
        
        print(f"  ? Saved wildtype")
        print(f"  ? Created mutant")
        
        os.chdir(cwd)
        
        # Cleanup
        try:
            shutil.rmtree(worker_dir)
        except:
            pass
        
        return True
        
    except Exception as e:
        print(f"  ? Error: {e}")
        if 'cwd' in locals():
            os.chdir(cwd)
        
        # Remove both wildtype and mutant if they exist
        if os.path.exists(wildtype_output):
            try:
                os.remove(wildtype_output)
                print(f"  ? Removed wildtype (error occurred)")
            except:
                pass
        if os.path.exists(mutant_output):
            try:
                os.remove(mutant_output)
                print(f"  ? Removed mutant (error occurred)")
            except:
                pass
        
        return False


def main():
    n_workers = cpu_count()
    print(f"Using {n_workers} worker processes")
    
    # Paths
    csv_path = "/n/netscratch/shakhnovich_lab/Lab/jonathanfeldman/ProSST_PPI-main/bloom_antibodies/df_bloom_antibodies.csv"
    output_csv_path = "/n/netscratch/shakhnovich_lab/Lab/jwang/ProtBFF/data/dms_cov016/cov016_with_index.csv"
    pdb_in = "/n/netscratch/shakhnovich_lab/Lab/jwang/ProtBFF/data/dms_cov016/7C01_AHL.pdb"
    foldx_bin = "/n/netscratch/shakhnovich_lab/Lab/jonathanfeldman/ProSST_PPI-main/bloom_antibodies/FoldX"
    workdir = "/n/netscratch/shakhnovich_lab/Lab/jwang/ProtBFF/data/dms_cov016/cov016_cache/"
    skempi_root = "/n/netscratch/shakhnovich_lab/Lab/jonathanfeldman/DDAffinity_up-master/data/SKEMPI2/"
    
    pdb_dir = os.path.dirname(pdb_in)
    
    # Set global config
    CONFIG['foldx_bin'] = foldx_bin
    CONFIG['pdb_in'] = pdb_in
    CONFIG['pdb_dir'] = pdb_dir
    CONFIG['workdir'] = workdir
    
    # Check FoldX
    if not os.path.isfile(foldx_bin):
        print(f"? FoldX binary not found at: {foldx_bin}")
        sys.exit(1)
    if not os.access(foldx_bin, os.X_OK):
        os.chmod(foldx_bin, 0o755)

    # Create directories
    os.makedirs(workdir, exist_ok=True)
    os.makedirs(os.path.join(workdir, "wildtype"), exist_ok=True)
    os.makedirs(os.path.join(workdir, "optimized"), exist_ok=True)
    
    # Copy rotabase
    rotabase_src = os.path.join(skempi_root, "rotabase.txt")
    rotabase_dst = os.path.join(pdb_dir, "rotabase.txt")
    if os.path.exists(rotabase_src) and not os.path.exists(rotabase_dst):
        shutil.copy(rotabase_src, rotabase_dst)
        print(f"? Copied rotabase.txt")
    
    # Read CSV
    print(f"Reading CSV...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows")
    
    # Add pdb_index column (matches the output filename index)
    df['pdb_index'] = df.index
    # optional smoke-test slice (indices preserved -> output filenames stay aligned to full df)
    if os.environ.get('COV016_TEST'):
        df = df[df['site'].between(333, 527)].head(int(os.environ['COV016_TEST'])).copy()
        print(f"TEST MODE: {len(df)} rows (sites {list(df['site'].astype(int))[:8]})")
    
    # Save updated CSV with pdb_index column
    df.to_csv(output_csv_path, index=False)
    print(f"? Saved CSV with pdb_index column to: {output_csv_path}")
    
    # Get residue map
    print(f"Reading PDB chain {CHAIN_ID}...")
    residue_map = get_residue_map_from_pdb(pdb_in, CHAIN_ID)
    print(f"Found {len(residue_map)} residues (range: {min(residue_map.keys())}-{max(residue_map.keys())})")
    
    CONFIG['residue_map'] = residue_map
    
    # Check missing sites
    csv_sites = df['site'].dropna().astype(int).unique()
    missing_sites = [s for s in csv_sites if s not in residue_map]
    if missing_sites:
        print(f"? {len(missing_sites)} sites not in PDB (will be skipped)")
    
    # Process mutations
    args_list = [(i, row) for i, row in df.iterrows()]
    
    print(f"\n{'='*80}")
    print(f"Processing {len(args_list)} mutations with {n_workers} workers")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    with Pool(processes=n_workers) as pool:
        results = pool.map(process_mutation, args_list)
    
    elapsed_time = time.time() - start_time
    
    success_count = sum(results)
    error_count = len(results) - success_count
    
    print(f"\n{'='*80}")
    print(f"Completed in {elapsed_time:.2f} seconds")
    print(f"Success: {success_count}, Errors: {error_count}")
    print(f"Output: {workdir}")
    print(f"Indexed CSV: {output_csv_path}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()