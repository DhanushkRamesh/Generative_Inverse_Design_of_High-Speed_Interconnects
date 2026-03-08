import os
import glob
from collections import Counter

def analyze_touchstone_ports(dataset_name, base_dir):
    print(f"Analyzing raw ports for dataset: {dataset_name}")
    variations_dir = os.path.join(base_dir, "variation")
    if not os.path.exists(variations_dir):
        print(f"Variations directory not found: {variations_dir}")
        return
    sim_folders = [f.path for f in os.scandir(variations_dir) if f.is_dir()]
    print(f"Found {len(sim_folders)} simulation folders in variations.")
    port_counts=[]

    for folder in sim_folders:
        #look for any files missing .s*p extension
        files = glob.glob(os.path.join(folder, "*.s*p"))
        if files:
            #grab first touchstone file found
            filename = files[0]
            extension = filename.split('.')[-1]

            try:
                num_ports = int(extension[1:-1])  # Extract the number after 's' in the extension
                port_counts.append(num_ports)
            except ValueError:
                print(f"Could not parse number of ports from file: {filename}")
                continue
        else:
            print(f"No touchstone files found in folder: {folder}")
    #count the occurrences of each port count
    port_count_summary = Counter(port_counts)
    print("-" * 50)
    print("Port count summary:")
    for ports, count in sorted(port_count_summary.items()):
        print(f"{ports} ports: {count} simulations")
    print("-" * 50)

if __name__ == "__main__":

    # Define your raw data paths
    array_dir = os.path.expanduser("~/mece_project_inverse_model/Generative_Inverse_Design_of_High-Speed_Interconnects/data/raw/Universal-Diff-SI-Array")
    link_dir = os.path.expanduser("~/mece_project_inverse_model/Generative_Inverse_Design_of_High-Speed_Interconnects/data/raw/Universal-Diff-SI-Link")
    
    # Run the analysis on both datasets
    analyze_touchstone_ports("Universal-Diff-SI-Array", array_dir)
    analyze_touchstone_ports("Universal-Diff-SI-Link", link_dir)