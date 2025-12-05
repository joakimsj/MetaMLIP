import os
from ase.io import read, write
import shutil

if os.path.exists("farming_driver.inp") and len([d for d in os.listdir() if d.startswith("run")]) > 0:
    print("Farming input already exists. Skipping regeneration.")
    exit(0)

# === Configuration ===
xyz_file = "frames_for_DFT_eval_filtered.xyz"
template_input = "template.inp"  # Must contain @CELL@ as a placeholder
output_prefix = "run"
output_input_name = "sp.inp"
output_xyz_name = "structure.xyz"
farming_input_file = "farming_driver.inp"

# === Hardware configuration ===
total_cores = 512
cores_per_job = 128
ngroups = total_cores // cores_per_job

# === Load frames ===
frames = read(xyz_file, index=":")
nframes = len(frames)
print(f"Found {nframes} frames. Preparing {nframes} jobs...")

# === Read the CP2K input template ===
with open(template_input, "r") as f:
    template_text = f.read()

# === Create job directories ===
for i, frame in enumerate(frames, start=1):
    run_dir = f"{output_prefix}{i}"
    os.makedirs(run_dir, exist_ok=True)

    # Write the structure
    xyz_path = os.path.join(run_dir, output_xyz_name)
    write(xyz_path, frame)

    # Extract and format cell
    cell = frame.get_cell()
    a, b, c = cell[0], cell[1], cell[2]
    cell_block = (
        f"  &CELL\n"
        f"    A {a[0]} {a[1]} {a[2]}\n"
        f"    B {b[0]} {b[1]} {b[2]}\n"
        f"    C {c[0]} {c[1]} {c[2]}\n"
        f"    PERIODIC XYZ\n"
        f"  &END CELL"
    )

    # Inject cell into the input file
    customized_input = template_text.replace("@CELL@", cell_block)

    # Save the customized input
    input_dest = os.path.join(run_dir, output_input_name)
    with open(input_dest, "w") as f:
        f.write(customized_input)

# === Generate FARMING input ===
with open(farming_input_file, "w") as f:
    f.write("&GLOBAL\n")
    f.write("  PROJECT cp2k_farming\n")
    f.write("  PROGRAM FARMING\n")
    f.write("  RUN_TYPE NONE\n")
    f.write("&END GLOBAL\n\n")

    f.write("&FARMING\n")
    f.write(f"  NGROUPS {ngroups}\n")

    for i in range(1, nframes + 1):
        f.write("  &JOB\n")
        f.write(f"    DIRECTORY {output_prefix}{i}\n")
        f.write(f"    INPUT_FILE_NAME {output_input_name}\n")
        f.write("  &END JOB\n")
    
    f.write("&END FARMING\n")

print(f"\nAll jobs prepared in {nframes} directories.")
print(f"FARMING input written to: {farming_input_file}")
print(f"Parallel jobs: {ngroups} at a time")
