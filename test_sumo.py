import os
import subprocess
import time

# Define paths to SUMO binaries and configuration files
sumo_binary = "C:/Program Files (x86)/Eclipse/Sumo/bin/sumo-gui"  # Adjust this path if necessary
net_file = "C:/automatic_vehicular_control/results/highway_ramp/baselines/sumo/net.xml"
rou_file = "C:/automatic_vehicular_control/results/highway_ramp/baselines/sumo/rou.xml"
add_file = "C:/automatic_vehicular_control/results/highway_ramp/baselines/sumo/add.xml"
gui_file = "C:/automatic_vehicular_control/results/highway_ramp/baselines/sumo/gui.xml"

# Command to start SUMO-GUI
sumo_cmd = [
    sumo_binary,
    "--net-file", net_file,
    "--route-files", rou_file,
    "--additional-files", add_file,
    "--gui-settings-file", gui_file,
    "--collision.action", "remove",
    "--begin", "0",
    "--step-length", "0.5",
    "--no-step-log", "true",
    "--time-to-teleport", "-1",
    "--no-warnings", "true",
    "--collision.check-junctions", "true",
    "--max-depart-delay", "0.5",
    "--random", "true",
    "--start", "true"
]

# Print the command to be executed
print("Starting SUMO-GUI with command:", " ".join(sumo_cmd))

# Start the SUMO-GUI process
sumo_process = subprocess.Popen(sumo_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# Allow the simulation to run for some steps
time.sleep(10)  # Adjust the sleep time as necessary to see vehicles

# Capture output and errors
stdout, stderr = sumo_process.communicate()
print("SUMO-GUI STDOUT:\n", stdout.decode())
print("SUMO-GUI STDERR:\n", stderr.decode())

# Terminate the SUMO-GUI process
sumo_process.terminate()
sumo_process.wait()

# Check if the SUMO-GUI process was terminated
if sumo_process.returncode is not None:
    print(f"SUMO-GUI terminated with return code: {sumo_process.returncode}")
else:
    print("SUMO-GUI did not terminate as expected.")
