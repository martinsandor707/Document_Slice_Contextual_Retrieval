import pynvml
import threading
import time
import matplotlib
matplotlib.use('Agg') # Forces Matplotlib to run without a GUI
import matplotlib.pyplot as plt

class GPUMonitor:
    def __init__(self, interval_sec=1.0):
        self.interval_sec = interval_sec
        self.keep_measuring = True
        self.times = []
        self.utilizations = []
        
        pynvml.nvmlInit()
        # Grabs the first GPU (Index 0). Change if using multiple GPUs.
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0) 

    def _monitor(self):
        start_time = time.time()
        while self.keep_measuring:
            util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            current_time = time.time() - start_time
            
            self.times.append(current_time)
            self.utilizations.append(util.gpu) # Percent utilization (0-100)
            time.sleep(self.interval_sec)

    def start(self):
        self.thread = threading.Thread(target=self._monitor)
        self.thread.start()

    def stop(self):
        self.keep_measuring = False
        self.thread.join()
        pynvml.nvmlShutdown()

# --- 1. Run Your Experiment ---
monitor = GPUMonitor(interval_sec=1.0) # Poll every 1 second
monitor.start()

# ... Your training script goes here ...
time.sleep(15) # Simulating 15 seconds of workload

monitor.stop()

# --- 2. Generate the Publication Figure ---
# Set font family to serif (often preferred in IEEE/Nature/Elsevier)
plt.rcParams['font.family'] = 'serif'

fig, ax = plt.subplots(figsize=(7, 4)) # Standard 1-column width for papers

# Fill under the curve for a clean visual look
ax.plot(monitor.times, monitor.utilizations, color='black', linewidth=1.2)
ax.fill_between(monitor.times, monitor.utilizations, color='gray', alpha=0.3)

ax.set_xlabel('Time (seconds)', fontsize=11)
ax.set_ylabel('GPU Utilization (%)', fontsize=11)
ax.set_ylim(0, 105)
ax.set_xlim(0, max(monitor.times))

# Clean up axes (spines)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()

# Save as PDF (Vector format required by most publishers)
plt.savefig('gpu_timeseries.pdf', format='pdf', bbox_inches='tight')