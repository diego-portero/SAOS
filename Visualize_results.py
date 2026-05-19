import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# -----------------------------
# FILE PATH
# -----------------------------
file_path = "/home/dportero/Tesis/SAOS/simulations_5/savepoints/saos_savepoint_20260518_1602.h5"

def print_h5_structure(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"DATASET: {name}")
    elif isinstance(obj, h5py.Group):
        print(f"GROUP:   {name}")

with h5py.File(file_path, "r") as f:
    f.visititems(print_h5_structure)

# -----------------------------
# DATASETS (NOW INCLUDING DM)
# -----------------------------
dataset_paths = [
    "LightPath_0/sci_opd/data",
    "LightPath_0/sci_frame_shortExp/data",
    "LightPath_0/wfs_frame/data",
    "LightPath_0/dm_opd_0/data",   # 👈 DM ADDED
]

datasets = []
titles = []

with h5py.File(file_path, "r") as f:
    for path in dataset_paths:
        data = f[path][:]
        datasets.append(data)
        titles.append(path.split("/")[-2])

# -----------------------------
# FRAME COUNT
# -----------------------------
n_frames = datasets[0].shape[0]

# -----------------------------
# FIGURE SETUP
# -----------------------------
n_plots = len(datasets)
fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
plt.subplots_adjust(bottom=0.25)

if n_plots == 1:
    axes = [axes]

images = []

# -----------------------------
# INITIAL IMAGES
# -----------------------------
for ax, data, title in zip(axes, datasets, titles):

    vmin = np.min(data)
    vmax = np.max(data)

    # handle (N,1,H,W) vs (N,H,W)
    if data.ndim == 4:
        img = ax.imshow(data[0, 0],
                        cmap="inferno",
                        vmin=vmin,
                        vmax=vmax)
    else:
        img = ax.imshow(data[0],
                        cmap="inferno",
                        vmin=vmin,
                        vmax=vmax)

    ax.set_title(title)
    images.append(img)

    fig.colorbar(img, ax=ax)

# -----------------------------
# SLIDER
# -----------------------------
ax_slider = plt.axes([0.2, 0.1, 0.6, 0.04])
slider = Slider(ax_slider, "Frame", 0, n_frames - 1,
                valinit=0, valstep=1)

def update(val):
    frame = int(slider.val)

    for img, data in zip(images, datasets):

        if data.ndim == 4:
            img.set_data(data[frame, 0])
        else:
            img.set_data(data[frame])

    fig.canvas.draw_idle()

slider.on_changed(update)

# -----------------------------
# RMS PLOT (unchanged)
# -----------------------------
with h5py.File(file_path, "r") as f:
    rms = f["LightPath_0/sci_opd/rms"][:]
    iterations = f["LightPath_0/sci_opd/iteration"][:]

rms_nm = rms * 1e9

plt.figure(figsize=(8, 5))
plt.plot(iterations, rms_nm)
plt.xlabel("Iteration")
plt.ylabel("Residual RMS Wavefront Error (nm)")
plt.title("Science Residual RMS vs Iteration")
plt.grid(True)

# -----------------------------
# STREHL RATIO PLOT
# -----------------------------
with h5py.File(file_path, "r") as f:
    iter_short = f["LightPath_1/sci_frame_shortExp/iteration"][:]
    strehl_short = f["LightPath_1/sci_frame_shortExp/strehl"][:]

    # If you also want long exposure:
    iter_long = f["LightPath_1/sci_frame_longExp/iteration"][:]
    strehl_long = f["LightPath_1/sci_frame_longExp/strehl"][:]

plt.figure(figsize=(8, 5))

plt.plot(iter_short, strehl_short, label="Short Exposure")
plt.plot(iter_long, strehl_long, label="Long Exposure")

plt.xlabel("Iteration")
plt.ylabel("Strehl Ratio")
plt.title("Strehl Ratio vs Iteration")
plt.grid(True)
plt.legend()

# -----------------------------
# SAVE FIRST + LAST SUN IMAGE TO SEPARATE PDFs
# -----------------------------
sun_data = datasets[1]   # LightPath_0/sci_frame_shortExp/data

# ---- First frame PDF ----
fig1, ax1 = plt.subplots(figsize=(6, 6))

if sun_data.ndim == 4:
    img1 = ax1.imshow(sun_data[0, 0], cmap="inferno")
else:
    img1 = ax1.imshow(sun_data[0], cmap="inferno")

ax1.set_title("Sun - First Frame")
ax1.axis("off")   # optional: cleaner figure
plt.colorbar(img1, ax=ax1)

fig1.savefig("sun_first_frame.pdf", bbox_inches="tight")
plt.close(fig1)

# ---- Last frame PDF ----
fig2, ax2 = plt.subplots(figsize=(6, 6))

if sun_data.ndim == 4:
    img2 = ax2.imshow(sun_data[-1, 0], cmap="inferno")
else:
    img2 = ax2.imshow(sun_data[-1], cmap="inferno")

ax2.set_title("Sun - Last Frame")
ax2.axis("off")
plt.colorbar(img2, ax=ax2)

fig2.savefig("sun_last_frame.pdf", bbox_inches="tight")
plt.close(fig2)

print("Saved:")
print(" - sun_first_frame.pdf")
print(" - sun_last_frame.pdf")
plt.show()



print("Visualization complete.")

