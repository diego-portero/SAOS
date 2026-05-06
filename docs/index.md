<div class="saos-hero" markdown>

# ☀️ SAOS

**Solar Adaptive Optics Simulator** — a Python-first, free-threaded simulation framework for wide-field solar AO.

<div class="saos-hero-badge" markdown>
[📚 Documentation](installation.md){ .md-button }
[🚀 Get Started](tutorials/01_intro_to_saos.ipynb){ .md-button }
[💻 GitHub](https://github.com/nrodlin/SAOS){ .md-button }
</div>

</div>

<div class="saos-cards" markdown>

<div class="saos-card" markdown>
<div class="saos-card-icon">🧵</div>

### Free-Threaded Python

Built for Python 3.13t. Scales across CPU cores with no GIL bottleneck — essential for solar AO's wide-field computations.

</div>

<div class="saos-card" markdown>
<div class="saos-card-icon">🛤️</div>

### LightPath Abstraction

Each line of sight is an independent `LightPath` — a transparent pipeline connecting atmosphere, DM, and WFS.

</div>

<div class="saos-card" markdown>
<div class="saos-card-icon">☀️</div>

### Solar & Night-Time AO

Supports extended sources (Sun) via `CorrelatingShackHartmann` and point sources via classical `ShackHartmann`.

</div>

<div class="saos-card" markdown>
<div class="saos-card-icon">🔥</div>

### PyTorch Accelerated

Heavy image and FFT operations run on PyTorch — CPU today, GPU-ready tomorrow.

</div>

</div>

---

## Where to go next?

| | |
|---|---|
| **[Installation](installation.md)** | Set up your Python 3.13t environment and install SAOS. |
| **[Introduction tutorial](tutorials/01_intro_to_saos.ipynb)** | Step-by-step notebook: build your first SCAO simulation. |
| **[API Reference](api/index.md)** | Full documentation of every class and method. |

---

> **SAOS is based on [OOPAO](https://github.com/cheritier/OOPAO)** by Cédric Taïssir Héritier. The LightPath architecture and solar AO extensions are the main new contributions of SAOS.

