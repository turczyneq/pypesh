import pypesh.visualisation as visual
from pathlib import Path

parent_dir = Path(__file__).parent

savename = "pe100" + ".pdf"
tosave = parent_dir / "graphics" / savename
visual.visualise_trajectories_with_streamplot(100, 0.8, {0: 20}, save=tosave, show=True)

savename = "pe1000" + ".pdf"
tosave = parent_dir / "graphics" / savename
visual.visualise_trajectories_with_streamplot(1000, 0.8, {0: 20}, save=tosave, show=True)
