
# SPHEREx Sky Tracker (Team 10LY2NHUEr)


An automated infrared image-differencing pipeline for **SPHEREx Level 2** data products, built to detect moving celestial objects (asteroids) and variable sources across multi-epoch observations ($T_1, T_2$).

## Key Processing Architecture
1. **SPHEREx Level 2 Pre-processing:** Subtracts zodiacal background emission (`ZODI`) and masks cosmic-ray hits and detector artifacts using quality bitmasks (`FLAGS` bits 0–19).
2. **Two-Step Sub-Pixel WCS Alignment:** Reprojects epoch $T_2$ onto the $T_1$ World Coordinate System (WCS) grid via `reproject`, paired with sub-pixel registration refinement and median aperture flux normalization.
3. **Four-Layer Morphological & PSF Filtering (Default $9.0\sigma$):** Suppresses linear satellite trails, flat/elongated artifacts, sub-3-pixel noise spikes, and subtraction dipoles while validating candidates against the empirical Point Spread Function (PSF).
4. **Parallel Catalog Cross-Matching (SIMBAD):** Queries `simbad.harvard.edu` concurrently via `astroquery.simbad` ($30''$ search radius) to separate known static stars and cataloged sources from uncataloged moving candidates.

## Local Installation & Usage
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Data Attribution & Acknowledgements
- **SPHEREx Mission (NASA / JPL-Caltech):**
  > "This publication makes use of data products from the Spectro-Photometer for the History of the Universe, Epoch of Reionization and Ices Explorer (SPHEREx), which is a joint project of the Jet Propulsion Laboratory and the California Institute of Technology, and is funded by the National Aeronautics and Space Administration."
- **Archive & Astronomical Catalogs:** Level 2 FITS data products accessed via the **NASA/IPAC Infrared Science Archive (IRSA)**; astrometric cross-matching powered by the **SIMBAD** database, operated at CDS, Strasbourg, France.


*## AI Usage Disclosure
Generative AI assistants (Google Gemini, Anthropic Claude, Antigravity IDE) were used solely to assist with Python/Streamlit debugging, UI/CSS styling, and documentation refinement. All astrophysics algorithms, data processing logic, and candidate validations were developed and verified by Team 10LY2NHUEr.
