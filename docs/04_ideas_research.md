# Stage 4: Ideas research

27 research directions across 10 top-ranked problems. Three ideas per problem except BF-02, BF-16, and BF-18 (two each), where a third would duplicate rather than differentiate.

Prior art for each problem was searched via web queries in April 2026. Key sources cited inline.

---

## BF-03: Short-interval reburns

### BF-03-i: Time-since-fire reburn probability surfaces from Landsat archive

**Concept.** Build a pan-boreal reburn probability model that predicts, for each 30 m pixel, the likelihood that a previously burned area will reburn within the next 1-5 years. The model conditions on time-since-fire, spectral recovery trajectory, climate zone, and fire weather indices. Output: annually updated probability rasters.

**How it addresses the problem.** Fire managers currently have no spatial product telling them which burned areas are re-entering the flammable window. Tepley et al. (2025) quantified the fire-vegetation feedback at biome scale, showing that reburn resistance decays predictably as fuels rebuild. Converting this relationship into a gridded probability surface would give fire agencies a decision-support layer for pre-positioning resources and prioritizing fuel breaks.

**Sketched experimental design.**
- *Training data*: CanadaFireSat (Landsat burned area labels, 1985-present) merged with NBAC fire perimeters for fire history. Identify all pixels burned twice within the archive period. Extract spectral trajectories (NBR, dNBR, NDVI recovery curve) between fire events.
- *Predictor stack*: time-since-fire, NBR recovery rate, climate normals (growing degree days, annual precipitation), fire weather index (FWI) climatology from ERA5-Land, topographic wetness index from ArcticDEM/CDEM.
- *Model*: LightGBM or random forest for interpretability at pixel scale, with spatial cross-validation (leave-one-ecoregion-out). Compare against a spatiotemporal CNN operating on 5-year spectral recovery windows.
- *Validation*: hold out 2018-2024 fires and test whether reburn events fall in high-probability zones. Report AUC, calibration curve, and spatial transferability across ecoregions.
- *Study area*: Northwest Territories and Interior Alaska (highest reburn rates). Test transfer to Fennoscandia.
- *Ground truth*: ABoVE field dataset (1,538 sites, 58 fire perimeters) for fuel recovery validation.

**Source / gap.** Tepley et al. (2025, Ecosystems) and Whitman et al. (2024, Global Change Biology) provide the ecological basis. CanadaFireSat provides the labeled training data. No one has built a gridded reburn probability product from these ingredients.

**Data modalities.** Optical satellite (Landsat, Sentinel-2), climate reanalysis (ERA5-Land), topographic (ArcticDEM), field survey (ABoVE).

**Institutional fit.** Canadian Forest Service (Whitman's group, Northern Forestry Centre). Woodwell Climate (Rogers, Potter). Northern Arizona University (GEODE Lab, Goetz/Walker/Mack). CIFFC for operational deployment.

---

### BF-03-ii: Overwintering fire detection from Sentinel-2 pre-greenup imagery

**Concept.** Detect overwintering ("zombie") fires — fires that smolder through the winter in peat and organic soil, then re-emerge in spring — using early-spring Sentinel-2 imagery acquired before vegetation greenup. At this phenological window, thermal anomalies on snow or bare ground from holdover fires are spectrally distinct.

**How it addresses the problem.** Overwintering fires are a documented but poorly quantified source of reburns. Scholten et al. (2021) identified overwintering events from VIIRS, and Baltzer's group collected field samples at overwintering fire sites (the first team to do so). A systematic pre-greenup detection system would quantify their frequency and spatial distribution, closing a significant gap in the reburn pathway.

**Sketched experimental design.**
- *Data*: Sentinel-2 Level-2A imagery in the pre-greenup window (late April to mid-May in Canada/Alaska, mid-April to early May in Fennoscandia). Acquire 3-5 years of spring imagery over known overwintering fire sites.
- *Labeled events*: VIIRS-detected overwintering fires from Scholten et al. (2021, 2024) as positive labels. Random snow-covered pixels as negatives.
- *Method*: Change detection between fall post-fire imagery and spring pre-greenup imagery. Train a pixel-level classifier (logistic regression or small CNN) on spectral bands + thermal anomaly + NDSI (snow index). Test whether overwintering sites show anomalous SWIR reflectance or early snowmelt.
- *Validation*: field-verified overwintering fire sites from Baltzer's group. ABoVE soil temperature loggers for subsurface thermal validation.
- *Study area*: NWT, northern Alberta, Interior Alaska (where overwintering fires are reported most frequently).

**Source / gap.** Scholten et al. (2021, Nature, circumpolar fire tracking) identified overwintering events from VIIRS active fire data. Baltzer's group (Wilfrid Laurier) collected the first field samples. No one has attempted systematic spatial detection from optical imagery. The pre-greenup phenological window is unexploited.

**Data modalities.** Optical satellite (Sentinel-2), active fire (VIIRS), field survey (Baltzer lab, ABoVE), climate reanalysis (ERA5 snow depth).

**Institutional fit.** VU Amsterdam (Veraverbeke, Scholten — FireIce ERC project). Wilfrid Laurier (Baltzer). Woodwell Climate.

---

### BF-03-iii: Conifer-to-deciduous conversion mapping as a reburn regime indicator

**Concept.** Map post-fire vegetation type conversion (conifer → deciduous) at 30 m resolution across the boreal zone using Landsat time series and spectral unmixing. Areas that have converted from conifer to deciduous after fire have lower fuel loads and altered fire behavior, meaning the reburn feedback loop has been broken. Conversely, areas that regenerate as conifer within 10-15 years re-enter the high-flammability window.

**How it addresses the problem.** The reburn problem is fundamentally about the rate at which fuels rebuild. Whether a burned area regenerates as conifer (high flammability, high crown fire potential) or deciduous (lower flammability, surface fire regime) determines its reburn trajectory. Mapping this conversion at scale would separate burned areas into "reburn-prone" (conifer regeneration) and "reburn-resistant" (deciduous conversion) categories.

**Sketched experimental design.**
- *Method*: Extend Zheng et al.'s (2024, RSE) regression-based spectral unmixing approach from the 1987 Siberian megafire to pan-boreal application. Combine with the needleleaf index (2025, npj Natural Hazards) for conifer fraction estimation. Produce decadal conifer/deciduous fraction maps for all fire perimeters in the Landsat archive.
- *Training/validation*: ABoVE post-fire regeneration dataset (1,538 sites with species composition data). Berner & Goetz (2022) satellite vegetation time series.
- *Analysis*: correlate vegetation type conversion with subsequent fire occurrence. Test whether conifer regeneration rate predicts next-fire probability.
- *Study area*: NWT, Interior Alaska, northern Quebec (where conversion has been documented).

**Source / gap.** Zheng et al. (2024) demonstrated the approach for one megafire. Walker/Mack (2019) documented conversion at field scale. The needleleaf index (2025) provides the spectral tool. No pan-boreal application combining these ingredients exists.

**Data modalities.** Optical satellite (Landsat), field survey (ABoVE, Walker/Mack datasets).

**Institutional fit.** Northern Arizona University (GEODE Lab — Walker, Mack, Goetz). University of Maryland (GLAD — Feng, Sexton). Canadian Forest Service.

---

## BF-09: Treeline ecotone resolution

### BF-09-i: 10 m treeline ecotone structure from ICESat-2 + Sentinel-2 fusion

**Concept.** Produce a continuous treeline ecotone product at 10 m resolution that maps canopy height, cover fraction, and stem density gradient across the forest-tundra transition zone. Fuse ICESat-2 ATL08 (sparse but accurate height profiles) with Sentinel-2 spectral data (wall-to-wall coverage) using a deep learning regression model.

**How it addresses the problem.** Current tree cover products (Hansen, Feng) are binary or coarse. The ecotone is a gradient: from closed canopy forest through open woodland to krummholz to tundra. Mapping this gradient at 10 m captures the structural changes that indicate where treeline is advancing, stalling, or retreating. A height + cover product is more ecologically informative than tree cover alone.

**Sketched experimental design.**
- *Data*: ICESat-2 ATL08 canopy height profiles as training targets. Sentinel-2 Level-2A surface reflectance as input features (10 bands at 10-20 m). ArcticDEM for topographic features.
- *Model*: Encoder-decoder network (UNet-style) with ICESat-2 transects as sparse supervision. Loss function: regression loss on height and cover fraction where ICESat-2 data exists, spatial consistency regularization elsewhere. Compare against random forest baseline (as in Feng et al. 2026) and the 2025 deep learning fusion approach (MAE 1.42 m).
- *Validation*: ALS strips from national surveys (Canada, Finland, Sweden) as independent height reference. Field plots from Danby (Queen's) and Berner (NAU).
- *Study area*: three transects spanning the treeline — northern Quebec (steep gradient), NWT/Yukon (moderate gradient), Finnish Lapland (gentle gradient).
- *Temporal dimension*: produce 2017-2025 annual maps to track ecotone shift rate.

**Source / gap.** ICESat-2 ecotone study (2024, 88% detection accuracy). Deep learning fusion (2025, MAE 1.42 m). Neither produced wall-to-wall ecotone mapping at the structural-gradient level.

**Data modalities.** Spaceborne LiDAR (ICESat-2), optical satellite (Sentinel-2), DEM (ArcticDEM), airborne LiDAR (national surveys).

**Institutional fit.** SLU (Remote Sensing Division — boreal forest mapping, ALS expertise). NASA Goddard (ICESat-2 mission science, HLS). Northern Arizona University (Berner, Goetz — satellite vegetation time series). University of Maryland (GLAD).

---

### BF-09-ii: Treeline advance rate estimation from 40-year Landsat time series

**Concept.** Estimate treeline advance rates (meters per decade) at circumpolar scale using the full 1984-2025 Landsat archive. Apply breakpoint detection and trend analysis to spectral indices (NDVI, NBR) along north-south transects crossing the forest-tundra ecotone. Attribute advance rates to local climate drivers (growing degree days, snow-free season length, soil temperature).

**How it addresses the problem.** Feng et al. (2026) showed net northward shift in tree cover but did not quantify advance rates at the granularity needed for ecological forecasting. Knowing the rate — and where it's accelerating, stalling, or reversing — is the input that treeline-shift models need.

**Sketched experimental design.**
- *Data*: Landsat Collection 2 Level-2 surface reflectance, 1984-2025. ERA5-Land for climate covariates. MODIS land surface phenology for snow-free season. ArcticDEM for slope/aspect.
- *Method*: For each 30 m pixel in the ecotone zone (defined by Feng et al. tree cover maps), compute annual peak-season NDVI. Apply BFAST (Breaks For Additive Seasonal and Trend) for breakpoint detection. Extract trend slopes for segments between breakpoints. Classify pixels into advancing/stable/retreating.
- *Attribution*: correlate advance rates with growing degree day trends (ERA5-Land), snow-free season length (MODIS), and permafrost probability (ESA CCI Permafrost).
- *Validation*: compare against dendrochronological records (tree ring dating of treeline establishment) and photo-repeat stations.
- *Study area*: circumpolar. Stratify by continentality (maritime Fennoscandia vs. continental Siberia vs. continental Canada).

**Source / gap.** Berner et al. (2022, Global Change Biology) documented biome shift from Landsat. Feng et al. (2026) produced 30 m maps. Neither produced spatially explicit advance rates with climate attribution. BFAST-based trend analysis has been applied to tropical deforestation but not systematically to treeline.

**Data modalities.** Optical satellite (Landsat), climate reanalysis (ERA5-Land), land surface phenology (MODIS), DEM (ArcticDEM).

**Institutional fit.** University of Maryland (GLAD — Feng, Sexton, Hansen). Northern Arizona University (Berner, Goetz). Tsinghua University (Feng). SLU.

---

### BF-09-iii: Krummholz detection from very high resolution satellite imagery

**Concept.** Map individual krummholz patches (stunted, wind-shaped conifers) in the treeline ecotone using very high resolution (VHR, <1 m) commercial satellite imagery (Planet SkySat, Maxar) and object detection deep learning. Krummholz presence indicates the leading edge of potential treeline advance — these patches are the "seedbed" for forest expansion but are invisible at Sentinel-2/Landsat resolution.

**How it addresses the problem.** The ecotone's advancing front is not a line but a scattered field of isolated krummholz patches that gradually densify into woodland over decades. Mapping these individual patches gives the earliest detectable signal of where treeline will advance next, years before canopy cover becomes visible at 10-30 m resolution.

**Sketched experimental design.**
- *Data*: Planet SkySat (50 cm) or Maxar WorldView (31 cm) imagery over ecotone transects. Label krummholz patches from drone orthomosaics (5-10 cm) collected during field campaigns.
- *Model*: object detection (YOLOv8 or Faster R-CNN) trained on drone-labeled VHR imagery. Transfer to new ecotone sites. Test generalization across biogeographic regions.
- *Validation*: field GPS-mapped krummholz locations. Compare patch density against local climate and microtopographic variables.
- *Study area*: pilot at 3-4 sites with existing drone data (likely Finnish Lapland, NWT, northern Quebec).

**Source / gap.** No published work maps individual krummholz patches from satellite imagery. Danby (Queen's) and Macias-Fauria (Oxford) have field-based krummholz studies. VHR satellite-based detection is a gap.

**Data modalities.** VHR optical satellite (Planet SkySat, Maxar), drone imagery, field GPS survey.

**Institutional fit.** SLU (Remote Sensing Division). Queen's University (Danby — treeline ecology). University of Oxford (Macias-Fauria). Planet Labs (data access).

---

## BF-01: Small and low-intensity fire detection

### BF-01-i: SAR-optical fusion for sub-100 ha burned area detection

**Concept.** Develop a multi-sensor burned area detection system that fuses Sentinel-1 C-band SAR (cloud-independent, 6-day revisit) with Sentinel-2 optical (10 m, cloud-limited) for detecting fires under 100 ha across the boreal zone. SAR provides detection capability during persistent cloud/smoke cover; optical provides higher spectral discrimination for burn severity.

**How it addresses the problem.** Existing deep learning approaches (Potter 2026, BiAU-Net) use optical data only and fail during cloud/smoke episodes that can last days during active fire events. Hall et al. (2021) showed half of boreal burned area goes undetected — most of these are small fires missed during cloudy periods. A SAR component removes the cloud dependency.

**Sketched experimental design.**
- *Data*: Sentinel-1 GRD (VV/VH backscatter) pre- and post-fire pairs. Sentinel-2 Level-2A dNBR as optical complement. NBAC fire perimeters as labels, filtered to fires <100 ha.
- *Model*: dual-encoder architecture — one branch for SAR (temporal backscatter change), one for optical (spectral indices) — with late fusion. Handle missing optical data (clouds) via attention masking. Train on 2017-2023 fire seasons; validate on 2024.
- *Baselines*: Potter et al. UNet++ (optical only), Sentinel-1 backscatter threshold (SAR only).
- *Evaluation*: F1, IoU, and detection rate stratified by fire size (<10 ha, 10-50 ha, 50-100 ha). Report cloud-cover detection gap: for how many fire events does SAR provide the first detection?
- *Study area*: Canada (NBAC-covered) and Sweden (SMHI system for comparison).

**Source / gap.** Potter (2026) achieved F1=0.85 with optical only. Finnish SAR study (2021) demonstrated SAR windthrow detection at 79% accuracy. No published SAR-optical fusion specifically for small boreal fire detection.

**Data modalities.** SAR (Sentinel-1 C-band), optical satellite (Sentinel-2), fire perimeters (NBAC).

**Institutional fit.** VU Amsterdam (Veraverbeke, Scholten — circumpolar fire atlas). SMHI (Hassellof — operational system). Woodwell Climate (Potter). Canadian Forest Service.

---

### BF-01-ii: Active fire detection at high latitudes from Sentinel-3 SLSTR night-time thermal

**Concept.** Exploit Sentinel-3 SLSTR's dual-view thermal channels (1 km, daily revisit at high latitudes) for detecting active fires in the boreal zone. At latitudes above 55°N, Sentinel-3's near-polar orbit provides multiple daily overpasses. Night-time acquisitions eliminate solar reflection contamination and increase thermal contrast between fire and background. Current VIIRS-based detection has known gaps at high latitudes.

**How it addresses the problem.** VIIRS provides 375 m active fire detection but has gaps in temporal coverage, especially at latitudes where scan overlap is minimal. Sentinel-3 SLSTR's dual-view geometry (nadir + oblique) provides independent thermal measurements that can reduce false alarm rates. The high revisit frequency at boreal latitudes (up to 4 passes/day) could close the detection gap.

**Sketched experimental design.**
- *Data*: Sentinel-3 SLSTR Level-1 thermal bands (S7-S9, F1-F2), 2018-2025. VIIRS active fire products as reference labels. ERA5 surface temperature for background estimation.
- *Method*: contextual fire detection algorithm adapted for SLSTR's spectral configuration. Compare single-view (nadir) vs. dual-view detection to quantify the value of the oblique view. Train a neural network detector (1D CNN on spectral/temporal features) to distinguish fire from warm-ground false alarms.
- *Validation*: co-located VIIRS detections. Known fire start times from agency records. SMHI's validated detections for Sweden.
- *Study area*: pan-boreal, focusing on latitudes 55-70°N.

**Source / gap.** SLSTR fire products exist for tropical regions but are not optimized for high-latitude boreal conditions. The dual-view advantage for reducing false positives in boreal landscapes (lake glint, heated rock faces, industrial heat sources) has not been systematically evaluated.

**Data modalities.** Thermal satellite (Sentinel-3 SLSTR), active fire (VIIRS), climate reanalysis (ERA5).

**Institutional fit.** SMHI (operational fire detection expertise). ESA (SLSTR mission). VU Amsterdam (fire tracking).

---

### BF-01-iii: Low-intensity prescribed burn verification from Sentinel-2 time series in Fennoscandia

**Concept.** Build a monitoring system that verifies whether prescribed burns in Fennoscandian boreal forests achieved their ecological objectives (opening canopy, exposing mineral soil, reducing fuel load) by tracking post-burn spectral trajectories at 10 m resolution. Currently, prescribed burn outcomes are verified only through costly field visits months after burning.

**How it addresses the problem.** Low-intensity prescribed burns used for conservation in Finland and Sweden are essentially invisible to operational fire detection (MODIS/VIIRS). They don't need to be detected in real time — they're planned events. But their ecological outcomes need verification: did the burn create the intended mosaic of burned and unburned patches? Spectral trajectory analysis can distinguish successful burns (exposed mineral soil → early colonizers → target vegetation) from failed burns (canopy closure, no soil exposure).

**Sketched experimental design.**
- *Data*: Sentinel-2 time series (2017-2025) over documented prescribed burn sites in Finland and Sweden. Burn locations from Luke (Finland) and county forestry boards (Sweden).
- *Method*: extract multi-year spectral trajectories (dNBR, NDVI, SWIR bands) for burned polygons. Classify burn outcomes into success categories (full mosaic, partial, failed) using trajectory shape features. Compare against field assessment data.
- *Validation*: field-measured burn severity and regeneration data from Luke and SLU prescribed burn monitoring programs.
- *Study area*: Finland and Sweden — the two countries with operational prescribed burning programs in the boreal zone.

**Source / gap.** Prescribed burn monitoring in Fennoscandia is entirely field-based. The satellite community has focused on wildfire detection, not verification of planned burns. This is a practical gap with clear institutional demand.

**Data modalities.** Optical satellite (Sentinel-2), field survey (Luke, SLU prescribed burn records).

**Institutional fit.** Luke (Finland — prescribed burn programs). SLU (Sweden — forest management). University of Helsinki (boreal ecology).

---

## BF-06: Permafrost thaw in forested regions

### BF-06-i: NISAR L-band InSAR subsidence mapping in forested discontinuous permafrost

**Concept.** Apply NISAR L-band InSAR to measure seasonal and multi-year permafrost subsidence in the forested discontinuous permafrost zone — the terrain where C-band Sentinel-1 InSAR fails due to canopy-induced coherence loss. L-band's longer wavelength (23.6 cm vs. C-band 5.6 cm) penetrates the canopy and maintains coherence in treed terrain.

**How it addresses the problem.** Zwieback (2024) identified coherence loss in forested areas as the primary barrier to satellite InSAR permafrost monitoring. UAVSAR airborne L-band demonstrated the approach works (Michaelides 2021), but no satellite-based L-band InSAR has been applied in boreal permafrost. NISAR, operational since early 2026, provides the first opportunity.

**Sketched experimental design.**
- *Data*: NISAR L-band SLC (single-look complex) acquisitions over the first full annual cycle (2026-2027). Sentinel-1 C-band InSAR over the same period for comparison. ArcticDEM for topographic correction.
- *Method*: SBAS (Small Baseline Subset) InSAR time series analysis. Estimate seasonal subsidence amplitude and multi-year subsidence trend. Compare L-band vs. C-band coherence as a function of canopy density (estimated from Sentinel-2 NDVI). Test whether L-band recovers the subsidence signal in forested areas where C-band fails.
- *Validation*: borehole temperature profiles from GTN-P. UAVSAR L-band airborne data (ABoVE campaign) as calibration reference. In-situ GNSS-based subsidence measurements if available.
- *Study area*: Interior Alaska (Fairbanks corridor) — where Zwieback's group operates, extensive borehole network, and UAVSAR reference data exist. The discontinuous permafrost zone is the most dynamic and most forested.

**Source / gap.** Sadeghi Chorsi (2024) demonstrated Sentinel-1 InSAR in tundra. Zwieback (2024) identified the forest coherence problem. UAVSAR proved L-band works. No satellite L-band InSAR in forested permafrost exists yet.

**Data modalities.** SAR (NISAR L-band, Sentinel-1 C-band), DEM (ArcticDEM), in-situ (borehole temperature, GNSS), airborne SAR (UAVSAR).

**Institutional fit.** University of Alaska Fairbanks (Zwieback, Meyer — Geophysical Institute, ASF DAAC). NASA JPL (NISAR mission, ARIA/OPERA products). ESA CCI Permafrost (Bartsch).

---

### BF-06-ii: Multi-sensor permafrost vulnerability mapping integrating InSAR, optical, and thermal data

**Concept.** Combine InSAR-derived subsidence rates with optical indicators (NDVI decline, wetland expansion) and thermal data (land surface temperature anomalies from ECOSTRESS or Landsat TIRS) to produce a permafrost vulnerability index at landscape scale. Each sensor captures a different dimension of thaw: subsidence (InSAR), vegetation response (optical), and surface energy balance (thermal).

**How it addresses the problem.** InSAR alone measures displacement but doesn't distinguish cause (seasonal thaw vs. irreversible degradation). Optical indices detect vegetation change but can't measure ground movement. Thermal data captures the energy flux driving thaw but not the resulting deformation. Integrating all three disambiguates the thaw signal and identifies areas transitioning from seasonal freeze-thaw to irreversible permafrost loss.

**Sketched experimental design.**
- *Data*: Sentinel-1 InSAR time series (SBAS), Landsat/Sentinel-2 NDVI and NDWI time series, Landsat TIRS or ECOSTRESS land surface temperature. ERA5-Land soil temperature and soil moisture as covariates.
- *Method*: pixel-level fusion using a random forest or gradient boosting model trained on labeled permafrost state (stable, degrading, thawed) from borehole records and expert interpretation. Produce a vulnerability index raster with uncertainty estimates.
- *Validation*: GTN-P borehole network. Existing permafrost maps (e.g., ESA CCI Permafrost).
- *Study area*: start with Yukon (Teslin area — well-studied, accessible, discontinuous permafrost with mixed forest).

**Source / gap.** Individual sensors have been applied. No published multi-sensor fusion product integrates InSAR + optical + thermal for permafrost vulnerability classification in forested terrain.

**Data modalities.** SAR (Sentinel-1, NISAR), optical satellite (Landsat, Sentinel-2), thermal satellite (Landsat TIRS, ECOSTRESS), climate reanalysis (ERA5-Land), in-situ (borehole).

**Institutional fit.** University of Alaska Fairbanks (Zwieback, Meyer). NASA JPL (ECOSTRESS, NISAR). ESA CCI Permafrost (Bartsch). NRCan (Geological Survey of Canada, permafrost mapping).

---

### BF-06-iii: Fire-permafrost interaction mapping: post-fire thaw acceleration from InSAR

**Concept.** Quantify post-fire permafrost thaw acceleration by comparing InSAR subsidence rates inside and outside fire perimeters in the discontinuous permafrost zone. Fire removes the insulating organic layer, increasing ground heat flux and accelerating permafrost degradation. This creates a direct link between the fire regime problems (BF-03) and permafrost thaw.

**How it addresses the problem.** The fire-permafrost feedback is documented in field studies but not measured at landscape scale. Veraverbeke's FireIce project explicitly targets this interaction, but satellite-scale subsidence mapping inside fire perimeters has not been demonstrated.

**Sketched experimental design.**
- *Data*: NISAR L-band InSAR over areas burned 2020-2025 in Interior Alaska. Pre-fire Sentinel-1 InSAR for baseline comparison (where coherence is sufficient). Fire perimeters from NBAC / Alaska Interagency Coordination Center.
- *Method*: compute annual subsidence rates inside fire perimeters vs. adjacent unburned control areas (matched by terrain, vegetation type, permafrost zone). Test whether subsidence increases as a function of time-since-fire and burn severity (dNBR).
- *Validation*: Yoshikawa & Hinzman (2003) field measurements of active layer deepening after fire. ABoVE post-fire soil temperature data.
- *Study area*: Interior Alaska (Fairbanks area, Tanana Flats), NWT.

**Source / gap.** The fire-permafrost interaction is well-documented in field studies (Yoshikawa & Hinzman, 2003; Gibson et al., 2018). InSAR-based measurement of post-fire subsidence was demonstrated with UAVSAR in one study (Li et al., 2018, Remote Sensing). Satellite-scale measurement with NISAR is the natural next step.

**Data modalities.** SAR (NISAR L-band, Sentinel-1), optical satellite (Sentinel-2 for burn severity), fire perimeters, field survey (ABoVE, borehole data).

**Institutional fit.** VU Amsterdam (Veraverbeke — FireIce ERC project, fire-permafrost feedbacks). University of Alaska Fairbanks (Zwieback, Meyer). Woodwell Climate (fire-carbon-permafrost research).

---

## BF-18: Storm damage detection

### BF-18-i: NISAR L-band windthrow detection and severity mapping

**Concept.** Apply NISAR L-band SAR to detect and classify windthrow severity in boreal forests. L-band penetrates the canopy, meaning pre-storm L-band backscatter captures the trunk/branch layer. Post-storm L-band backscatter changes reflect actual tree fall (trunk removal, gap creation) rather than just canopy roughness changes visible in C-band. This should give better severity classification (standing dead, partial throw, complete windthrow) than Sentinel-1.

**How it addresses the problem.** The Finnish SAR study (2021) achieved 79% accuracy with Sentinel-1 C-band, which primarily senses the upper canopy. L-band provides complementary structural information. NISAR's 12-day revisit cycle enables rapid detection, and the free/open data policy removes access barriers.

**Sketched experimental design.**
- *Data*: NISAR L-band pre-storm and post-storm acquisitions over windstorm events. Sentinel-1 C-band over the same events for comparison. National forest inventory data (Luke, SLU) for baseline forest structure.
- *Method*: change detection on calibrated backscatter (VV, VH, VV/VH ratio). Train a severity classifier (random forest or CNN on backscatter change patches) using ALS-derived damage maps as training labels. Compare L-band vs. C-band detection rates across severity classes.
- *Validation*: ALS surveys conducted by Luke/SLU after major storms. Forest inventory re-measurements.
- *Study area*: Finland and Sweden (regular storm events, excellent reference data from national forest inventories, ALS coverage).

**Source / gap.** C-band SAR windthrow detection demonstrated (Finland 2021). Deep learning on Sentinel-2 demonstrated (RSE 2026). L-band SAR for windthrow severity classification is untested but physically motivated by the volume scattering mechanism.

**Data modalities.** SAR (NISAR L-band, Sentinel-1 C-band), airborne LiDAR (ALS), national forest inventory.

**Institutional fit.** Luke (Finland — national forest inventory, ALS, storm damage records). VTT (Finland — SAR processing). SLU (Sweden). Chalmers University (Ulander — L-band SAR expertise). ICEYE (rapid SAR revisit for operational comparison).

---

### BF-18-ii: Near-real-time windthrow alerting pipeline from Sentinel-1 time series

**Concept.** Build an operational alerting system that detects windthrow events within 48 hours of a storm using Sentinel-1 temporal change detection and pushes alerts to forest management agencies. The system runs continuously, processing new Sentinel-1 acquisitions against a stable baseline stack.

**How it addresses the problem.** Current windthrow assessment is done retrospectively (weeks to months after storm). Rapid detection enables faster salvage logging, which reduces bark beetle outbreak risk (beetles colonize windthrown timber within weeks) and recovers timber value.

**Sketched experimental design.**
- *Data*: Sentinel-1 GRD time series (6-day revisit in Europe). Pre-storm baseline stack (12-month moving average of VV, VH backscatter). ECMWF wind speed forecasts to trigger processing for specific regions.
- *Method*: for each new Sentinel-1 acquisition after a wind event (triggered by ECMWF wind speed exceeding a threshold), compute pixel-level z-score against the baseline stack. Flag pixels with z-score > 3 in VH (cross-pol, sensitive to volume scattering change). Cluster flagged pixels into damage polygons. Push alerts via API.
- *Latency*: Sentinel-1 data available via Copernicus within 3-6 hours of acquisition. Processing: ~1 hour per tile. Total: 48-hour target.
- *Validation*: historical storm events in Finland (compare alert timing against Luke's assessment reports).
- *Study area*: Finland and Sweden initially (dense Sentinel-1 coverage, excellent reference data). Expand to Canada/Norway.

**Source / gap.** Rapid Sentinel-1 detection demonstrated at research level (producer's accuracy 0.88). No operational alerting pipeline exists for boreal windthrow. The triggering mechanism (wind forecast → targeted processing) reduces computational cost.

**Data modalities.** SAR (Sentinel-1), weather forecast (ECMWF), national forest inventory.

**Institutional fit.** Luke (Finland — end user). SMHI/MSB (Sweden — operational satellite monitoring). VTT (SAR processing). ICEYE (operational benchmarking).

---

## BF-19: Bioacoustic monitoring

### BF-19-i: Boreal-specific BirdNET fine-tuning with circumpolar training data

**Concept.** Fine-tune BirdNET (or its embedding layer, following the ArcticSoundsNet approach) on a curated dataset of boreal bird vocalizations spanning the circumpolar belt. Current BirdNET performance degrades in boreal environments because its training data underrepresents boreal species assemblages, vocalizations during midnight sun conditions (continuous daylight), and soundscapes with heavy wind/insect noise.

**How it addresses the problem.** The Yukon BBMP study showed BirdNET achieves comparable detection to human observers at 22% of the cost, but only after intensive validation filtering. A boreal-specific fine-tuned model would reduce the validation burden, increase detection rates for hard-to-identify species (e.g., warblers with similar songs, owls recorded at distance), and enable deployment across the circumpolar belt rather than single-region validation.

**Sketched experimental design.**
- *Training data*: aggregate recordings from Xeno-canto (filtered for boreal species), Finnish Museum of Natural History (Luomus) archives, Canadian Wildlife Service Breeding Bird Survey recordings, and the Boreal Bird Monitoring Program (Yukon). Target: 100+ boreal species, 500+ recordings per species, balanced across circumpolar regions.
- *Method*: extract BirdNET embeddings (penultimate layer). Fine-tune a classification head on boreal species subset. Alternatively, full fine-tuning of last N layers with region-specific augmentation (wind noise, rain, insect chorus overlays). Compare against ArcticSoundsNet baseline.
- *Evaluation*: species-level precision/recall on held-out PAM recordings from 3 regions (Finland, Yukon, Alaska). Report performance stratified by SNR, time of day, and species rarity.
- *Study area*: three-region validation: Finnish Lapland, Yukon, Interior Alaska.

**Source / gap.** BirdNET fine-tuning for Italian Alps (2025) demonstrated the approach. ArcticSoundsNet adapted embeddings for Arctic species. No fine-tuning specifically targeting the circumpolar boreal species assemblage.

**Data modalities.** Audio (passive acoustic monitoring recordings), species occurrence (eBird, BBS).

**Institutional fit.** Cornell Lab of Ornithology (Kahl, Klinck — BirdNET team). Finnish Museum of Natural History (Luomus). Environment and Climate Change Canada (Canadian Wildlife Service). Chemnitz University of Technology (Kahl).

---

### BF-19-ii: Foundation model transfer for multi-taxon boreal acoustic monitoring

**Concept.** Apply a general-purpose audio foundation model (e.g., BioLingual, AVES, or a large-scale audio SSL model) to classify boreal soundscapes across multiple taxa — birds, amphibians, and mammals — in a single pipeline. BirdNET handles bird classification well but misses the broader acoustic biodiversity picture. Foundation models trained on large, diverse audio corpora may generalize better to non-bird taxa.

**How it addresses the problem.** Boreal biodiversity monitoring focuses on birds because that's what BirdNET can identify. But boreal soundscapes also contain ecologically important signals: frog choruses (spring phenology indicators), moose calls (population density proxies), wolf howls (apex predator monitoring), and insect abundance (background buzz levels). A multi-taxon classifier captures biodiversity information that single-taxon tools miss.

**Sketched experimental design.**
- *Data*: PAM deployments from existing monitoring programs (BBMP Yukon, Finnish bird monitoring, Swedish county-level ARU networks). Supplement with Xeno-canto and GBIF audio for mammal/amphibian species.
- *Method*: evaluate multiple foundation models (AVES, BioLingual, Audio-MAE) on boreal multi-taxon classification. Fine-tune the best-performing model on labeled boreal recordings spanning birds, amphibians, and mammals. Compare against BirdNET (birds only) + separate taxon-specific classifiers.
- *Evaluation*: species-level F1 across taxa. Report the marginal biodiversity information gained by including non-bird taxa. Compute acoustic diversity indices (ACI, bioacoustic index) and correlate with field-measured species richness.
- *Study area*: Finnish Lapland (long-term monitoring sites with concurrent field surveys).

**Source / gap.** Foundation models for bioacoustics reviewed (2026, Ecological Informatics) — comparative study exists but not applied to boreal. ArcticSoundsNet is bird-only. No multi-taxon boreal classifier exists.

**Data modalities.** Audio (PAM recordings), species occurrence (eBird, GBIF), field survey (species richness counts).

**Institutional fit.** Cornell Lab of Ornithology (Klinck). Tilburg University (Stowell — computational bioacoustics, foundation models). University of Helsinki (Luomus). WCS Canada (boreal biodiversity monitoring).

---

### BF-19-iii: Acoustic phenology tracking as a climate indicator in boreal forests

**Concept.** Use continuous PAM recordings to track the timing of acoustic events — spring dawn chorus onset, first frog calls, peak insect buzz, bird migration arrival dates — as phenological indicators of climate change in boreal forests. Acoustic phenology can be measured passively at high temporal resolution (hourly) and compared against traditional phenological records.

**How it addresses the problem.** Phenological shifts are among the most sensitive indicators of climate change in boreal ecosystems, but traditional phenology monitoring relies on human observers and has sparse spatial coverage. PAM can fill this gap by detecting phenological milestones acoustically. If acoustic phenology correlates with ecological-process phenology (leaf-out, snowmelt, insect emergence), it becomes a scalable, automated climate indicator.

**Sketched experimental design.**
- *Data*: multi-year PAM deployments (3+ years of continuous recording) at sites with concurrent phenological observation (Finnish Phenology Network, Pan-European Phenology network).
- *Method*: automated detection of acoustic phenological events using BirdNET (bird migration arrivals) + custom detectors (frog call onset, insect buzz onset). Compute annual timing of each event. Correlate with temperature, snow cover, and green-up timing (MODIS phenology).
- *Evaluation*: compare acoustic phenology dates against traditional observer records. Report trend significance over the recording period. Test whether acoustic phenology predicts ecological phenology (leaf-out, insect emergence) better than temperature alone.
- *Study area*: Hyytiala SMEAR II station (Finland, longest-running boreal ecosystem monitoring). Supplement with 2-3 additional Nordic PAM sites.

**Source / gap.** Acoustic phenology is an emerging concept (2024 PNAS paper on passive acoustic data as phenological distributions). Not applied specifically to boreal climate monitoring. The long-running SMEAR II station provides the ideal ground-truth context.

**Data modalities.** Audio (PAM recordings), phenology observations (Finnish Phenology Network), land surface phenology (MODIS), climate (ERA5).

**Institutional fit.** University of Helsinki (INAR — Hyytiala SMEAR II station, Vesala). Cornell Lab of Ornithology (BirdNET, acoustic monitoring). Luke (Finnish phenology). SLU.

---

## BF-16: Intact forest landscape protection

### BF-16-i: IFL fragmentation risk forecasting under fire and climate scenarios

**Concept.** Build a spatiotemporal model that projects IFL fragmentation 10-30 years into the future under CMIP6 climate scenarios and fire regime projections. Current IFL monitoring (GLAD/Potapov) is retrospective — it reports loss after it happens. A forward-looking model would identify IFLs most at risk of crossing the 50,000 ha threshold (below which they lose IFL status) and prioritize protection before fragmentation occurs.

**How it addresses the problem.** 7% of global IFL area has been lost since 2000. In boreal regions, fire is the primary driver in Russia and Canada, while logging and road building dominate in Fennoscandia. Under projected fire regime intensification (2-3x burned area by 2100 in Canadian boreal), current IFLs will experience increasing fragmentation pressure. Identifying which IFLs are most vulnerable allows protection resources to be targeted before loss occurs rather than documented afterward.

**Sketched experimental design.**
- *Data*: IFL maps (GLAD, 2000/2013/2020). Global Forest Change (Hansen et al.) annual loss data. CMIP6 fire projections (FWI-based). Road network data (OpenStreetMap, government databases). Mining/forestry concession boundaries.
- *Model*: spatiotemporal random forest or gradient boosting predicting per-pixel probability of IFL loss in each 5-year window. Predictors: distance to IFL edge, distance to nearest road, fire weather projections, logging concession proximity, historical loss rate in surrounding area.
- *Scenarios*: run under SSP2-4.5 and SSP5-8.5 for fire; combine with BAU vs. expanded-protection scenarios for logging/roads.
- *Validation*: use 2000-2013 data to predict 2013-2020 loss, compare against observed.
- *Study area*: Canadian boreal (Ontario, Quebec — active logging frontier) and Russian boreal (where fire is the primary driver).

**Source / gap.** Potapov et al. produce the IFL dataset and documented loss rates. No forward-looking fragmentation risk model exists. Zonation/prioritizr are used for static prioritization, not dynamic risk projection.

**Data modalities.** Optical satellite (Landsat, via Global Forest Change), climate projections (CMIP6), infrastructure (OpenStreetMap), concession boundaries.

**Institutional fit.** University of Maryland (GLAD — Potapov, Hansen, Turubanova). WRI (Global Forest Watch). University of Helsinki (Moilanen — Zonation). WCS Canada.

---

### BF-16-ii: Connectivity-optimized IFL protection planning with Zonation 5

**Concept.** Apply Zonation 5's corridor retention and connectivity algorithms to identify the minimum set of unprotected boreal forest that, if protected, would maintain landscape connectivity between existing IFLs and protected areas. Optimize for ecological connectivity (species dispersal corridors, watershed integrity) rather than area alone.

**How it addresses the problem.** Current protection strategies focus on individual IFLs as isolated units. But IFL ecological value depends on connectivity — fragmented IFLs lose species that require large ranges (wolverine, caribou, bear). Zonation 5's boundary length penalty and corridor algorithms can identify bottleneck areas where protection of a small additional area maintains connectivity across large regions.

**Sketched experimental design.**
- *Data*: IFL boundaries (GLAD). Protected area boundaries (WDPA). Caribou/wolverine habitat models (NatureServe, ECCC species-at-risk maps). Resistance surface from land cover (Sentinel-2), road density, and terrain ruggedness.
- *Method*: Zonation 5 analysis with connectivity weighting. Identify the top 5% of unprotected boreal area that maximizes IFL connectivity. Compare against protection scenarios based on area alone (no connectivity weighting) and random selection.
- *Evaluation*: compare landscape connectivity metrics (effective mesh size, patch cohesion index, resistance-weighted distance between IFLs) across scenarios.
- *Study area*: Ontario Clay Belt (fragmented IFL landscape with active logging, caribou habitat) and northern Finland/Sweden (IFL-protected area interface).

**Source / gap.** Zonation applied to boreal retention forestry (Frontiers, 2020) but not to IFL connectivity. prioritizr (2025) provides complementary optimization but lacks Zonation's connectivity methods. No published connectivity-optimized IFL protection analysis exists for the boreal zone.

**Data modalities.** Optical satellite (Sentinel-2 land cover), protected area boundaries (WDPA), species habitat models, road network, DEM.

**Institutional fit.** University of Helsinki (Moilanen — Zonation development). WCS Canada (boreal conservation, caribou habitat). Pew Charitable Trusts (Boreal Conservation Campaign). TNC Canada.

---

## BF-02: Fire spread prediction

### BF-02-i: SAR-based active fire perimeter tracking for cloud-independent spread monitoring

**Concept.** Use Sentinel-1 SAR backscatter change to track active fire perimeter evolution through cloud and smoke, providing the fire progression maps that ML spread prediction models need as input. Current fire progression datasets rely on optical/thermal sensors that lose track of fires during cloud/smoke episodes. SAR provides cloud-independent observation.

**How it addresses the problem.** BCWildfire and other spread prediction models require spatiotemporally resolved fire progression maps for training. These maps have gaps during cloudy periods, which are common during active fire events (pyro-convective clouds, persistent smoke). SAR-derived perimeters would fill these gaps and improve both the training data and the real-time input for prediction models.

**Sketched experimental design.**
- *Data*: Sentinel-1 GRD time series over large (>1,000 ha) boreal fires. NBAC daily fire perimeters (infrared-derived) as reference. ERA5 wind and humidity for fire behavior context.
- *Method*: compute pre-fire vs. during-fire SAR backscatter change (VV and VH). Train a segmentation model (UNet on SAR temporal change images) to delineate active fire perimeter from backscatter depression (vegetation loss) and increase (soil moisture change). Evaluate against optical-derived perimeters on cloud-free days; test whether SAR-derived perimeters fill cloud-gap days.
- *Validation*: NBAC daily perimeters, MODIS/VIIRS active fire detections.
- *Study area*: Canada (2023 fire season — abundant large fires with well-documented perimeters).

**Source / gap.** Near-real-time SAR fire monitoring has been demonstrated for individual events (Scientific Reports, 2019) but not systematically for boreal fire progression mapping. BCWildfire uses optical-derived data only. No SAR-based fire progression dataset exists.

**Data modalities.** SAR (Sentinel-1), fire perimeters (NBAC), active fire (VIIRS/MODIS), weather (ERA5).

**Institutional fit.** University of Waterloo (Xu — BCWildfire team). VU Amsterdam (Veraverbeke — fire tracking). Canadian Forest Service. CIFFC.

---

### BF-02-ii: Pan-boreal next-day fire risk prediction with climate-terrain graph network

**Concept.** Extend BCWildfire's next-day fire risk prediction from British Columbia to the pan-boreal zone using a graph neural network (GNN) that represents the landscape as a network of interacting grid cells. Edges encode terrain connectivity (slope, fuel continuity), fire weather propagation (wind direction), and hydrological barriers (rivers, lakes). The GNN architecture naturally handles the irregular boundaries, varying spatial resolution, and geographic heterogeneity that a pan-boreal model must accommodate.

**How it addresses the problem.** BCWildfire's benchmark is BC-only. The boreal zone spans multiple fire regimes (Canadian boreal plains, Shield, Alaska Interior, Fennoscandia) with different fuels, terrain, and fire behavior. A CNN trained on BC may not transfer. A GNN with explicit landscape connectivity can encode these structural differences in its graph topology.

**Sketched experimental design.**
- *Data*: BCWildfire (BC) extended with NBAC fire perimeters and FWI grids for all of Canada. EFFIS for Fennoscandia. ERA5 wind fields. Fuel type maps (EOSD, Corine Land Cover). DEM (ArcticDEM, CDEM). Lake/river network (HydroLAKES, HydroRIVERS).
- *Model*: grid cells as nodes, spatial and terrain connectivity as edges. Node features: FWI, fuel type, NDVI, soil moisture (SMAP). Edge features: wind direction alignment, slope gradient, fuel continuity. GNN (GraphSAGE or GAT) predicting next-day fire probability per node. Compare against FireCastNet (3D CNN + GNN) and BCWildfire baselines (CNN, Transformer, Mamba).
- *Validation*: leave-one-year-out cross-validation. Test spatial transferability: train on Canada, test on Fennoscandia.
- *Study area*: pan-boreal (Canada, Alaska, Fennoscandia). Russia excluded due to data availability.

**Source / gap.** BCWildfire (2025) provides the BC benchmark. FireCastNet (2025) uses GNNs for seasonal prediction. No GNN-based next-day fire risk model exists for the pan-boreal zone.

**Data modalities.** Climate reanalysis (ERA5), optical satellite (NDVI from MODIS/Sentinel-2), soil moisture (SMAP), DEM, fire perimeters (NBAC, EFFIS), fuel type maps.

**Institutional fit.** University of Waterloo (Xu — BCWildfire). Imperial College London (Cheng — spatiotemporal DL). Canadian Forest Service (operational fire management). Thompson Rivers University (Flannigan — fire weather expertise).

---

## BF-04: Spectral vs. ecological recovery

### BF-04-i: Post-fire compositional recovery mapping from spectral unmixing time series

**Concept.** Apply regression-based spectral unmixing to the full Landsat archive (1985-2025) to map post-fire vegetation composition recovery (conifer fraction, deciduous fraction, exposed soil, water) across all fire perimeters in the boreal zone. This separates compositional recovery (return to pre-fire species mix) from spectral recovery (return to pre-fire greenness), directly addressing the documented mismatch.

**How it addresses the problem.** Zheng et al. (2024) demonstrated the approach for the 1987 Siberian megafire and showed spectral recovery outpaces compositional recovery, with broadleaved types recovering faster than needleleaf. The needleleaf index (2025) provides an improved spectral tool for conifer detection. Scaling this to all boreal fire perimeters would produce the first pan-boreal assessment of whether burned forests are recovering to their pre-fire composition or converting to different vegetation types.

**Sketched experimental design.**
- *Data*: Landsat Collection 2 Level-2 surface reflectance, 1985-2025. Fire perimeters (NBAC Canada, Alaska AICC, EFFIS Europe). Pre-fire and post-fire endmember spectra from forest inventory plots.
- *Method*: per-pixel regression-based spectral unmixing (following Zheng et al.) to estimate annual fractions of: needleleaf, broadleaf, shrub/grass, exposed soil, water. Apply the needleleaf index as a complementary metric. Compute compositional recovery time (years to reach 80% of pre-fire conifer fraction) and compare against spectral recovery time (years to reach 80% of pre-fire NDVI).
- *Validation*: ABoVE post-fire regeneration dataset (1,538 sites with field-measured species composition). FIA/NFI plots with post-fire remeasurements.
- *Study area*: pan-boreal, with detailed validation in NWT (high conversion rates) and northeastern China (Zheng et al. megafire site).

**Source / gap.** Zheng et al. (2024) applied this to one megafire. The needleleaf index (2025) provides the spectral tool. ABoVE provides the validation data. Pan-boreal application combining these is the gap.

**Data modalities.** Optical satellite (Landsat), fire perimeters (NBAC, AICC, EFFIS), field survey (ABoVE, FIA/NFI).

**Institutional fit.** Northern Arizona University (GEODE Lab — Goetz, Berner). University of Maryland (GLAD). Woodwell Climate (Rogers, Potter — ABoVE data). Chinese Academy of Sciences (Zheng).

---

### BF-04-ii: Recovery stage classification from multi-sensor time series (optical + SAR + LiDAR)

**Concept.** Classify post-fire forest into recovery stages — bare/charred, herbaceous, shrub, early forest, closed canopy — using a multi-sensor time series approach that combines optical spectral recovery (NDVI, NBR), SAR structural recovery (Sentinel-1 backscatter as a proxy for biomass accumulation), and sparse LiDAR height measurements (ICESat-2). Recovery stage classification captures structural and compositional information that single-sensor spectral indices miss.

**How it addresses the problem.** The core problem is that spectral recovery (greenness return) does not indicate ecological recovery (structural and compositional return to a functional forest). Adding SAR (sensitive to structural changes: stem density, biomass) and LiDAR (direct height measurement) provides orthogonal information to the spectral signal. A pixel classified as "spectrally recovered but structurally immature" is flagged as a false recovery.

**Sketched experimental design.**
- *Data*: Sentinel-2 time series (2017-2025), Sentinel-1 GRD time series, ICESat-2 ATL08 canopy height transects. Fire perimeters with year of burn.
- *Method*: define 5 recovery stages from field-based successional models. Train a temporal classifier (1D CNN or LSTM operating on multi-sensor time series per pixel) to predict recovery stage from the concatenated spectral + SAR + height trajectory. Compare against optical-only classification.
- *Validation*: field-assessed recovery stage at ABoVE sites. ALS-derived canopy metrics where available.
- *Study area*: fires from 2000-2015 in Canada (20+ years of recovery, spanning multiple stages) with ABoVE site overlap.

**Source / gap.** Single-sensor recovery assessment well-published. Multi-sensor recovery stage classification with SAR + LiDAR + optical is not published for boreal post-fire landscapes.

**Data modalities.** Optical satellite (Sentinel-2), SAR (Sentinel-1), spaceborne LiDAR (ICESat-2), field survey (ABoVE), airborne LiDAR (ALS).

**Institutional fit.** Woodwell Climate (ABoVE data, Potter, Rogers). Northern Arizona University (GEODE Lab). SLU (multi-sensor forest mapping). Canadian Forest Service.

---

### BF-04-iii: Linking spectral recovery to carbon recovery using eddy covariance + satellite fusion

**Concept.** Test whether spectral recovery indices (NDVI, EVI, NBR) predict carbon flux recovery (net ecosystem exchange, NEE) at post-fire eddy covariance tower sites. If spectral greenup correlates poorly with carbon uptake recovery, it confirms that spectral-based recovery assessments are misleading for carbon accounting. If a specific index or index combination does predict NEE recovery, it can be scaled spatially via satellite.

**How it addresses the problem.** The spectral vs. ecological recovery mismatch has implications for carbon accounting. National carbon inventories often assume spectral recovery = carbon recovery. If this assumption is wrong, post-fire carbon sink estimates are systematically biased. Testing the spectral-carbon link at flux tower sites provides the definitive answer.

**Sketched experimental design.**
- *Data*: ICOS and FLUXNET eddy covariance towers at post-fire boreal sites (Hyytiala post-prescribed burn, SOBS Saskatchewan, Canadian post-fire sites). Sentinel-2 and Landsat time series over tower footprints.
- *Method*: extract annual spectral recovery curves (NDVI, EVI, NBR, SIF) at tower footprint scale (~500 m radius). Correlate with annual NEE, GPP, and ecosystem respiration from flux partitioning. Compute the lag between spectral recovery and carbon recovery.
- *Evaluation*: report R² between spectral indices and NEE at annual and seasonal timescales. Identify which index best predicts carbon recovery timing.
- *Study area*: all ICOS/FLUXNET towers within or near post-fire boreal sites (estimated 8-12 towers).

**Source / gap.** Pierrat et al. (2022) showed SIF + vegetation indices predict GPP at boreal towers. The specific question — does spectral recovery predict NEE recovery after fire — has not been systematically tested across multiple tower sites.

**Data modalities.** Eddy covariance (ICOS, FLUXNET), optical satellite (Sentinel-2, Landsat), SIF (TROPOMI).

**Institutional fit.** University of Helsinki (INAR — Vesala, Mammarella, Hyytiala SMEAR II). University of Saskatchewan (SOBS tower). ICOS ERIC (data access). Woodwell Climate.

---

## BF-15: SIF drought stress monitoring

### BF-15-i: Boreal-specific SIF-GPP calibration network across ICOS and FLUXNET towers

**Concept.** Establish a boreal-specific SIF-GPP calibration relationship using TROPOMI SIF data matched to eddy covariance GPP measurements at 15-20 boreal flux tower sites. Current SIF-GPP relationships are derived globally or for temperate/tropical biomes. The 2025 Communications Earth & Environment study showed that SIF responds to temperature/light in boreal forests, not soil moisture — meaning tropical/temperate SIF drought frameworks don't transfer.

**How it addresses the problem.** SIF-based drought monitoring requires knowing the SIF-GPP relationship to detect anomalies. In boreal forests, this relationship differs from other biomes (light and temperature limitation dominate, not water). Without a boreal-specific calibration, applying global SIF-GPP relationships would produce biased drought stress estimates.

**Sketched experimental design.**
- *Data*: TROPOMI Level-2 SIF (740 nm) daily data, 2018-2025. Eddy covariance GPP from ICOS (Hyytiala, Sodankylä, Norunda, Zotino, and additional boreal sites) and FLUXNET (Canadian boreal sites — SOBS, Old Aspen, BERMS sites).
- *Method*: extract TROPOMI SIF at tower footprint scale (3.5 × 7 km). Compute daily, weekly, and seasonal SIF-GPP regressions per site. Test whether a single boreal SIF-GPP model generalizes across sites or whether site-specific calibration is required. Decompose the SIF-GPP relationship into PAR-driven and stress-driven components.
- *Evaluation*: cross-site R², RMSE, and bias. Compare against the global SIF-GPP regression of Pierrat et al. (2022) applied to boreal sites.
- *Study area*: circumpolar boreal (Finland, Sweden, Russia if ZOTTO data accessible, Canada).

**Source / gap.** Pierrat (2022) calibrated at SOBS only. Cheng showed land-cover-dependent slopes. The 2025 study showed SIF-stomatal decoupling in drought. No multi-site boreal calibration exists.

**Data modalities.** SIF (TROPOMI), eddy covariance (ICOS, FLUXNET), land cover (Sentinel-2).

**Institutional fit.** University of Helsinki (INAR — Vesala, Mammarella, Hyytiala). University of Saskatchewan (SOBS). NASA JPL / Caltech (Pierrat, Cheng — SIF retrieval expertise). ICOS ERIC.

---

### BF-15-ii: SIF anomaly-based boreal drought early warning system

**Concept.** Build an operational drought early warning system for boreal forests based on TROPOMI SIF anomalies (departure from climatological SIF for each pixel and week). When SIF drops below a threshold (e.g., 2σ below the 2018-2024 mean), flag the pixel as drought-stressed. Validate against observed drought impacts (growth reduction from tree rings, mortality events, reduced eddy covariance GPP).

**How it addresses the problem.** Boreal drought monitoring currently relies on precipitation deficit indices (PDSI, SPEI) that are computed from weather station networks — sparse in the boreal zone, especially in Russia and northern Canada. Satellite SIF provides wall-to-wall coverage and measures photosynthetic activity directly, rather than inferring drought from meteorological proxies.

**Sketched experimental design.**
- *Data*: TROPOMI SIF, 2018-2025 (7 years for climatology). ERA5-Land soil moisture and temperature for comparison. MODIS NDVI anomalies as baseline comparison. Tree ring data from the International Tree-Ring Data Bank for decadal drought validation.
- *Method*: compute pixel-level weekly SIF climatology (2018-2024). For each new week, compute SIF z-score. Flag pixels with z < -2 as "stressed." Compare SIF anomaly timing against NDVI anomaly timing (does SIF detect stress earlier than greenness loss?). Validate against tower-measured GPP drops during known drought years (2018 European drought, 2021 western Canadian drought).
- *Evaluation*: lead time advantage of SIF vs. NDVI for drought detection. Correlation with tree-ring growth anomalies. False positive rate in non-drought years.
- *Study area*: pan-boreal, with detailed validation at flux tower sites.

**Source / gap.** Behera et al. (2025, GRL) showed SIF yield as drought early warning. The 2025 study showed SIF-stomatal decoupling. No operational SIF-based drought early warning system exists for any biome, let alone boreal.

**Data modalities.** SIF (TROPOMI), climate reanalysis (ERA5-Land), optical satellite (MODIS NDVI), eddy covariance (ICOS, FLUXNET), tree ring (ITRDB).

**Institutional fit.** University of Helsinki (INAR — Hyytiala validation). NASA JPL (SIF retrieval, ECOSTRESS thermal drought). Luke (Finnish forest health monitoring). NRCan (Canadian Forest Service — forest health).

---

### BF-15-iii: SIF + land surface temperature fusion for disentangling boreal drought mechanisms

**Concept.** Fuse TROPOMI SIF with land surface temperature (LST) from Landsat TIRS or ECOSTRESS to disentangle the two boreal drought pathways: (1) heat stress (high LST, reduced SIF from photoprotective downregulation) and (2) soil moisture deficit (low LST or normal LST, reduced SIF from stomatal closure). The 2025 study showed that current SIF products fail to capture stomatal responses — adding thermal data may resolve this.

**How it addresses the problem.** Boreal drought is not a single phenomenon. The 2018 European drought was heat-dominated (Fennoscandia); the 2021 Canadian drought was a precipitation deficit. SIF alone responds to both but can't distinguish them. The LST-SIF phase relationship differs: heat stress shows simultaneous LST rise and SIF drop; moisture stress shows SIF drop preceding or decoupled from LST. Distinguishing these pathways matters because heat-stressed forests recover quickly while moisture-stressed forests may experience dieback.

**Sketched experimental design.**
- *Data*: TROPOMI SIF. ECOSTRESS LST (70 m, ISS platform, irregular revisit) or Landsat TIRS (100 m, 16-day). ERA5-Land soil moisture and temperature. NDVI from Sentinel-2.
- *Method*: compute SIF-LST scatter plots at pixel and weekly scale. Classify drought events into heat-dominated (quadrant: high LST, low SIF) vs. moisture-dominated (quadrant: normal/low LST, low SIF). Validate classification against tower-measured energy balance partitioning (latent heat flux / sensible heat flux ratio as moisture indicator).
- *Evaluation*: ability to correctly classify known drought events (2018 heat, 2021 moisture) by mechanism. Test whether mechanism classification predicts recovery trajectory (heat-stressed forests recover faster).
- *Study area*: Fennoscandia (2018 heat drought with excellent tower data) and central Canada (2021 moisture drought).

**Source / gap.** SIF-LST synergy for drought detection proposed (Bian et al., 2024, Science of the Total Environment) for Mediterranean/tropical. Not applied to boreal. The boreal-specific SIF-stomatal decoupling (2025) makes the thermal complement more important here than in other biomes.

**Data modalities.** SIF (TROPOMI), thermal satellite (ECOSTRESS, Landsat TIRS), eddy covariance (ICOS, FLUXNET), climate reanalysis (ERA5-Land).

**Institutional fit.** NASA JPL (Pierrat — SIF, ECOSTRESS). University of Helsinki (INAR — Hyytiala, tower data). Lund University (eddy covariance, 2018 drought analysis). Luke (forest health monitoring).

---

## Summary table

| ID | Problem | Idea | Core method |
|---|---|---|---|
| BF-03-i | Reburns | Time-since-fire reburn probability surfaces | LightGBM/RF on Landsat recovery trajectories |
| BF-03-ii | Reburns | Overwintering fire detection from pre-greenup imagery | Sentinel-2 change detection in spring snow window |
| BF-03-iii | Reburns | Conifer-to-deciduous conversion mapping | Landsat spectral unmixing + needleleaf index |
| BF-09-i | Treeline | 10 m ecotone structure from ICESat-2 + Sentinel-2 | Deep learning height/cover regression |
| BF-09-ii | Treeline | 40-year treeline advance rate estimation | BFAST trend analysis on Landsat NDVI |
| BF-09-iii | Treeline | Krummholz detection from VHR imagery | Object detection on sub-meter satellite/drone |
| BF-01-i | Small fires | SAR-optical fusion for sub-100 ha detection | Dual-encoder Sentinel-1 + Sentinel-2 |
| BF-01-ii | Small fires | Sentinel-3 SLSTR night-time thermal detection | Contextual fire detection at high latitudes |
| BF-01-iii | Small fires | Prescribed burn verification in Fennoscandia | Sentinel-2 spectral trajectory classification |
| BF-06-i | Permafrost | NISAR L-band InSAR in forested permafrost | SBAS time series, L-band vs. C-band coherence |
| BF-06-ii | Permafrost | Multi-sensor permafrost vulnerability mapping | InSAR + optical + thermal fusion index |
| BF-06-iii | Permafrost | Fire-permafrost interaction from InSAR | Post-fire subsidence rate comparison |
| BF-18-i | Storm damage | NISAR L-band windthrow severity mapping | L-band backscatter change classification |
| BF-18-ii | Storm damage | Near-real-time windthrow alerting pipeline | Sentinel-1 z-score anomaly detection |
| BF-19-i | Bioacoustics | Boreal BirdNET fine-tuning | Transfer learning on circumpolar audio data |
| BF-19-ii | Bioacoustics | Foundation model multi-taxon classifier | Audio foundation model fine-tuning |
| BF-19-iii | Bioacoustics | Acoustic phenology as climate indicator | PAM event detection + phenology correlation |
| BF-16-i | IFL protection | Fragmentation risk forecasting | Spatiotemporal ML on fire/climate projections |
| BF-16-ii | IFL protection | Connectivity-optimized protection planning | Zonation 5 corridor retention analysis |
| BF-02-i | Fire spread | SAR-based fire perimeter tracking | Sentinel-1 backscatter segmentation |
| BF-02-ii | Fire spread | Pan-boreal next-day fire risk with GNN | Graph neural network on landscape graph |
| BF-04-i | Recovery | Compositional recovery from spectral unmixing | Landsat unmixing + needleleaf index |
| BF-04-ii | Recovery | Multi-sensor recovery stage classification | Optical + SAR + LiDAR temporal classifier |
| BF-04-iii | Recovery | Spectral-carbon recovery link at flux towers | SIF/NDVI vs. NEE at eddy covariance sites |
| BF-15-i | SIF drought | Boreal SIF-GPP calibration network | TROPOMI SIF vs. tower GPP regression |
| BF-15-ii | SIF drought | SIF anomaly drought early warning | SIF z-score flagging system |
| BF-15-iii | SIF drought | SIF + LST fusion for drought mechanism | SIF-LST scatter plot classification |
