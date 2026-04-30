# Stage 6: Top 10 research proposals

One-pager write-ups for the 10 highest-scoring research directions from Stage 5. Ordered by aggregate score.

---

## 1. BF-03-i: Time-since-fire reburn probability surfaces (7.7)

### Motivation

Boreal fire managers have no spatial product telling them which previously burned areas are re-entering the flammable window. Tepley et al. (2025) showed that fire-vegetation feedbacks operate predictably at biome scale: reburn resistance decays as fuels rebuild, and the rate depends on climate zone and post-fire vegetation trajectory. Whitman et al. (2024) demonstrated that even modest fire weather increases overcome this resistance in Canadian boreal forests. But this ecological knowledge has not been translated into a gridded decision-support product. Fire agencies allocate suppression resources reactively. A reburn probability surface — annually updated, 30 m resolution — would let them pre-position resources and prioritize fuel treatments in areas approaching the flammability threshold.

### Approach

**Data.** CanadaFireSat (Landsat burned-area labels, 1985-present). NBAC fire perimeters for fire history. ERA5-Land for climate covariates (growing degree days, FWI climatology, annual precipitation). ArcticDEM/CDEM for topographic wetness index. ABoVE post-fire regeneration dataset (1,538 sites) for fuel recovery calibration.

**Method.** Identify all 30 m pixels burned twice within the Landsat archive period. Extract spectral recovery trajectories (NBR, dNBR, NDVI) between fire events. Build a pixel-level model predicting reburn probability within 1-5 years, conditioned on time-since-fire, spectral recovery rate, climate zone, FWI climatology, and topographic wetness. Start with LightGBM or random forest for interpretability; compare against a spatiotemporal CNN operating on 5-year spectral recovery windows. Spatial cross-validation using leave-one-ecoregion-out.

**Study area.** Northwest Territories and Interior Alaska (highest documented reburn rates). Test transfer to Fennoscandia.

**Timeline.** 12-18 months. Data assembly: 2 months. Model development: 4-6 months. Validation and transfer testing: 3-4 months. Paper writing: 2-3 months.

### Expected outcome

**Best case.** A pan-boreal reburn probability product at 30 m, updated annually, with AUC > 0.80 and well-calibrated probability estimates. Adopted as a decision-support layer by CIFFC or provincial fire agencies.

**Realistic case.** A regionally validated product (NWT + Alaska) with AUC 0.70-0.80 and reasonable calibration. Spatial transferability to Fennoscandia is limited and requires local recalibration. Published as a methods paper with the product released as a data paper.

**Null-result scenario.** Reburn probability is too dependent on stochastic fire weather to predict spatially — the model has low discrimination (AUC < 0.65) because fire weather dominates over fuel state. This would itself be an informative finding (reburn is weather-driven, not fuel-driven) but would limit the product's operational utility.

### Fieldwork requirements

Minimal for the core product. ABoVE field data (1,538 sites with fuel recovery measurements) provides existing ground truth. A targeted field campaign (2-4 weeks in summer, NWT or Interior Alaska) to measure fuel loads at sites spanning the reburn probability gradient would strengthen validation but is not required for the first paper.

### Risks and dependencies

- **Data risk (low).** CanadaFireSat and NBAC are open and well-documented. ERA5-Land is freely available.
- **Methodological risk (medium).** The key assumption — that fuel recovery trajectory is a useful predictor beyond fire weather — is supported by Tepley and Whitman but not yet tested in a predictive modeling framework. If fire weather dominates, the model will have skill only at very short time horizons (next year, not next 5 years).
- **Domain knowledge gap (low-medium).** Fuel recovery ecology is well-documented by Tepley, Whitman, and Walker/Mack. The computational challenge is straightforward. The ecological interpretation of model features requires domain collaboration but is not the binding constraint.
- **Partnership dependency (low).** Can be done independently with open data. Collaboration with Whitman (NRCan) or Tepley (USDA) would improve ecological interpretation and operational credibility.

### Venue fit

**Journals.** Remote Sensing of Environment (methods + product), Global Change Biology (ecological interpretation), or Fire (applied fire science). A companion data paper in Earth System Science Data for the product itself.

**Conferences.** AGU (Natural Hazards or Biogeosciences sections), IGARSS, Canadian Wildland Fire & Smoke Conference.

**Funding.** NSERC Discovery (Canada), NASA Terrestrial Ecology, ESA Living Planet Fellowship.

**Institutions.** Canadian Forest Service / NRCan (Whitman — Northern Forestry Centre). Woodwell Climate (Rogers, Potter). NAU GEODE Lab (Goetz, Walker). CIFFC (operational deployment).

### Honest assessment

**Strengths.** The data pipeline is clean. The ecological basis is established. The computational approach is well within a strong ML/remote sensing skill set. The product has an obvious end user (fire agencies). This is the most likely of the 10 ideas to produce a useful, publishable result within 18 months.

**Weaknesses.** The idea is a product, not a scientific discovery. It translates existing ecological knowledge into a spatial layer. Novelty is in the engineering, not the science. A reviewer could argue this is an application paper rather than a contribution to fire ecology. To counter this, the model's feature importances need to tell us something new about what drives reburn probability spatially — if it just confirms what Tepley already showed, the contribution is incremental.

**Where ecological expertise matters.** Defining meaningful reburn probability thresholds (at what probability should a fire manager act?), interpreting model failures (why did the model miss a reburn event — was it fuel, weather, or ignition source?), and ensuring the product communicates uncertainty appropriately to non-technical users.

---

## 2. BF-09-ii: Treeline advance rate from 40-year Landsat (7.3)

### Motivation

Feng et al. (2026) confirmed net northward tree cover shift across the boreal zone using 224,026 Landsat images, and Berner et al. (2022) documented biome shift from satellite vegetation indices. But neither study produced spatially explicit advance rates — meters per decade — at the granularity that ecological forecasting models need. Knowing where treeline is advancing at 50 m/decade vs. 5 m/decade, and where it has stalled or reversed, is the input that species range projections and carbon models require. The 40-year Landsat archive (1984-2025) is long enough to measure trends above interannual noise.

### Approach

**Data.** Landsat Collection 2 Level-2 surface reflectance, 1984-2025. ERA5-Land for climate covariates (growing degree days, snow-free season length, soil temperature). MODIS land surface phenology. ArcticDEM for slope and aspect. ESA CCI Permafrost for permafrost probability. Feng et al. (2026) tree cover maps for ecotone zone delineation.

**Method.** For each 30 m pixel in the ecotone zone (defined as Feng et al. tree cover 5-60%), compute annual peak-season NDVI from Landsat. Apply BFAST (Breaks For Additive Seasonal and Trend) for breakpoint detection. Extract trend slopes for segments between breakpoints. Classify pixels into advancing (positive trend crossing a tree-cover threshold), stable, or retreating. Attribute advance rates to growing degree day trends, snow-free season length, and permafrost probability using partial correlation and random forest importance. Stratify results by continentality (maritime Fennoscandia vs. continental Siberia vs. continental Canada).

**Study area.** Circumpolar, with three detailed transects: northern Quebec (steep elevation gradient), NWT/Yukon (moderate gradient), Finnish Lapland (gentle latitudinal gradient).

**Timeline.** 12-15 months. Landsat data processing: 3 months (GEE or HPC). BFAST analysis: 2-3 months. Climate attribution: 2-3 months. Paper: 2-3 months.

### Expected outcome

**Best case.** A circumpolar treeline advance rate map at 30 m resolution, showing advance rates of 10-100 m/decade varying by region, with strong attribution to specific climate drivers. Published in a high-impact journal (Nature Climate Change, Global Change Biology) because it quantifies a climate indicator at unprecedented spatial resolution.

**Realistic case.** Advance rates are measurable in some regions (northern Quebec, Fennoscandia) but BFAST produces noisy results in others (cloudy regions, short growing seasons in Siberia). The circumpolar product has coverage gaps. Climate attribution is suggestive but not definitive because multiple drivers co-vary. Published in Remote Sensing of Environment or Global Change Biology as a methods + results paper.

**Null-result scenario.** Interannual NDVI variability in the ecotone is too high relative to the trend signal for BFAST to detect meaningful breakpoints. The advance rate is below the detection threshold of 30 m pixels over 40 years (< ~1 m/decade). This would indicate that treeline advance is too slow for Landsat-based detection and would redirect effort toward higher-resolution sensors.

### Fieldwork requirements

None for the core analysis. Validation against dendrochronological records (tree ring dating of treeline establishment) and historical photo-repeat stations would strengthen the paper. These records exist in the literature and from collaborators (Danby at Queen's, Macias-Fauria at Oxford). A 1-2 week field visit to one ecotone transect to ground-truth classification (advancing vs. stable vs. retreating) would help but is not required.

### Risks and dependencies

- **Data risk (low).** Landsat and ERA5 are freely available. GEE provides processing infrastructure.
- **Methodological risk (medium).** BFAST was designed for tropical deforestation detection where the signal is abrupt. Treeline advance is gradual. The method may lack sensitivity for slow trends. May need to supplement with linear trend analysis or change-point detection tuned for slow signals.
- **Ecological risk (low-medium).** Advance rate depends on the definition of "treeline" — different definitions (canopy cover threshold, height threshold, species presence) give different rates. Need ecological guidance on which definition is most meaningful.
- **Compute (low-medium).** Circumpolar Landsat processing is large but feasible on GEE or institutional HPC.

### Venue fit

**Journals.** Global Change Biology (ecological significance), Remote Sensing of Environment (methods), Nature Climate Change (if the pan-boreal result is strong enough).

**Conferences.** AGU (Biogeosciences), EGU, Arctic Science Summit Week.

**Funding.** ESA Living Planet Fellowship, ERC Starting Grant (if framed as part of a larger treeline dynamics program), Academy of Finland.

**Institutions.** University of Maryland GLAD (Feng, Sexton — tree cover products). NAU GEODE Lab (Berner, Goetz — satellite vegetation time series). SLU (boreal remote sensing). Queen's University (Danby — treeline ecology).

### Honest assessment

**Strengths.** The data and methods are mature. The analysis can be done without partnerships (open data, established tools). The result — spatially explicit advance rates — fills a specific gap that multiple research groups have identified. The 40-year record is uniquely long.

**Weaknesses.** This is analysis of existing data, not a new method. Novelty is in the scale and specificity of the result, not the technique. BFAST may not be the right tool for slow-trend detection and could require modification. The circumpolar scope is ambitious for a single study; a regional pilot might be more realistic as a first paper.

**Where ecological expertise matters.** Defining what "treeline advance" means ecologically (not just spectrally), interpreting why advance stalls at specific locations (permafrost, fire, herbivory, seed dispersal limitation), and placing the rates in the context of climate velocity (is treeline keeping up with its climate envelope?).

---

## 3. BF-09-i: 10 m treeline ecotone structure from ICESat-2 + Sentinel-2 (7.2)

### Motivation

Current tree cover products map the forest-tundra transition as a binary boundary: tree or no tree. The ecotone is a gradient — from closed canopy through open woodland to krummholz to tundra — and the gradient's structure (height, cover fraction, stem density) determines its ecological function. A 10 m product capturing this gradient would show not just where treeline is, but what it looks like: tall dense forest vs. sparse stunted krummholz. This structural information indicates whether a treeline is actively advancing (tall, dense edge), stable (sparse, wind-shaped), or retreating (dead standing stems).

### Approach

**Data.** ICESat-2 ATL08 canopy height profiles (sparse along-track measurements, ~11 m footprint). Sentinel-2 Level-2A surface reflectance (10 bands, 10-20 m, wall-to-wall). ArcticDEM (2 m, pan-Arctic). ALS strips from national surveys (Finland, Sweden, Canada) for independent validation.

**Method.** Train an encoder-decoder network (UNet-style) to predict canopy height and cover fraction at 10 m from Sentinel-2 imagery, using ICESat-2 transects as sparse supervision targets. The loss function combines regression on height/cover where ICESat-2 data exists with spatial consistency regularization elsewhere. Compare against a random forest baseline and the 2025 deep learning fusion approach (MAE 1.42 m). Produce annual maps for 2017-2025 to track ecotone evolution.

**Study area.** Three transects: northern Quebec, NWT/Yukon, Finnish Lapland.

**Timeline.** 15-18 months. Data preparation and ICESat-2/Sentinel-2 co-registration: 3 months. Model development: 4-5 months. Validation: 2-3 months. Temporal analysis: 2-3 months. Paper: 2-3 months.

### Expected outcome

**Best case.** A 10 m ecotone product with height RMSE < 2 m and cover fraction RMSE < 15%, showing annual structural changes across the ecotone. Detects advance/retreat signals that binary tree cover products miss.

**Realistic case.** Height prediction achieves RMSE 2-3 m (comparable to or slightly worse than the 2025 benchmark). Cover fraction estimation is noisier. Annual changes are at the noise floor for most of the ecotone, but multi-year trends (2017-2025) are detectable at some sites. The product is useful as a baseline map but temporal change detection is marginal.

**Null-result scenario.** The sparse ICESat-2 supervision is insufficient for the model to generalize to Sentinel-2-only pixels in the ecotone. The ecotone's spectral diversity (lichen, shrub, exposed rock, snow patches) confounds the height prediction. The model produces accurate heights along ICESat-2 tracks but degrades rapidly away from them.

### Fieldwork requirements

Low. ALS validation data exists from national surveys (SLU, Luke, NRCan). A 1-2 week field campaign at one ecotone transect (Finnish Lapland is most accessible) to measure canopy height and cover fraction at reference plots would validate both the ICESat-2 training targets and the model predictions. Not required for the first paper if ALS data is available.

### Risks and dependencies

- **ICESat-2 coverage.** Sparse across-track. The ecotone may have insufficient ICESat-2 samples for training in some regions. Mitigation: aggregate across years (2018-2025) to increase sample density.
- **Sentinel-2 cloud cover.** High latitudes have persistent cloud cover. May need to composite across multiple dates per growing season, which introduces phenological variation.
- **ALS access.** National survey ALS data is open in Finland and Sweden but restricted in parts of Canada. May need data-sharing agreements.

### Venue fit

**Journals.** Remote Sensing of Environment, IEEE Transactions on Geoscience and Remote Sensing.

**Institutions.** SLU (Remote Sensing Division). NASA Goddard (ICESat-2 mission science). NAU GEODE Lab. University of Helsinki.

### Honest assessment

**Strengths.** The multi-sensor fusion approach is well-suited to a strong ML profile. The product fills a specific gap. ICESat-2 and Sentinel-2 are both freely available.

**Weaknesses.** The approach is technically sound but may produce a product that is only marginally better than existing tree cover maps at the ecotone. The ecological value depends on whether the height/cover gradient contains information that binary tree cover doesn't — and that's an ecological question, not a computational one. If the ecotone transitions over 100+ km, a 10 m product may be overresolved for the ecological question. Needs an ecologist collaborator to define what structural resolution is actually useful.

---

## 4. BF-01-i: SAR-optical fusion for sub-100 ha fire detection (7.2)

### Motivation

Hall et al. (2021) showed that MODIS missed nearly half of boreal burned area, with detection rates below 10% for fires under 100 ha. Potter et al. (2026) achieved F1=0.85 with UNet++ on Landsat/Sentinel-2, but optical detection fails during cloud and smoke episodes that can last days during active fire events. Cloud cover during boreal fire season is common — during the 2024 Jasper fire, MODIS/VIIRS had multi-day detection gaps. Sentinel-1 C-band SAR operates independent of cloud and smoke. Fusing SAR and optical detection would close the cloud-gap problem and improve detection of small fires.

### Approach

**Data.** Sentinel-1 GRD (VV/VH, 10 m, 6-day revisit in Europe, 12-day in Canada). Sentinel-2 Level-2A (dNBR, 10 m). NBAC fire perimeters as labels, filtered to fires < 100 ha. ERA5 for fire weather context.

**Method.** Dual-encoder architecture: one branch processes SAR temporal change (pre-fire vs. post-fire VV, VH, VV/VH ratio), one branch processes optical spectral indices (dNBR, dNDVI, SWIR). Late fusion via attention layer. Handle missing optical data (cloud-covered dates) with attention masking — the model learns to rely more on SAR when optical is missing and vice versa. Train on 2017-2023 fire seasons; validate on 2024.

**Study area.** Canada (NBAC-labeled) and Sweden (SMHI system for comparison).

**Timeline.** 12-15 months. Data assembly: 2-3 months. Model development: 4-5 months. Evaluation: 2-3 months. Paper: 2-3 months.

### Expected outcome

**Best case.** F1 > 0.85 for all fire sizes, including < 100 ha. The SAR branch detects fires during 60%+ of cloud-gap periods, meaning the fusion system has no multi-day detection gaps. The system matches or exceeds SMHI's operational VIIRS system.

**Realistic case.** F1 of 0.75-0.85 for fires < 100 ha (below Potter's performance for all-size fires, because small fires are harder). SAR detects fires during cloud gaps at 40-60% recall — better than optical-only but not perfect. Published as a methods contribution showing the value of SAR for cloud-gap filling.

**Null-result scenario.** SAR backscatter change from small fires is too weak to distinguish from background variation (soil moisture changes, phenology, wind roughening). The fusion model defaults to optical-only performance and SAR adds nothing. This is informative: it would establish the lower bound of fire size detectable by Sentinel-1 C-band.

### Fieldwork requirements

None. The work is entirely satellite-based with open label data (NBAC).

### Risks and dependencies

- **SAR sensitivity (medium).** C-band SAR may not detect small, low-intensity surface fires where canopy is intact (fire burns understory but not overstory). L-band (NISAR) would be better but data is only beginning to accumulate.
- **Label quality (low-medium).** NBAC perimeters for small fires may be inaccurate — small fires are often mapped from a single satellite overpass with coarse boundaries.
- **Compute (low-medium).** Pan-boreal SAR + optical processing is large but feasible with cloud computing or HPC.

### Venue fit

**Journals.** Remote Sensing of Environment, IEEE TGRS, International Journal of Applied Earth Observation and Geoinformation.

**Conferences.** IGARSS, AGU.

**Institutions.** VU Amsterdam (Veraverbeke, Scholten). SMHI (Hassellof). Woodwell Climate (Potter). Canadian Forest Service.

### Honest assessment

**Strengths.** The dual-encoder architecture with missing-modality handling is a well-understood ML pattern. The data is open. The problem (cloud gaps in fire detection) is well-documented and the solution (add SAR) is physically motivated. The paper writes itself: show detection curves stratified by fire size and cloud cover.

**Weaknesses.** The idea is engineering-heavy and science-light. The reviewer question is: does fusing SAR with optical tell us anything we didn't already know, or is it just better plumbing? To make this a scientific contribution, the paper needs to quantify how much burned area is missed during cloud-gap periods and what that means for carbon emission estimates. Without the carbon accounting angle, it's an applied remote sensing paper.

**Where ecological expertise matters.** Less than other ideas. The binding constraint is SAR processing skill, not ecological interpretation. An ecologist collaborator is useful for framing the carbon accounting implications but not essential for the core work.

---

## 5. BF-03-iii: Conifer-to-deciduous conversion mapping (7.0)

### Motivation

Post-fire vegetation type conversion — boreal conifer forest regenerating as deciduous — is one of the most consequential ecological changes underway in the boreal zone. Walker and Mack documented it at field scale. Zheng et al. (2024) demonstrated that spectral recovery outpaces compositional recovery after the 1987 Siberian megafire, with broadleaved types recovering faster than conifers. But nobody has mapped this conversion pan-boreally. The consequence is direct: conifer forests that regenerate as deciduous have lower flammability, altered albedo, different carbon stocks, and changed habitat value. Whether these conversions are permanent or temporary determines the boreal zone's long-term carbon trajectory.

### Approach

**Data.** Landsat Collection 2 surface reflectance, 1985-2025. NBAC/AICC/EFFIS fire perimeters. ABoVE post-fire regeneration dataset (1,538 sites with species composition). Needleleaf index (2025) for conifer fraction estimation.

**Method.** Apply regression-based spectral unmixing (following Zheng et al. 2024) to estimate annual fractions of needleleaf, broadleaf, shrub/grass, and exposed soil for every pixel inside fire perimeters. Combine with the needleleaf index as a complementary metric. Compute compositional recovery time (years to 80% of pre-fire conifer fraction) and compare against spectral recovery time (years to 80% of pre-fire NDVI). Map conversion zones where conifer fraction has not recovered after 15+ years.

**Study area.** Pan-boreal, with detailed validation in NWT (high conversion rates documented by Walker/Mack) and northeastern China (Zheng et al. megafire).

**Timeline.** 12-18 months.

### Expected outcome

**Best case.** A pan-boreal conifer-to-deciduous conversion map showing that X% of post-fire area burned 15+ years ago has not recovered to conifer. Spatial patterns reveal that conversion is concentrated in specific climate zones or fire severity classes, with clear attribution.

**Realistic case.** Conversion mapping works well in NWT and Interior Alaska (where field validation exists) but is noisier in Fennoscandia (smaller fires, more mixed forests) and Russia (no field validation). The pan-boreal product has regional quality variation. Published with strong results for North America and caveats for Eurasia.

**Null-result scenario.** The spectral separation between conifer and deciduous at Landsat resolution is insufficient for reliable unmixing in the boreal zone (where species mixtures are common at sub-pixel scales). The unmixing produces high-variance fraction estimates that don't match field data. This would redirect toward higher-resolution sensors (Sentinel-2 at 10 m) or toward classification rather than unmixing.

### Fieldwork requirements

Low. ABoVE field data provides the validation backbone. A 2-3 week field campaign in NWT to measure conifer/deciduous fraction at additional post-fire sites (spanning 5-30 years since fire) would strengthen validation and extend the ABoVE network's temporal coverage.

### Risks and dependencies

- **Spectral unmixing accuracy.** Sub-pixel fraction estimation from 30 m Landsat is inherently noisy. Conifer and deciduous have good spectral separation in peak season but can overlap in spring/fall.
- **Pre-fire baseline.** Requires knowing the pre-fire conifer fraction, which means the fire must have occurred during the Landsat archive period (post-1985). Older fires are excluded.
- **ABoVE data access.** The regeneration dataset is publicly available via ORNL DAAC.

### Venue fit

**Journals.** Global Change Biology, Remote Sensing of Environment, Ecosystems.

**Institutions.** NAU GEODE Lab (Walker, Mack — field data and ecological expertise). University of Maryland GLAD (Feng — tree cover products). Woodwell Climate (Rogers — fire-carbon research).

### Honest assessment

**Strengths.** Zheng et al. proved the method works. The needleleaf index provides an improved tool. ABoVE provides validation. The ecological question (is conversion permanent?) is important and unanswered at pan-boreal scale. The computation is within a standard remote sensing skill set.

**Weaknesses.** This is scaling an existing method, not inventing a new one. The ecological interpretation — why does conversion happen in some places and not others — requires fire ecology expertise. A computational lead without a fire ecologist collaborator risks producing a map that says "here" without explaining "why."

---

## 6. BF-16-ii: Connectivity-optimized IFL protection planning (7.0)

### Motivation

Intact Forest Landscapes (IFLs) are managed as isolated units — each assessed independently for loss. But their ecological value depends on connectivity. Caribou, wolverine, and other wide-ranging boreal species need connected habitat. When an IFL loses connectivity to adjacent IFLs or protected areas, its effective conservation value drops even if its internal area is maintained. Zonation 5 includes corridor retention and connectivity algorithms specifically designed for this kind of analysis, but no published study has applied them to boreal IFL connectivity.

### Approach

**Data.** IFL boundaries (GLAD, 2000/2013/2020). WDPA protected area boundaries. Caribou and wolverine habitat models (NatureServe, ECCC species-at-risk maps). Sentinel-2 land cover for resistance surface construction. Road density from OpenStreetMap. DEM for terrain ruggedness.

**Method.** Build a landscape resistance surface based on land cover, road density, and terrain. Run Zonation 5 with connectivity weighting (boundary length penalty, corridor retention) to identify the top 5% of unprotected boreal area that, if protected, would maximize connectivity between existing IFLs and protected areas. Compare against area-only prioritization (no connectivity weighting) and random selection. Evaluate using landscape connectivity metrics: effective mesh size, patch cohesion index, resistance-weighted distance between IFLs.

**Study area.** Ontario Clay Belt (fragmented IFL landscape, active logging, caribou habitat) and northern Finland/Sweden (IFL-protected area interface).

**Timeline.** 10-14 months. Data assembly and resistance surface: 2-3 months. Zonation analysis: 2-3 months. Scenario comparison: 2-3 months. Paper: 2-3 months.

### Expected outcome

**Best case.** Identifies specific unprotected corridors whose protection would increase landscape connectivity by 30-50% while adding < 5% to total protected area. Results are actionable for provincial/national protected area planning.

**Realistic case.** The analysis produces a prioritization map, but the specific corridors identified are already known to conservation practitioners (they're obvious landscape bottlenecks). The contribution is quantification, not discovery. Published as a methods paper demonstrating the Zonation 5 connectivity workflow for boreal IFL planning.

**Null-result scenario.** IFL connectivity in the study area is already too degraded for corridor-based solutions — the gaps are too wide for species dispersal. Or, conversely, connectivity is already adequate through existing protected areas and new corridors are unnecessary. Either result is informative for conservation planning.

### Fieldwork requirements

None. Fully GIS-based.

### Risks and dependencies

- **Species habitat models.** Caribou and wolverine habitat models may not be available at sufficient resolution for the study area. May need to use simpler resistance surfaces based on land cover alone.
- **Skill Fit concern.** This is more conservation biology and GIS than ML. Zonation is a specialized tool with a learning curve, but it's not deep learning. The computational component is lighter than other ideas in this list. This may be a good starting project for building ecological credibility, but it won't push computational skills.
- **Partnership strongly recommended.** Working with WCS Canada or Pew Boreal Campaign on the Ontario Clay Belt study area would provide caribou data, policy context, and credibility that a computational-only team would lack.

### Venue fit

**Journals.** Biological Conservation, Conservation Biology, Conservation Letters (if results are compact and policy-relevant).

**Institutions.** University of Helsinki (Moilanen — Zonation developer). WCS Canada (boreal conservation). Pew Charitable Trusts. TNC Canada.

### Honest assessment

**Strengths.** The tool (Zonation 5) exists and works. The data is open. The analysis is directly policy-relevant. This is the idea most likely to lead to a real conservation outcome (influence on protected area planning).

**Weaknesses.** The computational contribution is modest. Zonation is a tool, not a method you develop. A reviewer might ask: what is the scientific contribution beyond running existing software on a new dataset? The answer has to be in the ecological analysis — what the connectivity results reveal about IFL vulnerability that wasn't known. Without a conservation biology collaborator, this is hard to argue convincingly. Skill Fit scored 6 (lowest in the top 10) because this is spatial ecology work, not ML work. If the goal is to build ML/remote sensing publications, other ideas score higher on that dimension.

---

## 7. BF-04-i: Compositional recovery from spectral unmixing time series (7.0)

### Motivation

The documented mismatch between spectral recovery and ecological recovery after boreal fires is a known problem (Zheng et al. 2024, Fire Ecology 2024), but no one has quantified it at pan-boreal scale. National carbon inventories and forest management plans often assume that spectral greenup indicates forest recovery. If burned forests that appear green are actually deciduous shrubs or grassland rather than the pre-fire conifer forest, then post-fire carbon sink estimates are systematically wrong. The Landsat archive is long enough (1985-2025) to track 40 years of recovery and distinguish genuinely recovered forests from false-positive greenup.

### Approach

**Data.** Landsat Collection 2 Level-2 surface reflectance. Fire perimeters (NBAC, AICC, EFFIS). ABoVE post-fire regeneration dataset (1,538 sites). Needleleaf index (2025) for conifer fraction.

**Method.** Per-pixel spectral unmixing producing annual fractions of needleleaf, broadleaf, shrub/grass, exposed soil, and water. Compute compositional recovery time (years to 80% pre-fire conifer fraction) vs. spectral recovery time (years to 80% pre-fire NDVI). Map the gap between these two metrics as a "false recovery index" — high values mean the site looks green but has not returned to pre-fire composition.

**Study area.** Pan-boreal with ABoVE validation focus.

**Timeline.** 12-18 months.

### Expected outcome

**Best case.** Demonstrates that 20-40% of spectrally recovered post-fire area in Canada is not compositionally recovered, with clear spatial patterns (higher false recovery in drier, more fire-prone regions). Changes how recovery is assessed in national forest inventories.

**Realistic case.** The false recovery index reveals regional patterns but is noisy at individual pixel scale. Published as a demonstration of the spectral-compositional gap with regional estimates and uncertainty bounds.

**Null-result scenario.** Unmixing accuracy is too low in mixed boreal forests to reliably distinguish needleleaf from broadleaf at 30 m. The false recovery index has error bars wider than the signal.

### Fieldwork requirements

Low. ABoVE provides 1,538 field-validated sites. Additional field visits (2-3 weeks in NWT) would extend the validation dataset, especially for fires in the 10-20 year post-fire window where conversion is most active.

### Risks and dependencies

- **Overlap with BF-03-iii.** These ideas share methods (spectral unmixing + needleleaf index) but differ in framing: BF-03-iii links conversion to reburn probability, BF-04-i links it to carbon accounting. They could be the same paper or two papers from the same data.
- **Unmixing limitations.** As discussed in BF-03-iii. Landsat resolution may be too coarse.

### Venue fit

**Journals.** Remote Sensing of Environment, Global Change Biology, Biogeosciences.

**Institutions.** NAU GEODE Lab (Goetz, Berner). Woodwell Climate (Rogers, Potter). Chinese Academy of Sciences (Zheng).

### Honest assessment

**Strengths.** Same as BF-03-iii — the method works, the data exists, the ecological question is important.

**Weaknesses.** This idea and BF-03-iii are close enough that pursuing both risks redundancy. The choice between them depends on framing: fire management (BF-03-iii) vs. carbon accounting (BF-04-i). The carbon accounting angle has higher potential impact (national inventories) but requires understanding of how national forest inventories actually use satellite data — domain knowledge that a computationally-focused researcher may lack.

---

## 8. BF-06-i: NISAR L-band InSAR in forested discontinuous permafrost (6.8)

### Motivation

Permafrost underlies 22% of Northern Hemisphere land area. Its degradation releases stored carbon (estimated 1,300-1,600 Gt C in the top 3 m) and causes ground subsidence that damages infrastructure. Monitoring permafrost thaw at landscape scale requires measuring centimeter-level ground subsidence across thousands of square kilometers. InSAR from Sentinel-1 can do this in tundra (Sadeghi Chorsi 2024), but fails in boreal forests because C-band radar loses coherence when it scatters off tree canopies. NISAR's L-band, operational since early 2026, has a longer wavelength that penetrates the canopy. This idea tests whether NISAR enables the first satellite-based permafrost subsidence monitoring in forested terrain.

### Approach

**Data.** NISAR L-band SLC acquisitions over 2026-2027 (first full annual cycle). Sentinel-1 C-band InSAR over the same area and period. ArcticDEM for topographic correction. GTN-P borehole temperatures. UAVSAR airborne L-band data (ABoVE campaign) as calibration reference.

**Method.** SBAS InSAR time series analysis on NISAR L-band data. Estimate seasonal subsidence amplitude and multi-year trend. Compare L-band vs. C-band coherence as a function of canopy density (from Sentinel-2 NDVI). Test whether L-band recovers the subsidence signal in areas where C-band fails (coherence < 0.3). Report the minimum canopy density at which L-band coherence degrades.

**Study area.** Interior Alaska, Fairbanks corridor. Discontinuous permafrost zone with mixed boreal forest. Chosen because Zwieback's group operates here, the borehole network is dense, and UAVSAR reference data exists.

**Timeline.** 18-24 months (constrained by accumulation of NISAR temporal baseline). Data available starting 2026. First results after 12 months of data. Full annual cycle at 18 months.

### Expected outcome

**Best case.** NISAR L-band maintains coherence > 0.5 in boreal forest with canopy cover up to 60-70%. Seasonal subsidence (20-60 mm) is clearly resolved. This opens landscape-scale permafrost monitoring across the forested discontinuous zone — a major methodological advance.

**Realistic case.** L-band coherence is better than C-band but degrades in dense forest (canopy > 40-50%). Subsidence is measurable in open woodland and forest edges but not in dense stands. Partial solution. Published as a technical assessment of NISAR L-band for permafrost monitoring in forested terrain.

**Null-result scenario.** L-band coherence in boreal forest is not sufficient for multi-temporal InSAR (vegetation temporal decorrelation at L-band is still too high). This would mean satellite InSAR is fundamentally limited to tundra for permafrost monitoring. Important negative result worth publishing.

### Fieldwork requirements

Moderate. In-situ GNSS subsidence stations and borehole temperature loggers at the study site are needed for validation. Zwieback's group at UAF operates these stations in the Fairbanks corridor. A collaboration model where you process the NISAR data and they provide field measurements is natural. If operating independently, deploying 5-10 GNSS stations at forest sites of varying density requires a 1-2 week summer field campaign plus periodic maintenance visits.

### Risks and dependencies

- **NISAR data quality and timing (medium-high).** NISAR launched in January 2024 but entered commissioning/calibration phase. Routine data availability for Alaska was expected by mid-2025 but has experienced delays. If NISAR data quality or coverage is insufficient, the entire study is blocked.
- **InSAR expertise (medium).** InSAR time series analysis (SBAS, atmospheric correction, unwrapping) has a steep learning curve. This is specialized remote sensing, not general ML. Requires either deep self-study or collaboration with an InSAR expert (Zwieback, Meyer).
- **Timeline.** The 18-24 month requirement for a full annual cycle means this is a slow project. First results are 12+ months away.

### Venue fit

**Journals.** The Cryosphere, Remote Sensing of Environment, Geophysical Research Letters (if the L-band coherence result is definitive).

**Funding.** NASA New Investigator Program (NISAR science), ESA Third Party Mission research funding.

**Institutions.** University of Alaska Fairbanks (Zwieback, Meyer — the natural collaborators). NASA JPL (NISAR science team). ESA CCI Permafrost (Bartsch).

### Honest assessment

**Strengths.** This idea has the highest Ecological Impact score (8) in the top 10. Landscape-scale permafrost monitoring in forests is a genuine measurement gap with major carbon cycle implications. The timing is right — NISAR is new and the community is looking for demonstration studies. Being early to publish on NISAR for permafrost confers a first-mover advantage.

**Weaknesses.** The idea depends entirely on NISAR data quality and availability, which is not under your control. InSAR processing is a specialized skill that takes months to learn. The learning curve is steep enough that without an InSAR expert collaborator, the project could stall on technical processing issues. This is the idea in the top 10 where the skill fit is most challenging — it requires deep geophysical remote sensing expertise, not ML. The ecological interpretation (what does subsidence mean for carbon release? for ecosystem change?) requires permafrost ecology expertise that is beyond a remote sensing researcher's normal scope.

---

## 9. BF-19-i: Boreal BirdNET fine-tuning with circumpolar training data (6.8)

### Motivation

BirdNET recognizes 6,000+ species globally but its performance degrades in boreal environments. The Yukon BBMP study showed it achieves reasonable detection at 22% of human validation cost, but only after intensive post-hoc filtering. The Italian Alps fine-tuning study (2025) demonstrated that regional fine-tuning improves accuracy substantially. No equivalent fine-tuning has been done for the circumpolar boreal bird assemblage. Boreal soundscapes have specific challenges: midnight sun recording conditions (continuous daylight changes dawn chorus timing and duration), heavy wind and insect noise, and species assemblages (warblers, sparrows, thrushes) with similar songs that require fine spectral discrimination.

### Approach

**Data.** Aggregate recordings from: Xeno-canto (filtered for boreal species list), Finnish Museum of Natural History (Luomus) archives, Canadian Wildlife Service Breeding Bird Survey recordings, Boreal Bird Monitoring Program (Yukon). Target: 100+ boreal species, 500+ recordings per species, balanced across circumpolar regions.

**Method.** Extract BirdNET embeddings (penultimate layer). Fine-tune a classification head on the boreal species subset. Compare against: (a) full fine-tuning of the last N layers with boreal-specific data augmentation (wind noise, rain, insect chorus overlays), (b) ArcticSoundsNet as an Arctic-specific baseline. Report per-species precision and recall. Evaluate on held-out PAM recordings from three regions (Finnish Lapland, Yukon, Interior Alaska).

**Study area.** Training data circumpolar; evaluation at three PAM sites spanning the boreal belt.

**Timeline.** 10-14 months. Data curation: 3-4 months (the bottleneck — cleaning and verifying species labels in archived recordings). Model fine-tuning: 2-3 months. Evaluation: 2-3 months. Paper: 2 months.

### Expected outcome

**Best case.** Boreal-specific fine-tuned model improves species-level F1 by 10-15% over base BirdNET across the target species list. Reduces false positives for confusion pairs (e.g., similar warbler songs). Released as a public model checkpoint usable by monitoring programs across the circumpolar belt.

**Realistic case.** Improvement of 5-10% in F1 for the target species, with larger gains for specific confusion-prone species. Performance varies by region — the model works best for the region with the most training data and degrades for underrepresented areas. Published as a regional fine-tuning case study.

**Null-result scenario.** BirdNET's base model is already near-optimal for boreal species, and fine-tuning provides < 3% improvement. The main performance limitation is recording quality (wind, distance), not model architecture. This would redirect effort toward hardware and deployment optimization rather than model fine-tuning.

### Fieldwork requirements

Low-moderate. Deploying AudioMoth or Swift recorders at 3-5 sites per region (Finnish Lapland, Yukon, Alaska) for one breeding season (May-July) provides evaluation data. Each deployment is 1-2 days of fieldwork per site. Retrieval at end of season requires a second visit. Total: 2-3 weeks across three regions, which likely requires local collaborators handling their regional deployments.

### Risks and dependencies

- **Training data quality (medium).** Xeno-canto recordings are often clean focal recordings, while PAM data is noisy soundscape recordings. Fine-tuning on Xeno-canto may not transfer well to PAM conditions. Augmentation with noise profiles helps but may not fully bridge this gap.
- **Species label verification (high effort).** Aggregating data from multiple archives requires verifying species labels, which is time-consuming and requires ornithological expertise.
- **Cornell collaboration.** BirdNET is maintained by Kahl at Cornell. Fine-tuning without their involvement is technically possible (the model is open) but publishing without them may create awkward dynamics. A collaborative model is preferable.

### Venue fit

**Journals.** Methods in Ecology and Evolution, Ecological Informatics, Ecological Applications.

**Conferences.** BirdCLEF (workshop paper), ICML (workshop on ML for biodiversity), Ecology conferences (ESA, IBFRA).

**Institutions.** Cornell Lab of Ornithology (Kahl, Klinck). Chemnitz University (Kahl). Finnish Museum of Natural History (Luomus). Environment and Climate Change Canada.

### Honest assessment

**Strengths.** The approach is proven (fine-tuning works for BirdNET, demonstrated in multiple biomes). The audio ML skill set is a strong fit. The product (a fine-tuned model) has immediate practical value for monitoring programs. The boreal conservation community is eager for better acoustic tools.

**Weaknesses.** Novelty is limited — this is regional fine-tuning of an existing model, not a methodological advance. The data curation bottleneck (cleaning and verifying species labels) is tedious and requires ornithological expertise that a computational researcher may lack. The project is useful but unlikely to appear in a top-tier venue without a broader scientific story (e.g., what does the model's performance tell us about boreal acoustic ecology that we didn't know?).

---

## 10. BF-06-ii: Multi-sensor permafrost vulnerability mapping (6.7)

### Motivation

Each satellite sensor captures one dimension of permafrost thaw: InSAR measures ground displacement, optical indices track vegetation change (thermokarst lake expansion, greening/browning), and thermal data captures surface energy balance shifts. Individually, each is ambiguous — InSAR displacement could be seasonal active-layer dynamics or irreversible degradation; vegetation change could be thaw-driven or climate-driven. Combining all three disambiguates the signal. A landscape-scale permafrost vulnerability classification (stable / seasonally active / actively degrading / fully degraded) doesn't exist for forested terrain.

### Approach

**Data.** Sentinel-1 InSAR SBAS time series (seasonal subsidence amplitude and trend). Landsat/Sentinel-2 NDVI and NDWI time series (vegetation and wetland expansion). Landsat TIRS or ECOSTRESS LST (surface temperature anomalies). ERA5-Land soil temperature and moisture. GTN-P borehole records as labels.

**Method.** Pixel-level fusion using random forest or gradient boosting trained on four permafrost states labeled from borehole records and expert interpretation: stable (< 2 mm/yr subsidence, no vegetation change), seasonally active (> 10 mm seasonal amplitude but stable trend), actively degrading (increasing subsidence trend, wetland expansion), and fully degraded (no seasonal signal, thaw-lake formation). Produce a vulnerability raster with uncertainty (class probability).

**Study area.** Start with Yukon (Teslin area — discontinuous permafrost, mixed forest, accessible, studied).

**Timeline.** 15-20 months.

### Expected outcome

**Best case.** A four-class permafrost state product with overall accuracy > 80% validated against borehole records. Identifies hotspots of active degradation not captured by any single sensor. Demonstrates clear added value of multi-sensor fusion.

**Realistic case.** Classification works for the three-class case (stable / dynamic / degraded) but can't distinguish seasonally active from actively degrading without long time series. Accuracy 65-75%. Published as a proof-of-concept for the Teslin area with methodological discussion of sensor complementarity.

**Null-result scenario.** The sensors provide redundant rather than complementary information — InSAR displacement correlates with vegetation change which correlates with thermal anomaly, and the fusion adds no discrimination power beyond InSAR alone. This would simplify future monitoring (just use InSAR) but is a less interesting paper.

### Fieldwork requirements

Moderate. Need borehole data for training labels. GTN-P provides some coverage, but the Teslin area may need supplementary boreholes or at minimum ground-truthing visits. Collaboration with permafrost researchers at UAF, NRCan, or Yukon Geological Survey would provide both data and field access. A 1-2 week summer field campaign for ground-truthing (visual assessment of thaw features: thermokarst, drunken forest, thaw ponds) is strongly recommended.

### Risks and dependencies

- **Label scarcity.** Borehole records are sparse, especially in the Yukon study area. The training set may be too small for supervised learning. May need semi-supervised approaches or expert-labeled training from satellite interpretation.
- **InSAR quality.** If using Sentinel-1 C-band, coherence will be poor in forested areas (the same problem as BF-06-i). This idea benefits from NISAR L-band but can proceed without it using C-band in open areas.
- **Domain expertise.** Defining the four permafrost state classes and labeling training data requires permafrost expertise. This is not something a computational researcher can do independently.

### Venue fit

**Journals.** Remote Sensing of Environment, The Cryosphere, Permafrost and Periglacial Processes.

**Institutions.** UAF (Zwieback, Meyer). NRCan Geological Survey of Canada. NASA JPL (ECOSTRESS, NISAR). ESA CCI Permafrost (Bartsch).

### Honest assessment

**Strengths.** Multi-sensor fusion is a computationally natural approach and a strong ML fit. The product has clear end users (infrastructure planners, climate modelers). The idea complements BF-06-i (if L-band InSAR works in forest, this idea integrates it with other sensors).

**Weaknesses.** The idea depends on having good InSAR data in forested terrain, which is the unsolved problem that BF-06-i addresses. If BF-06-i fails (L-band coherence insufficient), this idea is weakened. The label scarcity problem is real — with only a handful of borehole records in the study area, supervised classification is strained. The domain knowledge requirement (permafrost state classification) is high, and getting it wrong would produce a misleading product. Of the top 10 ideas, this is the one most dependent on having the right collaborator.

---

## Summary: Top 10 at a glance

| Rank | ID | Title | Score | Timeline | Fieldwork | Skill demand |
|---|---|---|---|---|---|---|
| 1 | BF-03-i | Reburn probability surfaces | 7.7 | 12-18 mo | Minimal | ML + remote sensing |
| 2 | BF-09-ii | Treeline advance rates | 7.3 | 12-15 mo | Low | Time series + RS |
| 3 | BF-09-i | 10 m treeline ecotone | 7.2 | 15-18 mo | Low | Deep learning + fusion |
| 4 | BF-01-i | SAR-optical fire detection | 7.2 | 12-15 mo | None | Multi-sensor DL |
| 5 | BF-03-iii | Conifer-deciduous conversion | 7.0 | 12-18 mo | Low | Spectral unmixing |
| 6 | BF-16-ii | IFL connectivity planning | 7.0 | 10-14 mo | None | Conservation GIS |
| 7 | BF-04-i | Compositional recovery mapping | 7.0 | 12-18 mo | Low | Spectral unmixing |
| 8 | BF-06-i | NISAR L-band InSAR permafrost | 6.8 | 18-24 mo | Moderate | InSAR (specialized) |
| 9 | BF-19-i | Boreal BirdNET fine-tuning | 6.8 | 10-14 mo | Low-moderate | Audio ML |
| 10 | BF-06-ii | Multi-sensor permafrost vuln. | 6.7 | 15-20 mo | Moderate | Multi-sensor fusion |

### Distribution notes

**Fire problems dominate (4/10).** BF-03-i, BF-03-iii, BF-01-i, BF-04-i are all fire-related. The same rubric bias that promoted fire problems in Stage 3 carries through to ideas: fire problems have the cleanest data and the most established computational methods.

**Three ideas share methods.** BF-03-iii, BF-04-i, and (partially) BF-03-i all use Landsat spectral unmixing + needleleaf index. Pursuing all three means three papers from the same data processing pipeline, which is efficient but risks appearing incremental.

**The two permafrost ideas (BF-06-i, BF-06-ii) are coupled.** BF-06-ii assumes BF-06-i works. A researcher would pursue BF-06-i first and BF-06-ii only if L-band coherence is sufficient. This sequential dependency is a risk.

**Ideas that just missed the cut.** BF-06-iii (fire-permafrost interaction, 6.7), BF-18-i (NISAR windthrow, 6.7), BF-18-ii (windthrow alerting, 6.7), BF-19-ii (foundation model multi-taxon, 6.7), BF-16-i (IFL risk forecasting, 6.7), and BF-04-ii (multi-sensor recovery staging, 6.7) all scored 6.7. Any of these could replace the #10 idea without a meaningful quality difference. The 6.7 tier contains some of the most ecologically ambitious ideas (fire-permafrost interaction, IFL risk forecasting) that were penalized by lower Confidence and Feasibility scores.
