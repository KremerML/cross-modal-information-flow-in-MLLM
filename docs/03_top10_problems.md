# Stage 3: Top 10 problems

Mechanical cut by aggregate score. No curation.

## Top 10

| Rank | ID | Title | Aggregate |
|---|---|---|---|
| 1 | BF-03 | Short-interval reburns increasing with no operational reburn risk product | 6.8 |
| 2 | BF-09 | Boreal tree cover shifting northward but treeline ecotone poorly resolved | 6.5 |
| 3 | BF-01 | Small and low-intensity fires missed by satellite monitoring | 6.3 |
| 4 | BF-06 | Permafrost thaw monitored at point scale but not landscape scale in forested regions | 6.3 |
| 5 | BF-18 | Storm damage detection in boreal forests is slow and incomplete | 6.3 |
| 6 | BF-19 | Bioacoustic monitoring of boreal biodiversity limited by lack of boreal-specific ML models | 6.3 |
| 7 | BF-16 | Intact forest landscapes shrinking faster than they are being protected | 6.2 |
| 8 | BF-02 | No operational fire spread prediction for boreal forests | 6.0 |
| 9 | BF-04 | Spectral recovery after fire does not indicate ecological recovery | 6.0 |
| 10 | BF-15 | SIF-based drought stress monitoring in boreal forests promising but unvalidated | 6.0 |

## Ties

Three-way tie at rank 3-6 (BF-01, BF-06, BF-18, BF-19 all at 6.3). Ordering within the tie is arbitrary. All four advance.

Three-way tie at rank 8-10 (BF-02, BF-04, BF-15 all at 6.0). Ordering within the tie is arbitrary. The next three problems below the cut (BF-11, BF-12, also at 5.8) are close but did not advance.

## Sub-area distribution

| Sub-area | In top 10 | Total in Stage 1 |
|---|---|---|
| Fire regimes | 4 | 4 |
| Carbon/permafrost | 1 | 4 |
| Treeline dynamics | 1 | 1 |
| Forest health/pests | 1 | 3 |
| Biodiversity | 1 | 2 |
| Land use/disturbance | 2 | 3 |
| Hydrology/wetlands | 0 | 1 |
| Indigenous monitoring | 0 | 1 |
| Cross-cutting | 0 | 1 |

## Distribution analysis

The skew is real. All four fire problems made the top 10, while carbon/permafrost dropped from 4 to 1, and Indigenous monitoring, wetlands, and the Russia cross-cutting problem all fell out.

This is exactly the rubric bias I flagged before scoring. Fire problems score high on Tractability (clean satellite data), Confidence (measurable signals), and Skill Fit (classic remote sensing + ML). Carbon attribution (BF-05, BF-07), deep soil carbon (BF-08), and Indigenous tools (BF-17) scored high on Ecological Impact but were penalized by the Tractability-Confidence correlation and the Skill Fit criterion.

A few specific tensions worth naming:

**BF-05 (carbon sink attribution, aggregate 5.3) vs. BF-18 (storm damage, aggregate 6.3).** Storm damage detection is arguably a solved problem waiting for NISAR data, while carbon sink attribution is a frontier question with enormous policy implications. The rubric put storm damage 1.0 points higher because it's more tractable. I think this is a case where the rubric is doing its job (identifying computationally tractable entry points) but the user should know what it's trading away.

**BF-17 (Indigenous tools, aggregate 5.2) vs. BF-01 (small fire detection, aggregate 6.3).** Indigenous tools scored low on Tractability (4), Feasibility (4), Confidence (4), and Skill Fit (4), which together overwhelmed its high Novelty (8) and reasonable Impact (7). The rubric is right that this problem requires social science more than ML, and a computationally-oriented researcher would struggle to lead it. But it's worth noting that a collaboration role (providing computational support within an Indigenous-led project) might be the highest-impact use of ML skills in this entire list.

**BF-06 (permafrost in forest, aggregate 6.3) is the strongest carbon/permafrost entry.** It survived because NISAR provides a concrete, tractable computational problem (L-band InSAR processing) while the other carbon problems (BF-05, BF-07, BF-08) involve harder inverse problems with less clear computational pathways.

**BF-19 (bioacoustics, aggregate 6.3) is a surprise inclusion.** It ranked alongside fire detection and storm damage despite being outside the traditional remote-sensing-heavy boreal monitoring space. This is because it scores high on Novelty (7, genuinely underexplored) and Skill Fit (8, audio ML and transfer learning are strong computational fits). Whether it belongs above BF-13 (caribou, 5.7) or BF-05 (carbon sink, 5.3) depends on whether you weight tractability or ecological urgency. The rubric weighted tractability.

**Overall assessment:** The top 10 is biased toward computationally tractable, remote-sensing-heavy problems. This is the intended behavior of the pipeline (identifying where computational skills have leverage) but the user should treat the ranking as "where computation can contribute" not "what matters most ecologically." The four problems that fell out on ecological impact grounds (BF-05, BF-07, BF-14, BF-17) remain important, just harder to crack with a computational-first approach.

---

## Potential collaborators by problem

For each top-10 problem: key research groups, institutions, NGOs, and individual researchers whose work directly overlaps. Organized as concrete starting points for outreach, not an exhaustive directory.

### BF-03: Short-interval reburns

**Research groups and researchers:**
- **Jennifer Baltzer** (Wilfrid Laurier University, Canada). Tier I Canada Research Chair in Forests and Global Change. Leads fieldwork on post-fire recovery, overwintering fires, and conifer-to-deciduous conversion across the NWT and Alaska. First team in the world to collect samples at overwintering fire sites. Direct overlap with reburn ecology. [wlu.ca](https://www.wlu.ca/academics/faculties/faculty-of-science/faculty-profiles/jennifer-baltzer/index.html)
- **Ellen Whitman** (Natural Resources Canada, Canadian Forest Service). Lead author of the 2024 Global Change Biology paper demonstrating that modest fire weather increases overcome reburn resistance. Based at the Northern Forestry Centre in Edmonton. Works at the interface of fire science and operational forest management.
- **Brian Buma** (University of Colorado Denver). Published the foundational work on short-interval fire increases in Alaskan boreal forests (Scientific Reports, 2022). Focuses on forest resilience and disturbance interactions.
- **Alan Tepley** (University of Montana / USDA Forest Service Rocky Mountain Research Station). Lead author of the 2025 Ecosystems paper quantifying the fire-vegetation feedback at biome scale. Brings the pan-boreal perspective.
- **Xanthe Walker, Michelle Mack** (Northern Arizona University / University of Florida). Extensively published on fire-driven vegetation change and regeneration failure in boreal North America, including ABoVE field datasets.
- **Sander Veraverbeke** (Vrije Universiteit Amsterdam). ERC Consolidator Grant holder for the FireIce project. Combines fieldwork, remote sensing, and modeling to study Arctic-boreal fire-permafrost feedbacks. Built the circumpolar fire tracking system with Scholten et al. (2024). Strong on the remote sensing side of reburn detection. [vu.nl](https://vu.nl/en/research/scientists/sander-veraverbeke)

**Institutions:**
- Woodwell Climate Research Center (Falmouth, MA). Houses the Arctic Program led by Susan Natali and the fire research led by Brendan Rogers. Directly involved in ABoVE and boreal fire carbon accounting. Published the major Virkkala et al. (2024) paper on wildfire-CO2 sink interactions.
- Canadian Forest Service / Natural Resources Canada (Edmonton, Northern Forestry Centre). Operational fire science for Canadian boreal. Whitman and colleagues are based here.
- USDA Forest Service, Rocky Mountain Research Station. Tepley and colleagues.

**NGOs:**
- Woodwell Climate (also functions as an NGO with policy engagement)
- Pew Charitable Trusts, International Boreal Conservation Campaign. Advocacy-focused but commissions technical reports.

**International programs:**
- NASA ABoVE (Arctic-Boreal Vulnerability Experiment). Winding down but generated the largest post-fire field dataset (1,538 sites). Science Team Lead: Scott Goetz (NAU).

---

### BF-09: Treeline ecotone resolution

**Research groups and researchers:**
- **Min Feng** (Tsinghua University, Beijing). Lead author of the 2026 Biogeosciences paper confirming northward tree cover shift using 224,026 Landsat images. Developed the 30 m annual tree cover maps. Collaborates with Sexton at UMD.
- **Joseph Sexton** (University of Maryland / terraPulse Inc.). Co-author on Feng et al. (2026). Long track record in global land cover products from Landsat. Expertise in calibrating tree cover time series.
- **Logan Berner, Scott Goetz** (Northern Arizona University, GEODE Lab). Goetz is ABoVE Science Team Lead. Berner lead-authored the 2022 Global Change Biology paper documenting boreal biome shift from Landsat. Their lab has decades of Arctic-boreal satellite vegetation monitoring. [goetzlab.rc.nau.edu](https://goetzlab.rc.nau.edu/)
- **Ryan Danby** (Queen's University, Canada). Works on treeline ecology, climate change effects on forest-tundra ecotone. Field-based complement to remote sensing approaches.
- **Marc Macias-Fauria** (University of Oxford). Forest-tundra dynamics and treeline ecology across the pan-Arctic. Published on climate constraints on tree range expansion.

**Institutions:**
- University of Maryland, GLAD Lab (Global Land Analysis and Discovery). Hansen, Potapov, and Turubanova. Primary producers of Landsat-based global forest products. Potential for treeline product development.
- NASA Goddard Space Flight Center. Houses the HLS (Harmonized Landsat-Sentinel) project and ICESat-2 mission science. Direct relevance for fusion approaches.
- Swedish University of Agricultural Sciences (SLU), Department of Forest Resource Management, Remote Sensing Division. Active in boreal forest mapping, airborne LiDAR, and multi-sensor fusion. Currently recruiting PhD students in boreal remote sensing. [slu.se](https://www.slu.se/en/departments/forest-resource-management/sections/forest-remote-sensing/)

**International programs:**
- ESA Climate Change Initiative (CCI) Biomass. Produces pan-boreal AGB maps. Relevant for carbon-in-new-forest quantification.

---

### BF-01: Small and low-intensity fire detection

**Research groups and researchers:**
- **Stefano Potter** (Woodwell Climate). Lead author of the 2026 UNet++ Arctic-boreal burned area paper. Working directly on the deep-learning solution to this problem.
- **Sander Veraverbeke, Randi Scholten** (VU Amsterdam). Built the circumpolar fire atlas (2012-2023) from VIIRS active fire detections. Scholten's sub-daily fire tracking system is directly relevant.
- **Elsa Hassellof** (Swedish Meteorological and Hydrological Institute, SMHI). Lead author of the 2026 validation paper on Sweden's automated VIIRS-based wildfire detection system. The only fully operational satellite fire detection system in the boreal zone with published validation statistics.
- **Robert Hall** (Canadian Forest Service). Lead author of the 2021 "Missing Burns" paper documenting MODIS's failure to detect half of boreal burned area.

**Institutions:**
- SMHI / MSB (Swedish Civil Contingencies Agency). Operate the Nordic satellite fire detection system. Expanding to Norway, Finland, Estonia.
- NASA FIRMS (Fire Information for Resource Management System). Operational global fire detection. Key contact for data access and algorithm development.

**NGOs:**
- Woodwell Climate Research Center (as above).

---

### BF-06: Permafrost thaw in forested regions

**Research groups and researchers:**
- **Simon Zwieback** (University of Alaska Fairbanks, Geophysical Institute). Leading researcher on InSAR-based permafrost monitoring. Multiple NASA grants. Published on excess ground ice profiling from InSAR subsidence (Water Resources Research, 2024) and advances in InSAR analysis of permafrost terrain (Permafrost and Periglacial Processes, 2024). The person closest to solving the forested-permafrost InSAR problem.
- **Franz Meyer** (University of Alaska Fairbanks / Alaska Satellite Facility). Co-author on InSAR permafrost work. Director of ASF DAAC which distributes NISAR data. Deep SAR expertise.
- **Taha Sadeghi Chorsi** (University of South Florida). Lead author of the 2024 Cryosphere paper demonstrating long-term permafrost monitoring with Sentinel-1 InSAR in Alaska's North Slope.
- **Annett Bartsch** (b.geos GmbH, Austria / ESA CCI Permafrost). Leading the ESA CCI Permafrost project producing Essential Climate Variable products. Expertise in SAR-based permafrost indicators.

**Institutions:**
- University of Alaska Fairbanks, Permafrost Laboratory and Geophysical Institute. Houses the remote sensing group focused on permafrost landscape dynamics. Also home to Santosh Panda (boreal wildfire fuel mapping, permafrost community monitoring).
- Alaska Satellite Facility DAAC (ASF). Distributes all NISAR data. Key for data access and processing support.
- NASA Jet Propulsion Laboratory (JPL). NISAR mission lead. ARIA and OPERA initiatives producing standardized InSAR displacement products.

**International programs:**
- Global Terrestrial Network for Permafrost (GTN-P). Borehole temperature network. Field validation data source.
- ESA CCI Permafrost project.

---

### BF-18: Storm damage detection

**Research groups and researchers:**
- **Markus Holopainen** (University of Helsinki). Published the 2021 Finnish windthrow SAR detection study. Long track record in forest remote sensing and LiDAR.
- **Lars Ulander** (Chalmers University, Sweden). SAR specialist who has worked on windthrow mapping in Swedish boreal forests. L-band SAR expertise relevant to NISAR applications.
- **Matthieu Molinier** (VTT Technical Research Centre, Finland). Active in SAR-based forest monitoring including storm damage detection in Finnish boreal forests.

**Institutions:**
- VTT Technical Research Centre of Finland. Active in SAR processing and forest applications.
- Luke (Natural Resources Institute Finland). National forest inventory data, ALS coverage, and storm damage statistics.
- SLU (Sweden). Department of Forest Resource Management has ongoing work on windthrow risk modeling.

**NGOs/industry:**
- ICEYE (Finland/US). Operates a commercial SAR satellite constellation. Provides rapid-revisit SAR for emergency response including storm damage. Potential for rapid detection partnerships.
- Capella Space (US). Another commercial high-resolution SAR provider demonstrated for windthrow assessment in Scotland.

---

### BF-19: Bioacoustic monitoring

**Research groups and researchers:**
- **Stefan Kahl** (Cornell Lab of Ornithology, K. Lisa Yang Center for Conservation Bioacoustics). Lead of the BirdNET team. BirdNET recognizes 6,000+ species but boreal-specific fine-tuning is limited. The person to talk to about building boreal acoustic classifiers. [birdnet.cornell.edu](https://birdnet.cornell.edu/)
- **Holger Klinck** (Cornell Lab of Ornithology). Director of the K. Lisa Yang Center. Oversees the broader bioacoustics research program.
- **Dan Stowell** (Tilburg University / formerly Queen Mary, Naturalis Biodiversity Center). Computational bioacoustics researcher, co-organizer of BirdCLEF challenges, published on foundation models for bioacoustics.

**Institutions:**
- Cornell Lab of Ornithology, K. Lisa Yang Center for Conservation Bioacoustics. The global hub for bioacoustics AI. BirdNET, Raven software, Merlin app.
- Chemnitz University of Technology (Germany). Kahl's home institution. Collaborates with Cornell on BirdNET development.
- Finnish Museum of Natural History (Luomus, University of Helsinki). Maintains Finland's bird monitoring programs and acoustic archives. Potential source of boreal training data.
- Environment and Climate Change Canada, Canadian Wildlife Service. Runs the Breeding Bird Survey and boreal bird monitoring programs. Key source of validation data.

**NGOs:**
- Boreal Songbird Initiative (US/Canada). Advocacy organization focused on boreal bird conservation. Could provide field network access.
- Wildlife Conservation Society (WCS) Canada. Active in boreal biodiversity monitoring with field presence.

**International programs:**
- BirdCLEF (annual competition). Benchmark for bird sound identification from ML. Could push for a boreal-specific track.

---

### BF-16: Intact forest landscape protection

**Research groups and researchers:**
- **Peter Potapov, Svetlana Turubanova** (University of Maryland, GLAD Lab). Primary producers of the IFL dataset. Potapov is also at WRI. Published the 2025 PNAS paper on unprecedented fire-driven forest disturbance.
- **Matthew Hansen** (University of Maryland, GLAD Lab). Senior figure in global forest monitoring. Oversees the Global Forest Change dataset that underpins GFW.
- **Atte Moilanen** (University of Helsinki). Developer of Zonation conservation prioritization software. Directly relevant for spatial optimization of IFL protection.

**Institutions:**
- World Resources Institute (WRI), Global Forest Watch. Operational monitoring platform. Houses Potapov and collaborates with GLAD. The institutional home for IFL monitoring.
- University of Maryland, GLAD Lab. Produces the science behind GFW.
- University of Helsinki, Conservation Biology Informatics Group. Zonation development and application.

**NGOs:**
- WCS Canada. Active in boreal conservation advocacy and Indigenous Protected and Conserved Areas (IPCAs).
- Pew Charitable Trusts, International Boreal Conservation Campaign. Major funder and advocate for boreal IFL protection.
- The Nature Conservancy (TNC) Canada.
- Greenpeace International. Co-producer of the original IFL mapping methodology.

---

### BF-02: Fire spread prediction

**Research groups and researchers:**
- **Sibo Cheng** (Imperial College London). Co-author on BCWildfire benchmark. Expertise in spatiotemporal deep learning for environmental prediction.
- **Zhengsen Xu, Lincoln Xu** (University of Waterloo). BCWildfire dataset creators. Published at AAAI-26. Active in deep learning for geospatial applications.
- **Mike Flannigan** (Thompson Rivers University, Canada). Dean of Canadian fire science. Long career in fire weather, fire behavior modeling, and climate change impacts on fire regimes. Not a deep learning person, but the domain expert you'd want as a collaborator.
- **Piyush Jain** (Canadian Forest Service, NRCan). Lead author of the 2024 Nature Communications paper on drivers of the 2023 Canadian fire season. Works on fire weather indices and statistical fire prediction.

**Institutions:**
- Canadian Forest Service, NRCan (various centres). Operational fire management and fire weather forecasting. The end users of any fire spread prediction tool.
- Canadian Interagency Forest Fire Centre (CIFFC). Coordinates fire response across Canada. Potential partner for operational deployment.

---

### BF-04: Spectral vs. ecological recovery

**Research groups and researchers:**
- **Jennifer Baltzer** (Wilfrid Laurier University). As above. Her field data on post-fire conifer-to-deciduous conversion is the ground truth that satellite-based recovery products need to match.
- **Brendan Rogers, Stefano Potter** (Woodwell Climate). Rogers runs fire-carbon research. Potter's burned area mapping work is the foundation for recovery tracking. Woodwell also holds the ABoVE post-fire regeneration dataset.
- **Scott Goetz, Logan Berner** (Northern Arizona University). Their satellite time series (30 m, 40 years) provide the Landsat backbone for long-term recovery tracking. Berner's vegetation index work directly addresses the greening-vs-browning distinction.
- **Yili Zheng** (Chinese Academy of Sciences / collaborators). Lead author of the 2024 Remote Sensing of Environment paper demonstrating spectral vs. compositional recovery mismatch for the 1987 Siberian megafire.

**Institutions:**
- Woodwell Climate Research Center. Holds the ABoVE post-fire regeneration field dataset (1,538 sites, 58 fire perimeters).
- ORNL DAAC. Distributes ABoVE datasets. Data access for any recovery tracking project.

---

### BF-15: SIF drought monitoring

**Research groups and researchers:**
- **Zoe Pierrat** (NASA JPL / Caltech). Published the foundational boreal SIF-GPP work at the SOBS site in Saskatchewan (JGR Biogeosciences, 2022). Showed SIF + vegetation indices in random forest models achieve R2=0.94 for cross-site GPP prediction. The closest researcher to the boreal SIF validation problem.
- **Rui Cheng** (Caltech / Carnegie Institution). Evaluated TROPOMI SIF across ABoVE land cover types. Showed that SIF-GPP regression slopes vary strongly by land cover, meaning generalized relationships fail without region-specific calibration. Key finding for boreal-specific validation.
- **Prabhat Behera** (co-author, Geophysical Research Letters, 2025). Demonstrated SIF yield as a drought early warning indicator.
- **Timo Vesala, Ivan Mammarella** (University of Helsinki, ICOS station Hyytiala). Manage one of the longest-running boreal eddy covariance tower sites. Hyytiala provides the gold-standard ground truth for validating satellite SIF products in boreal conifer forest.

**Institutions:**
- NASA JPL / Caltech. Pierrat and colleagues. Also manages ECOSTRESS (thermal drought monitoring from ISS).
- University of Helsinki, Institute for Atmospheric and Earth System Research (INAR). Operates the Hyytiala SMEAR II station. Ground-truth tower data for boreal SIF validation.
- ICOS (Integrated Carbon Observation System). European flux tower network including multiple boreal sites in Finland and Sweden.
- University of Saskatchewan. Operates the SOBS flux tower. Another key validation site.

---

### Cross-cutting institutions worth noting

These institutions appear across multiple problems and represent broad potential partnerships:

| Institution | Relevant problems | Strength |
|---|---|---|
| Woodwell Climate Research Center | BF-01, BF-03, BF-04, BF-05 | Arctic Program, fire-carbon research, ABoVE datasets |
| Northern Arizona University (GEODE Lab) | BF-03, BF-04, BF-09 | ABoVE Science Team Lead, satellite vegetation time series |
| University of Alaska Fairbanks | BF-06, BF-03 | Permafrost InSAR, fire science, NISAR data access via ASF |
| VU Amsterdam | BF-01, BF-03 | Fire tracking, FireIce ERC project, pan-Arctic fire atlas |
| University of Maryland (GLAD Lab) | BF-09, BF-16 | Global forest products, IFL mapping, GFW |
| SLU (Sweden) | BF-09, BF-18 | Boreal forest remote sensing, ALS, multi-sensor fusion |
| Luke (Finland) | BF-18, BF-15 | National forest inventory, LiDAR, Nordic forestry |
| Canadian Forest Service / NRCan | BF-01, BF-02, BF-03 | Operational fire management, fire weather, policy interface |
| Cornell Lab of Ornithology | BF-19 | BirdNET, bioacoustics AI, global acoustic monitoring |
| University of Helsinki | BF-15, BF-16, BF-18 | Hyytiala flux tower, Zonation, boreal ecology |

### A note on approach

Cold-emailing a senior PI will get you a polite non-response. The more effective path is:

1. **Read their most recent 2-3 papers.** Identify the specific dataset, method, or gap you'd bring something to.
2. **Target PhD students and postdocs.** They're closer to the tools and more likely to engage with a computational collaborator.
3. **Conferences as entry points.** AGU, EGU, IGARSS, and the International Boreal Forest Research Association (IBFRA) annual meeting are where these communities gather.
4. **Bring a concrete offer.** "I can build X using your data" is stronger than "I'm interested in your area." If you've trained a model on open data that relates to their problem, show the result.
5. **For European institutions** (VU Amsterdam, Helsinki, SLU), Horizon Europe and ERC grants actively seek cross-institutional collaboration. Positioning as a computational collaborator for an ecology PI's next grant application is a viable entry strategy from ILLC/UvA.
