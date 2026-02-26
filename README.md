# Microbiome Dysbiosis Analysis in Inflammatory Bowel Disease

## Background
Inflammatory Bowel Disease (IBD), encompassing Crohn's Disease (CD) and Ulcerative Colitis (UC), is a complex chronic condition closely associated with disruptions to the gut community ecosystem, termed dysbiosis. The microbial communities occupying the intestinal mucosa serve critical roles in immune modulation and mucosal barrier maintenance. 

Sequencing the 16S rRNA gene allows us to profile these taxonomic features. However, next-generation sequencing generates **compositional data**—where sequence counts are intrinsically bounded by total read depth. To evaluate ecological perturbations effectively, we utilized **Centered Log-Ratio (CLR)** transformations to accurately model microbial dynamics alongside classical Alpha and Beta diversity indices that delineate ecosystem structure.

## Data Source
The data utilized in this repository was obtained from the [Inflammatory Bowel Disease Multi'omics Database (IBDMDB)](https://ibdmdb.org).
- **Product:** HMP2 2018.06
- **Data Type:** 16S rRNA Sequencing
- **Processing:** Processed and quality-controlled through the standardized HMP2 AnADAMA workflow.

Our pipeline specifically utilizes the publicly available processed taxonomic abundance tables natively rather than raw FASTQ reads. To prevent longitudinal data leakage in downstream modeling, biopsy records originating from identical participants were logically aggregated into cross-sectional participant-level profiles.

## Methods
Our analysis pipeline sequentially implemented the following strategies:
- **Taxonomic Processing:** Aggregation of raw Operational Taxonomic Units (OTUs) to the Genus level, followed by prevalence filtering (retaining taxa present in $\geq 10\%$ of samples).
- **Compositional Normalization:** Calculation of relative abundances and strict Centered Log-Ratio (CLR) geometric transformations.
- **Statistical Ecology:** 
  - Alpha Diversity: Shannon index distribution assessed via Kruskal-Wallis testing.
  - Beta Diversity: Explicit distance modeling using Bray-Curtis metrics, mapped via Principal Coordinate Analysis (PCoA) and assessed statistically against PERMANOVA.
- **Machine Learning Classification:** 
  - Participant-level data split separating training and testing cohorts iteratively.
  - Algorithm evaluations benchmarking Random Forest architectures against Multinomial Logistic Regression.
  - Model scoring utilizing 5-Fold Stratified Cross-Validation, One-vs-Rest ROC curve outputs, and Macro-AUC.
- **Biomarker Identification:** Feature extraction calculated utilizing test-set Permutation Importance to identify stable taxa.

## Results
The computational modeling assessed complex clinical phenotypes spanning CD, UC, and non-IBD:
- **Alpha Diversity:** We observed a visual trend toward reduced ecological complexity within IBD microenvironments, though this did not reach statistical significance ($p \approx 0.10$).
- **Beta Diversity:** Community structuring demonstrated a modest but statistically significant community separation between clinical phenotypes (PERMANOVA $p = 0.042$).
- **Machine Learning Performance:** 
  - Logistic Regression achieved the strongest generalized classification performance, logging a highest Macro F1 $\approx 0.62$.
  - The models highlighted moderate predictive performance across disease classes mapping Macro-AUC scores ranging from $\approx 0.75$ to $0.77$.
- **Predictive Genera:** The permutation model successfully extracted top stable community predictors (e.g., *Ruminococcus*, *Subdoligranulum*, *Akkermansia*).

## Biological Insights
Our evaluation identified specific compositional shifts that align conceptually with dysbiosis literature. Distinct relative depletions of strictly anaerobic, butyrate-synthesizing *Firmicutes* clades were noted when contrasting IBD phenotypes against non-IBD controls. Conversely, active inflammatory environments exhibited relative preservation or expansions of mucin-degrading clades (*Akkermansia*) and specific facultative anaerobes. These shifts characterize broader ecological restructuring (e.g., altered oxidative states and disrupted anti-inflammatory feedback loops) rather than stark delineations driven by isolated pathogen expansions.

## Limitations
- **Sample Size Constraints:** Modest patient dimension ($n \approx 81$ distinct participants post-aggregation) limits generalized statistical power.
- **Cross-Sectional Dynamics:** Evaluations utilizing solely cross-sectional abundance data cannot track active disease flaring dynamics.
- **Taxonomic Resolution:** 16S evaluations are inherently constrained to Genus level delineations, lacking explicit species/strain resolution.
- **Correlation vs Causation:** This investigation determines taxonomic correlations; the data structure fundamentally prevents asserting causations connecting microbial alterations strictly to inflammatory progression.

## Future Directions
- **Metagenomics Profiling:** Broadening taxonomic identification natively targeting functional gene profiles.
- **Multi-Omic Integration:** Incorporating Metatranscriptomics and targeted Metabolomics (specifically quantifying SCFA concentrations like butyrate) to validate explicitly active metabolic pathways.
- **Longitudinal Modeling:** Investigating IBD flaring mechanisms and remission dynamics natively tracking multi-timepoint host profiles.

## Reproducibility
This project natively utilizes Python 3 environments. Core processing infrastructures leverage:
- `pandas`
- `numpy`
- `scikit-learn`
- `scikit-bio`
- `scipy`
- `seaborn`
