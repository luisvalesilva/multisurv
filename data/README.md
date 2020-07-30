# Data

All data used in the study are from the The Cancer Genome Atlas (TCGA) program, which includes a rich body of imaging, clinical, and molecular data from 11,315 cases of 33 different cancer types ([Weinstein _et al._, Nat Genet 2013](https://www.nature.com/articles/ng.2764)). The data are made available by the National Cancer Institute (NCI) Genomic Data Commons (GDC) information system, publicly accessible at the [GDC Data Portal](https://portal.gdc.cancer.gov/).

Section <a href="#download"><emph>Download</emph></a> below details the procedures used to download the raw data.
After download, the raw data were preprocessed in preparation for modeling using the following dedicated Jupyter notebooks:

* Clinical data - [Jupyter notebook](preprocess_clinical.ipynb)
* Omics data - [Jupyter notebook](preprocess_omics.ipynb)
* WSIs - [Jupyter notebook](preprocess_wsi.ipynb)


## Download

<p>
  •
  <a href="#clinical-data">Clinical data</a><br />
  •
  <a href="#gene-expression">Gene expression</a><br />
  •
  <a href="#mirna-expression">miRNA expression</a><br />
  •
  <a href="#dna-methylation">DNA methylation</a><br />
  •
  <a href="#copy-number-variation">Copy number variation</a><br />
  •
  <a href="#whole-slide-images-(wsi)">Whole-slide images (WSI)</a>
</p>

Clinical data was downloaded using the `TCGAbiolinks` R package as detailed below. All other modalities were downloaded from the __GDC Data Portal__ for all cancer entities in the TCGA program using the
[GDC Data Transfer Tool](https://gdc.cancer.gov/access-data/gdc-data-transfer-tool) ([docs](https://docs.gdc.cancer.gov/Data_Transfer_Tool/Users_Guide/Getting_Started/)).
This takes an appropriate _Manifest_ file as input, obtained according to the following general procedure:

1. Go to [GDC Data Portal](https://portal.gdc.cancer.gov/);
1. Navigate to [Repository](https://portal.gdc.cancer.gov/repository) and check box for `TCGA` program in the `Cases` tab;
1. Filter the data (using the interactive pie charts or check boxes);
1. Use check boxes on the left to select the appropriate experimental strategies;
1. Push `Manifest` button to download file.

The generated _Manifest_ files are stored here in the `data` folder. The code used to download the data from the respective manifest file is reproduced below for each data modality.

### Clinical data

Used `TCGAbiolinks` R package to access clinical data using the following R code.

```r
# Download data for all TCGA projects
project_ids <- stringr::str_subset(TCGAbiolinks::getGDCprojects()$project_id, 'TCGA')

data <- list()

for (project_id in project_ids) {
    data[[project_id]] <- TCGAbiolinks::GDCquery_clinic(project=project_id, type='clinical')
}

# Merge into single table
# (the "disease" column identifies each original table)
data <- do.call(dplyr::bind_rows, data)

# Write to file
output_path <- '/mnt/dataA/TCGA/raw/clinical_data.tsv'
readr::write_tsv(data, output_path)
```

### Gene expression

The data are provided either as read counts or FPKM/FPKM-UQ. FPKM is designed for
within-sample gene comparisons and has actually fallen out of favor since the normalized
gene values it produces do not add up to one million exactly. In practice, however, the
deviation from one million is not dramatic and it often works well enough. Given that
normalizing such a large number of samples is challenging, here I will use the FPKM-UQ data.

* __Transcriptome profiling__ > __RNA-seq__ > __HTSeq - Counts__ (11'093 files; 2.8 Gb)

```console
$ mkdir /mnt/dataA/TCGA/RNA-seq_HTSeq_counts
$ sudo /opt/gdc-client download \
  -d /mnt/dataA/TCGA/raw/RNA-seq_HTSeq_counts/ \
  -m data/gdc_manifest.2019-08-21.txt
```

* __Transcriptome profiling__ > __RNA-seq__ > __HTSeq - FPKM-UQ__ (11'093 files; 5.78 Gb)

```console
$ mkdir /mnt/dataA/TCGA/raw/RNA-seq_FPKM-UQ
$ sudo /opt/gdc-client download \
  -d /mnt/dataA/TCGA/raw/RNA-seq_FPKM-UQ/ \
  -m data/gdc_manifest.2019-08-23.txt
```

A description of the mRNA expression data analysis pipeline can be found in the
[docs page](https://docs.gdc.cancer.gov/Data/Bioinformatics_Pipelines/Expression_mRNA_Pipeline/).

### miRNA expression

The data are provided in tables including both read counts and counts per million mapped miRNA (RPM).
RPM should be appropriate for the current project.

See the [docs page](https://docs.gdc.cancer.gov/Data/Bioinformatics_Pipelines/miRNA_Pipeline/)
for a description of the analaysis pipeline.

* __Transcriptome profiling__ > __miRNA Expression Quantification__ (11'082 files; 557.23 Mb)

```console
$ mkdir /mnt/dataA/TCGA/raw/miRNA-seq
$ sudo /opt/gdc-client download \
  -d /mnt/dataA/TCGA/raw/miRNA-seq/ \
  -m data/gdc_manifest.2019-08-22.txt
```

### DNA methylation

The data are provided in tables of array results of the level of methylation at known CpG
sites. They include unique ids for the array probes and methylation Beta values, representing
the ratio between the methylated array intensity and total array intensity (falls between 0,
lower levels of methylation, and 1, higher levels of methylation).

See the [docs page](https://docs.gdc.cancer.gov/Data/Bioinformatics_Pipelines/Methylation_LO_Pipeline/)
for a description of the analaysis pipeline.

* __Transcriptome profiling__ > __Methylation Array__ (12'359 files; 1.4 Tb)

```console
$ mkdir /mnt/dataA/TCGA/raw/Methylation
$ sudo /opt/gdc-client download \
  -d /mnt/dataA/TCGA/raw/Methylation/ \
  -m data/gdc_manifest.2019-09-09.txt
```

### Copy number variation

The data are provided in a table for each of the 33 cancer entities with a column
per patient and a row for each of 19'729 protein coding genes. Copy number variation (CNV)
values are represented as 0, 1, or -1 for each gene, corresponding to "neutral", "gain" or "loss".

See the [docs page](https://docs.gdc.cancer.gov/Data/Bioinformatics_Pipelines/CNV_Pipeline/)
for a description of the analaysis pipeline.

* __Copy Number Variation__ > __Gene Level Copy Number Scores__ (33 files; 474.29 Mb)

```console
$ mkdir /mnt/dataA/TCGA/raw/CNV
$ sudo /opt/gdc-client download \
  -d /mnt/dataA/TCGA/raw/CNV/ \
  -m data/gdc_manifest.2019-09-26.txt
```

### Whole-slide images (WSI)

The total data can be accessed as follows:

* __Slide image__ > __Diagnostic slide__ (11'766 files; 12.95 Tb)

```console
$ mkdir /net/data/Projects/imaging_genomics/TCGA_BRCA/diagnostic_slide/
$ /net/gdc-client download \
  -d /net/data/Projects/imaging_genomics/TCGA_BRCA/diagnostic_slide \
  -m /net/imaging_genomics/data/gdc_manifest.2019-05-24_Diagnostic_slide.txt
```
