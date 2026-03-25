# Awesome Biological Image Analysis [![Awesome](https://awesome.re/badge.svg)](https://awesome.re) with stars

<p align="center">
  <br>
    <img width="300" src="awesome-biological-image-analysis.svg" alt="Awesome Biological Image Analysis">
 <br>
 <br>
</p>

> Tools and resources for biological image analysis.

Biological image analysis aims to increase our understanding of biology through the use of various computational techniques and approaches to obtain valuable information from images.

## Contents

* [General image analysis software](#general-image-analysis-software)
* [Image processing and segmentation](#image-processing-and-segmentation)
* [Ecology](#ecology)
* [Neuroscience](#neuroscience)
* [Plant science](#plant-science)
* [Fluoresence in situ hybridization](#fluoresence-in-situ-hybridization)
* [Electron and super resolution microscopy](#electron-and-super-resolution-microscopy)
* [Image restoration and quality assessment](#image-restoration-and-quality-assessment)
* [Cell migration and particle tracking](#cell-migration-and-particle-tracking)
* [Pathology](#pathology)
* [Mycology](#mycology)
* [Microbiology](#microbiology)
* [Yeast imaging](#yeast-imaging)
* [Other](#other)
* [Publications](#publications)

## General image analysis software

* [OpenCV](https://github.com/opencv/opencv) ⭐ 86,776 | 🐛 2,748 | 🌐 C++ | 📅 2026-03-23 - Open source computer vision and machine learning software library.
* [Scikit-image](https://github.com/scikit-image/scikit-image) ⭐ 6,487 | 🐛 883 | 🌐 Python | 📅 2026-03-24 - Collection of algorithms for image processing.
* [Napari](https://github.com/napari/napari) ⭐ 2,612 | 🐛 1,159 | 🌐 Python | 📅 2026-03-24 - Fast, interactive, multi-dimensional image viewer for Python.
* [3D Slicer](https://github.com/Slicer/Slicer) ⭐ 2,387 | 🐛 666 | 🌐 C++ | 📅 2026-03-24 - Free, open source and multi-platform software package widely used for medical, biomedical, and related imaging research.
* [ImageJ2](https://github.com/imagej/imagej2) ⭐ 1,356 | 🐛 133 | 🌐 Java | 📅 2025-08-12 - A Rewrite of ImageJ for multidimensional image data, with a focus on scientific imaging.
* [ImagePy](https://github.com/Image-Py/imagepy) ⭐ 1,354 | 🐛 58 | 🌐 Python | 📅 2024-02-21 - Open source image processing framework written in Python.
* [CellProfiler](https://github.com/CellProfiler/CellProfiler) ⭐ 1,098 | 🐛 313 | 🌐 Python | 📅 2026-03-24 - Open-source software helping biologists turn images into cell measurements.
* [Fiji](https://github.com/fiji/fiji) ⭐ 942 | 🐛 132 | 🌐 Shell | 📅 2025-12-26 - A "batteries-included" distribution of ImageJ — a popular, free scientific image processing application.
* [ImageJ](https://github.com/imagej/ImageJ) ⭐ 744 | 🐛 84 | 🌐 Java | 📅 2026-03-22 - Public domain software for processing and analyzing scientific images.
* [Ilastik](https://github.com/ilastik/ilastik) ⭐ 395 | 🐛 580 | 🌐 Python | 📅 2026-03-20 - Simple, user-friendly tool for interactive image classification, segmentation and analysis.
* [Cell-ACDC](https://github.com/SchmollerLab/Cell_ACDC) ⭐ 193 | 🐛 165 | 🌐 Python | 📅 2026-03-24 - A GUI-based Python framework for segmentation, tracking, cell cycle annotations and quantification of microscopy data.
* [CellProfiler Analyst](https://github.com/CellProfiler/CellProfiler-Analyst) ⭐ 166 | 🐛 72 | 🌐 Python | 📅 2025-07-21 - Open-source software for exploring and analyzing large, high-dimensional image-derived data.
* [PYME](https://github.com/python-microscopy/python-microscopy) ⭐ 97 | 🐛 164 | 🌐 Python | 📅 2026-03-06 - Open-source application suite for light microscopy acquisition, data storage, visualization, and analysis.
* [Flika](https://github.com/flika-org/flika) ⭐ 25 | 🐛 6 | 🌐 Python | 📅 2026-03-04 - An interactive image processing program for biologists written in Python.
* [BiaPy](https://biapyx.github.io/) - Open source ready-to-use all-in-one library that provides deep-learning workflows for a large variety of bioimage analysis tasks.
* [Icy](https://github.com/Icy-imaging) - Open community platform for bioimage informatics, providing software resources to visualize, annotate and quantify bioimaging data.

## Image processing and segmentation

* [Cellpose](https://github.com/MouseLand/cellpose) ⭐ 2,124 | 🐛 85 | 🌐 Python | 📅 2026-03-20 - A generalist algorithm for cell and nucleus segmentation.
* [StarDist](https://github.com/stardist/stardist) ⭐ 1,190 | 🐛 66 | 🌐 Python | 📅 2026-02-14 - Object detection with Star-convex shapes.
* [HoVer-Net](https://github.com/vqdang/hover_net) ⭐ 700 | 🐛 68 | 🌐 Python | 📅 2023-10-27 - A multi-branch network for nuclear instance segmentation and classification with pre-trained weights.
* [MicroSAM](https://github.com/computational-cell-analytics/micro-sam) ⭐ 666 | 🐛 80 | 🌐 Jupyter Notebook | 📅 2026-03-24 - Tools for segmentation and tracking in microscopy build on top of SegmentAnything. Segment and track objects in microscopy images interactively.
* [Squidpy](https://github.com/scverse/squidpy) ⭐ 555 | 🐛 88 | 🌐 Python | 📅 2026-03-24 - Python framework that brings together tools from omics and image analysis to enable scalable description of spatial molecular data, such as transcriptome or multivariate proteins.
* [DeepSlide](https://github.com/BMIRDS/deepslide) ⭐ 515 | 🐛 0 | 🌐 Python | 📅 2024-06-07 - A sliding window framework for classification of high resolution microscopy images.
* [DeepCell](https://github.com/vanvalenlab/deepcell-tf) ⭐ 475 | 🐛 55 | 🌐 Python | 📅 2024-08-30 - Deep learning library for single cell analysis.
* [Suite2p](https://github.com/MouseLand/suite2p) ⭐ 440 | 🐛 55 | 🌐 Python | 📅 2026-03-21 - Pipeline for processing two-photon calcium imaging data.
* [PyImSegm](https://github.com/Borda/pyImSegm) ⚠️ Archived - Image segmentation - general superpixel segmentation and center detection and region growing.
* [CellVit++](https://github.com/TIO-IKIM/CellViT) ⭐ 367 | 🐛 20 | 🌐 Python | 📅 2025-07-23 - A framework for lightweight cell segmentation model training and inference.
* [AtomAI](https://github.com/pycroscopy/atomai) ⭐ 225 | 🐛 11 | 🌐 Python | 📅 2025-06-24 - PyTorch-based package for deep/machine learning analysis of microscopy data.
* [CellSAM](https://github.com/vanvalenlab/cellSAM) ⭐ 184 | 🐛 42 | 🌐 Python | 📅 2025-11-11 - A foundation model for cell segmentation trained on a diverse range of cells and data types.
* [Proseg](https://github.com/dcjones/proseg) ⭐ 160 | 🐛 74 | 🌐 Rust | 📅 2026-03-13 : A cell segmentation method for in situ spatial transcriptomics.
* [Trainable Weka Segmentation](https://github.com/fiji/Trainable_Segmentation) ⭐ 122 | 🐛 29 | 🌐 Java | 📅 2024-09-28 - Fiji plugin and library that combines a collection of machine learning algorithms with a set of selected image features to produce pixel-based segmentations.
* [MorpholibJ](https://github.com/ijpb/MorphoLibJ) ⭐ 118 | 🐛 24 | 🌐 Java | 📅 2026-02-19 - Collection of mathematical morphology methods and plugins for ImageJ.
* [Ark-Analysis](https://github.com/angelolab/ark-analysis) ⭐ 104 | 🐛 56 | 🌐 Jupyter Notebook | 📅 2026-01-07 - A pipeline toolbox for analyzing multiplexed imaging data.
* [Nellie](https://github.com/aelefebv/nellie) ⭐ 91 | 🐛 4 | 🌐 Python | 📅 2026-01-14 - Automated organelle segmentation, tracking, and hierarchical feature extraction in 2D/3D live-cell microscopy.
* [EBImage](https://github.com/aoles/EBImage) ⭐ 76 | 🐛 37 | 🌐 R | 📅 2024-10-14 - Image processing toolbox for R.
* [SplineDist](https://github.com/uhlmanngroup/splinedist) ⭐ 72 | 🐛 6 | 🌐 Jupyter Notebook | 📅 2024-09-29 - Object detection with spline curves.
* [MAPS](https://github.com/mahmoodlab/MAPS) ⭐ 64 | 🐛 4 | 🌐 Jupyter Notebook | 📅 2024-03-14 - MAPS (Machine learning for Analysis of Proteomics in Spatial biology) is a machine learning approach facilitating rapid and precise cell type identification with human-level accuracy from spatial proteomics data.
* [GPim](https://github.com/ziatdinovmax/GPim) ⭐ 57 | 🐛 1 | 🌐 Python | 📅 2023-11-24 - Gaussian processes and Bayesian optimization for images and hyperspectral data.
* [PartSeg](https://github.com/4DNucleome/PartSeg) ⭐ 36 | 🐛 17 | 🌐 Python | 📅 2026-03-24 - A GUI and a library for segmentation algorithms.
* [Cellshape](https://github.com/Sentinal4D/cellshape) ⭐ 31 | 🐛 5 | 🌐 Python | 📅 2025-10-08 - 3D single-cell shape analysis of cancer cells using geometric deep learning.
* [SyMBac](https://github.com/georgeoshardo/SyMBac) ⭐ 22 | 🐛 12 | 🌐 Jupyter Notebook | 📅 2026-03-10 - Accurate segmentation of bacterial microscope images using synthetically generated image data.
* [Classpose](https://github.com/sohmandal/classpose) ⭐ 20 | 🐛 3 | 🌐 Python | 📅 2026-03-24 - A foundation model-driven whole slide image-scale cell phenotyping method with QuPath integration.
* [FlashDeconv](https://github.com/cafferychen777/flashdeconv) ⭐ 14 | 🐛 0 | 🌐 Python | 📅 2026-03-16 - High-performance spatial transcriptomics deconvolution for cell type mapping using structure-preserving randomized sketching.
* [Salem²](https://github.com/JackieZhai/SALEM2) ⚠️ Archived - Segment Anything in Light and Electron Microscopy via Membrane Guidance.
* [CLIJ2](https://clij.github.io/) - GPU-accelerated image processing library for ImageJ/Fiji, Icy, MATLAB and Java.
* [HistoPLUS](https://huggingface.co/Owkin-Bioptimus/histoplus) - Pre-trained model for cell nuclei segmentation and classification in histology images.

## Ecology

* [PAT-GEOM](http://ianzwchan.com/my-research/pat-geom/) - A software package for the analysis of animal colour pattern.
* [ThermImageJ](https://github.com/gtatters/ThermImageJ) ⭐ 54 | 🐛 1 | 🌐 ImageJ Macro | 📅 2024-05-28 - ImageJ functions and macros for working with thermal image files.

## Neuroscience

* [Neuroglancer](https://github.com/google/neuroglancer/) ⭐ 1,300 | 🐛 211 | 🌐 TypeScript | 📅 2026-03-25 - WebGL-based viewer for volumetric data.
* [CaImAn](https://github.com/flatironinstitute/CaImAn) ⭐ 720 | 🐛 99 | 🌐 Python | 📅 2026-02-19 - Computational toolbox for large scale Calcium Imaging Analysis.
* [Brainrender](https://github.com/brainglobe/brainrender) ⭐ 635 | 🐛 14 | 🌐 Python | 📅 2026-03-13 - Python package for the visualization of three dimensional neuro-anatomical data.
* [Cellfinder](https://github.com/brainglobe/cellfinder) ⭐ 227 | 🐛 83 | 🌐 Python | 📅 2026-03-12 - Automated 3D cell detection and registration of whole-brain images.
* [PyTorch Connectomics](https://github.com/zudi-lin/pytorch_connectomics) ⭐ 191 | 🐛 0 | 🌐 Python | 📅 2026-03-20 - Deep learning framework for automatic and semi-automatic annotation of connectomics datasets, powered by PyTorch.
* [BG-atlasAPI](https://github.com/brainglobe/bg-atlasapi) ⭐ 179 | 🐛 219 | 🌐 Python | 📅 2026-03-24 - A lightweight Python module to interact with atlases for systems neuroscience.
* [CloudVolume](https://github.com/seung-lab/cloud-volume) ⭐ 170 | 🐛 91 | 🌐 Python | 📅 2026-03-17 - Read and write Neuroglancer datasets programmatically.
* [Brainreg](https://github.com/brainglobe/brainreg) ⭐ 146 | 🐛 17 | 🌐 Python | 📅 2026-03-03 - Automated 3D brain registration with support for multiple species and atlases.
* [AxonDeepSeg](https://github.com/axondeepseg/axondeepseg) ⭐ 124 | 🐛 46 | 🌐 Python | 📅 2026-03-23 - Segment axon and myelin from microscopy data using deep learning.
* [Wholebrain](https://github.com/tractatus/wholebrain) ⭐ 93 | 🐛 29 | 🌐 C++ | 📅 2021-07-16 - Automated cell detection and registration of whole-brain images with plot of cell counts per region and Hemishpere.
* [NeuroAnatomy Toolbox](https://github.com/natverse/nat) ⭐ 77 | 🐛 58 | 🌐 R | 📅 2026-03-07 - R package for the (3D) visualisation and analysis of biological image data, especially tracings of single neurons.
* [SNT](https://github.com/morphonets/SNT/) ⭐ 57 | 🐛 12 | 🌐 Java | 📅 2026-03-25 - ImageJ framework for semi-automated tracing and analysis of neurons.
* [TrailMap](https://github.com/AlbertPun/TRAILMAP/) ⭐ 46 | 🐛 5 | 🌐 Python | 📅 2020-05-21 - Software package to extract axonal data from cleared brains.
* [ZVQ - Zebrafish Vascular Quantification](https://github.com/ElisabethKugler/ZFVascularQuantification) ⭐ 11 | 🐛 1 | 🌐 ImageJ Macro | 📅 2023-05-24 - Image analysis pipeline to perform 3D quantification of the total or regional zebrafish brain vasculature using the image analysis software Fiji.
* [NeuronJ](https://imagescience.org/meijering/software/neuronj/) - An ImageJ plugin for neurite tracing and analysis.
* [Panda](https://www.nitrc.org/projects/panda/) - Pipeline for Analyzing braiN Diffusion imAges: A MATLAB toolbox for pipeline processing of diffusion MRI images.

## Plant science

* [PlantCV](https://github.com/danforthcenter/plantcv) ⭐ 772 | 🐛 112 | 🌐 Python | 📅 2026-03-24 - Open-source image analysis software package targeted for plant phenotyping.
* [PlantSeg](https://github.com/hci-unihd/plant-seg) ⭐ 124 | 🐛 34 | 🌐 Python | 📅 2026-03-24 - Tool for cell instance aware segmentation in densely packed 3D volumetric images.
* [RootPainter](https://github.com/Abe404/root_painter) ⭐ 75 | 🐛 40 | 🌐 Python | 📅 2026-03-19 - Deep learning segmentation of biological images with corrective annotation.
* [Aradeepopsis](https://github.com/Gregor-Mendel-Institute/aradeepopsis) ⭐ 49 | 🐛 3 | 🌐 Nextflow | 📅 2025-05-26 - A versatile, fully open-source pipeline to extract phenotypic measurements from plant images.
* [Rhizovision Explorer](https://github.com/rootphenomicslab/RhizoVisionExplorer) ⭐ 27 | 🐛 7 | 🌐 C++ | 📅 2025-10-21 - Free and open-source software developed for estimating root traits from images acquired from a flatbed scanner or camera.
* [PhenotyperCV](https://github.com/jberry47/ddpsc_phenotypercv) ⭐ 4 | 🐛 0 | 🌐 C++ | 📅 2022-04-28 - Header-only C++11 library using OpenCV for high-throughput image-based plant phenotyping.
* [LeafByte](https://zoegp.science/leafbyte) - Free and open source mobile app for measuring herbivory quickly and accurately.
* [PaCeQuant](https://mitobo.informatik.uni-halle.de/index.php/Applications/PaCeQuant) - An ImageJ-based tool which provides a fully automatic image analysis workflow for PC shape quantification.
* [RhizoTrak](https://prbio-hub.github.io/rhizoTrak/) - Open source tool for flexible and efficient manual annotation of complex time-series minirhizotron images.

## Fluoresence in situ hybridization

* [Spotiflow](https://github.com/weigertlab/spotiflow) ⭐ 120 | 🐛 6 | 🌐 Python | 📅 2026-02-05 - A deep learning-based, threshold-agnostic, and subpixel-accurate spot detection method developed for spatial transcriptomics workflows.
* [Big-fish](https://github.com/fish-quant/big-fish) ⭐ 74 | 🐛 19 | 🌐 Python | 📅 2023-10-31 - Python package for the analysis of smFISH images.
* [RS-FISH](https://github.com/PreibischLab/RS-FISH) ⭐ 50 | 🐛 10 | 🌐 Java | 📅 2025-02-04 - Fiji plugin to detect FISH spots in 2D/3D images which scales to very large images.
* [DypFISH](https://github.com/cbib/dypfish) ⭐ 2 | 🐛 13 | 🌐 Python | 📅 2023-10-03 - Python library for spatial analysis of smFISH images.
* [TissUUmaps](https://tissuumaps.github.io/) - Visualizer of NGS data, plot millions of points and interact, gate, export. ISS rounds and base visualization.

## Electron and super resolution microscopy

* [Picasso](https://github.com/jungmannlab/picasso) ⭐ 141 | 🐛 18 | 🌐 Python | 📅 2026-03-24 - A collection of tools for painting super-resolution images.
* [ThunderSTORM](https://github.com/zitmen/thunderstorm) ⚠️ Archived - A comprehensive ImageJ plugin for SMLM data analysis and super-resolution imaging.
* [SMAP](https://github.com/jries/SMAP) ⭐ 81 | 🐛 14 | 🌐 MATLAB | 📅 2026-03-15 - A modular super-resolution microscopy analysis platform for SMLM data.
* [ASI\_MTF](https://github.com/emx77/ASI_MTF) ⭐ 21 | 🐛 5 | 🌐 ImageJ Macro | 📅 2025-01-07 - ImageJ macro to calculate the modulation transfer function (MTF) based on a knife edge (or slanted edge) measurement.
* [Em-scalebartools](https://github.com/lukmuk/em-scalebartools) ⭐ 13 | 🐛 3 | 🌐 ImageJ Macro | 📅 2025-04-23 - Fiji/ImageJ macros to quickly add a scale bar to an (electron microscopy) image.
* [Empanada](https://github.com/volume-em/empanada) ⭐ 2 | 🐛 6 | 🌐 Python | 📅 2023-02-25 - Panoptic segmentation algorithms for 2D and 3D electron microscopy images.

## Image restoration and quality assessment

* [Image Quality](https://github.com/ocampor/image-quality) ⭐ 433 | 🐛 21 | 🌐 Python | 📅 2024-02-05 - Open source software library for Image Quality Assessment (IQA).
* [CSBDeep](https://github.com/CSBDeep/CSBDeep) ⭐ 329 | 🐛 26 | 🌐 Python | 📅 2025-12-08 - A deep learning toolbox for microscopy image restoration and analysis.
* [LLSpy](https://github.com/tlambert03/LLSpy) ⭐ 30 | 🐛 5 | 🌐 Python | 📅 2026-03-02 - Python library to facilitate lattice light sheet data processing.
* [NCS](https://github.com/HuanglabPurdue/NCS) ⭐ 28 | 🐛 0 | 🌐 Python | 📅 2020-01-15 - Noise correction algorithm for sCMOS cameras.
* [Ijp-color](https://github.com/ij-plugins/ijp-color) ⭐ 25 | 🐛 8 | 🌐 Scala | 📅 2026-03-23 - Plugins for ImageJ - color space conversions and color calibration.

## Cell migration and particle tracking

* [TrackMate](https://github.com/fiji/TrackMate) ⭐ 232 | 🐛 16 | 🌐 Java | 📅 2026-02-09 - User-friendly interface that allows for performing tracking, data visualization, editing results and track analysis in a convenient way.
* [Usiigaci](https://github.com/oist/usiigaci) ⭐ 205 | 🐛 13 | 🌐 Jupyter Notebook | 📅 2020-09-15 - Stain-free cell tracking in phase contrast microscopy enabled by supervised machine learning.
* [Ultrack](https://github.com/royerlab/ultrack) ⭐ 174 | 🐛 69 | 🌐 Python | 📅 2026-03-20 - Versatile cell tracking method for 2D, 3D, and multichannel timelapses, overcoming segmentation challenges in complex tissues.
* [TrackMateR](https://github.com/quantixed/TrackMateR) ⭐ 17 | 🐛 0 | 🌐 R | 📅 2026-03-07 - R package to analyze cell migration and particle tracking experiments using outputs from TrackMate.
* [CellMigration](https://github.com/quantixed/CellMigration) ⭐ 9 | 🐛 0 | 🌐 IGOR Pro | 📅 2021-08-20 - Analysis of 2D cell migration in Igor.
* [QuimP](https://github.com/CellDynamics/QuimP) ⭐ 9 | 🐛 33 | 🌐 Java | 📅 2023-11-27 - Software for tracking cellular shape changes and dynamic distributions of fluorescent reporters at the cell membrane.
* [Trackpy](https://soft-matter.github.io/trackpy) - Fast and Flexible Particle-Tracking Toolkit.
* [TracX](https://gitlab.com/csb.ethz/tracx) - MATLAB generic toolbox for cell tracking from various microscopy image modalities such as Bright-field (BF), phase contrast (PhC) or fluorescence (FL) with an automated track quality assessment in
  absence of a ground truth.
* [TraJClassifier](https://imagej.net/plugins/trajclassifier) - Fiji plugin that loads trajectories from TrackMate, characterizes them using TraJ and classifiies them into normal diffusion, subdiffusion, confined diffusion and directed/active motion by a random forest approach (through Renjin).

## Pathology

* [PathML](https://github.com/Dana-Farber-AIOS/pathml) ⭐ 445 | 🐛 50 | 🌐 Python | 📅 2026-03-24 - An open-source toolkit for computational pathology and machine learning.
* [FastPathology](https://github.com/AICAN-Research/FAST-Pathology) ⭐ 141 | 🐛 26 | 🌐 C++ | 📅 2024-06-14 - Open-source software for deep learning-based digital pathology.
* [PAQUO](https://github.com/bayer-science-for-a-better-life/paquo) ⭐ 133 | 🐛 34 | 🌐 Python | 📅 2026-03-14 - A library for interacting with QuPath from Python.
* [Minerva](https://github.com/labsyspharm/minerva-story) ⭐ 56 | 🐛 20 | 🌐 HTML | 📅 2024-05-08 - Image viewer designed specifically to make it easy for non-expert users to interact with complex tissue images.
* [HistoClean](https://github.com/HistoCleanQUB/HistoClean) ⭐ 32 | 🐛 3 | 🌐 Python | 📅 2022-02-18 - Tool for the preprocessing and augmentation of images used in deep learning models.
* [Orbit](http://www.orbit.bio) - A versatile image analysis software for biological image-based quantification using machine learning, especially for whole slide imaging.
* [QuPath](https://qupath.github.io/) - Open source software for digital pathology image analysis.

## Mycology

* [DeepMushroom](https://github.com/Olament/DeepMushroom) ⭐ 70 | 🐛 1 | 🌐 Go | 📅 2020-02-06 - Image classification of fungus using ResNet.
* [Fungal Feature Tracker (FFT)](https://github.com/hsueh-lab/FFT) ⭐ 13 | 🐛 3 | 🌐 HTML | 📅 2022-07-29 - Tool to quantitatively characterize morphology and growth of filamentous fungi.

## Microbiology

* [BactMap](https://github.com/vrrenske/BactMAP) ⭐ 4 | 🐛 5 | 🌐 R | 📅 2023-10-24 - A command-line based R package that allows researchers to transform cell segmentation and spot detection data generated by different programs into various plots.
* [BacStalk](https://drescherlab.org/data/bacstalk/docs/index.html) - Interactive and user-friendly image analysis software tool to investigate the cell biology of common used bacterial species.
* [BiofilmQ](https://drescherlab.org/data/biofilmQ/docs/) - Advanced biofilm analysis tool for quantifying the properties of cells inside large 3-dimensional biofilm communities in space and time.

## Yeast imaging

* [YeaZ](https://github.com/lpbsscientist/YeaZ-GUI) ⭐ 29 | 🐛 2 | 🌐 Python | 📅 2026-01-14 - An interactive tool for segmenting yeast cells using deep learning.
* [htsimaging](https://github.com/rraadd88/htsimaging) ⭐ 7 | 🐛 0 | 🌐 Python | 📅 2023-12-22 - Python package for high-throughput single-cell imaging analysis.
* [BABY](https://git.ecdf.ed.ac.uk/swain-lab/baby/) - An image processing pipeline for accurate single-cell growth estimation of
  budding cells from bright-field stacks.
* [YeastMate](https://yeastmate.readthedocs.io/en/latest/) - Neural network-assisted segmentation of mating and budding events in S. cerevisiae.

## Other

* [SimpleITK](https://github.com/SimpleITK/SimpleITK) ⭐ 1,050 | 🐛 75 | 🌐 SWIG | 📅 2026-03-24 - Open-source multi-dimensional image analysis in Python, R, Java, C#, Lua, Ruby, TCL and C++.
* [ZeroCostDL4Mic](https://github.com/HenriquesLab/ZeroCostDL4Mic) ⭐ 634 | 🐛 59 | 🌐 Jupyter Notebook | 📅 2026-03-19 - Google Colab to develop a free and open-source toolbox for deep-Learning in microscopy.
* [Neurite](https://github.com/adalca/neurite) ⭐ 375 | 🐛 20 | 🌐 Python | 📅 2026-02-20 - Neural networks toolbox focused on medical image analysis.
* [AICSImageIO](https://github.com/AllenCellModeling/aicsimageio) ⚠️ Archived - Image reading, metadata conversion, and image writing for nicroscopy images in Python.
* [CaPTk](https://github.com/CBICA/CaPTk) ⭐ 197 | 🐛 174 | 🌐 C++ | 📅 2023-12-09 - Cancer Imaging Phenomics Toolkit: A software platform to perform image analysis and predictive modeling tasks.
* [OAD](https://github.com/zeiss-microscopy/OAD) ⭐ 163 | 🐛 0 | 🌐 Jupyter Notebook | 📅 2026-03-03 - Collection of tools and scripts useful to automate microscopy workflows in ZEN Blue using Python and Open Application Development tools.
* [Pycytominer](https://github.com/cytomining/pycytominer) ⭐ 135 | 🐛 66 | 🌐 Python | 📅 2026-03-17 - Data processing functions for profiling perturbations.
* [Pyotf](https://github.com/david-hoffman/pyotf) ⭐ 82 | 🐛 7 | 🌐 Python | 📅 2024-04-08 - A simulation software package for modelling optical transfer functions (OTF)/point spread functions (PSF) of optical microscopes written in Python.
* [Nd2reader](https://github.com/Open-Science-Tools/nd2reader) ⭐ 49 | 🐛 29 | 🌐 Python | 📅 2025-01-15 - A pure-Python package that reads images produced by NIS Elements 4.0+.
* [ZetaStitcher](https://github.com/lens-biophotonics/ZetaStitcher) ⭐ 46 | 🐛 5 | 🌐 Python | 📅 2025-11-13 - Tool designed to stitch large volumetric images such as those produced by light-sheet fluorescence microscopes.
* [Napari-aicsimageio](https://github.com/AllenCellModeling/napari-aicsimageio) ⚠️ Archived - Multiple file format reading directly into napari using pure Python.
* [NEFI2](https://github.com/05dirnbe/nefi) ⭐ 39 | 🐛 26 | 🌐 Python | 📅 2021-05-13 - Python tool created to extract networks from images.
* [Quanfima](https://github.com/rshkarin/quanfima) ⭐ 30 | 🐛 6 | 🌐 Python | 📅 2024-02-03 - Quantitative Analysis of Fibrous Materials: A collection of useful functions for morphological analysis and visualization of 2D/3D data from various areas of material science.
* [ColiCoords](https://github.com/Jhsmit/ColiCoords) ⭐ 26 | 🐛 47 | 🌐 Python | 📅 2021-07-22 - Python project for analysis of fluorescence microscopy data from rodlike cells.
* [BoneJ](https://github.com/bonej-org/BoneJ2) ⭐ 25 | 🐛 35 | 🌐 Java | 📅 2026-03-17 - Collection of Fiji/ImageJ plug-ins for skeletal biology.
* [Z-stack Depth Color Code](https://github.com/ekatrukha/ZstackDepthColorCode) ⭐ 25 | 🐛 0 | 🌐 Java | 📅 2025-11-06 - ImageJ/Fiji plugin to colorcode Z-stacks/hyperstacks.
* [MIA](https://github.com/mianalysis/mia) ⭐ 16 | 🐛 45 | 🌐 Java | 📅 2026-03-17 - Fiji plugin which provides a modular framework for assembling image and object analysis workflows.
* [CompactionAnalyzer](https://github.com/davidbhr/CompactionAnalyzer) ⭐ 10 | 🐛 0 | 🌐 Python | 📅 2025-01-17 - Python package to quantify the tissue compaction (as a measure of the contractile strength) generated by cells or multicellular spheroids that are embedded in fiber materials.
* [Cytominer-database](https://github.com/cytomining/cytominer-database) ⭐ 10 | 🐛 14 | 🌐 Python | 📅 2024-10-10 - Command-line tools for organizing measurements extracted from images.
* [DetecDiv](https://github.com/gcharvin/DetecDiv) ⭐ 8 | 🐛 2 | 🌐 MATLAB | 📅 2026-03-21 - Comprehensive set of tools to analyze time microscopy images using deep learning methods.
* [XitoSBML](https://github.com/spatialsimulator/XitoSBML) ⭐ 7 | 🐛 2 | 🌐 HTML | 📅 2021-11-17 - ImageJ plugin which creates a Spatial SBML model from segmented images.
* [Biobeam](https://maweigert.github.io/biobeam) - Open source software package that is designed to provide fast methods for in-silico optical experiments with an emphasize on image formation in biological tissues.
* [MorphoGraphX](https://morphographx.org) - Open source application for the visualization and analysis of 4D biological datasets.
* [PyScratch](https://bitbucket.org/vladgaal/pyscratch_public.git/src) - Open source tool that autonomously performs quantitative analysis of in vitro scratch assays.
* [Vaa3D](https://alleninstitute.org/what-we-do/brain-science/research/products-tools/vaa3d/) - Open-source software for 3D/4D/5D image visualization and analysis.

## Publications

* [A Hitchhiker's guide through the bio-image analysis software universe](https://febs.onlinelibrary.wiley.com/doi/10.1002/1873-3468.14451) - An article presenting a curated guide and glossary of bio-image analysis terms and tools.
* [Biological imaging software tools](https://dx.doi.org/10.1038%2Fnmeth.2084) - The steps of biological image analysis and the appropriate tools for each step.
* [Data-analysis strategies for image-based cell profiling](https://doi.org/10.1038/nmeth.4397) - In-detail explanations of image analysis pipelines.
* [Large-scale image-based screening and profiling of cellular phenotypes](https://onlinelibrary.wiley.com/doi/10.1002/cyto.a.22909) - A workflow for phenotype extraction from high throughput imaging experiments.
* [Workflow and metrics for image quality control in large-scale high-content screens](https://linkinghub.elsevier.com/retrieve/pii/S2472555222075943) - Approaches for quality control in high-content imaging screens.

## Footnotes

### Similar lists and repositories

* [OpenMicroscopy](https://github.com/HohlbeinLab/OpenMicroscopy) ⭐ 174 | 🐛 0 | 📅 2026-03-19 - Non-comprehensive list of projects and resources related to open microscopy.
* [Cytodata](https://github.com/cytodata/awesome-cytodata) ⭐ 90 | 🐛 4 | 📅 2023-11-15 - A curated list of awesome cytodata resources.
* [BIII](https://biii.eu) - Repository of bioimage analysis tools.
* [Bio-image Analysis Notebooks](https://haesleinhuepf.github.io/BioImageAnalysisNotebooks/intro.html) - Notebooks for bioimage analysis in Python.
* [Bioimaging Guide](https://www.bioimagingguide.org) - Microscopy for beginners reference guide.
* [Napari hub](https://www.napari-hub.org) - Collection of napari plugins.
